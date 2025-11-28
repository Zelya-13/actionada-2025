import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skforecast.recursive import ForecasterRecursive


class ATMCashForecastPipeline:
    """
    Финальный пайплайн для прогнозирования спроса на наличные на 14 дней
    """

    def __init__(self, model=None, lags=14):
        self.pipeline = None
        self.model = model
        self.lags = lags
        self.forecasters = {}
        self.label_encoders = {}

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def validate_input_data(self, df):
        """Проверка целостности входных данных"""
        required_columns = [
            "ATM_ID",
            "Date",
            "Total_Withdrawals",
            "Total_Deposits",
            "Location_Type",
            "Holiday_Flag",
            "Weather_Condition",
            "Cash_Demand_Next_Day",
        ]

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют колонки: {missing_columns}")

        self.logger.info("Валидация данных пройдена")
        return True

    def encode_categorical_features(self, df):
        """Кодирование категориальных признаков в числовые"""
        df_encoded = df.copy()

        categorical_columns = ["Location_Type", "Weather_Condition", "ATM_ID"]

        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Обучаем на ВСЕХ данных чтобы охватить все категории
                    self.label_encoders[col].fit(df_encoded[col])

                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])

        return df_encoded

    def create_features(self, df):
        """Создание признаков для временного ряда"""
        features_df = df.copy()

        aggregation_strategy = {
            # Столбцы, которые одинаковы в течение дня - берем первое значение
            "Location_Type": "first",
            "Holiday_Flag": "first",
            "Special_Event_Flag": "first",
            "Weather_Condition": "first",
            "Nearby_Competitor_ATMs": "first",
            "Cash_Demand_Next_Day": "first",
            "Previous_Day_Cash_Level": "first",
            # Столбцы, которые нужно суммировать за день
            "Total_Withdrawals": "sum",
            "Total_Deposits": "sum",
        }

        features_df = (
            features_df.groupby(["ATM_ID", "Date"])
            .agg(aggregation_strategy)
            .reset_index()
        )

        # Кодируем категориальные признаки
        features_df = self.encode_categorical_features(features_df)

        # Базовые фичи
        features_df["Date"] = pd.to_datetime(features_df["Date"])
        features_df["day_of_week"] = features_df["Date"].dt.dayofweek
        features_df["month"] = features_df["Date"].dt.month
        features_df["day_of_year"] = features_df["Date"].dt.dayofyear
        features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)

        # Фичи из операций
        features_df["withdrawal_to_deposit_ratio"] = features_df[
            "Total_Withdrawals"
        ] / (features_df["Total_Deposits"] + 1)
        features_df["total_operations"] = (
            features_df["Total_Withdrawals"] + features_df["Total_Deposits"]
        )

        return features_df

    def build_pipeline(self):
        """Построение пайплайна с созданием признаков"""
        self.pipeline = Pipeline(
            [
                (
                    "feature_creator",
                    FunctionTransformer(self.create_features, validate=False),
                )
            ]
        )
        return self.pipeline

    def train(self, df):
        """Обучение пайплайна и форкастера"""
        self.logger.info("Начало обучения пайплайна...")

        # Строим пайплайн для создания признаков
        pipeline = self.build_pipeline()
        featured_data = pipeline.fit_transform(df)
        self.pipeline = pipeline

        # Создаем и обучаем форкастер для каждого банкомата
        self.forecasters = {}

        for atm_id in df["ATM_ID"].unique():
            self.logger.info(f"Обучение форкастера для {atm_id}...")

            # Фильтруем данные по банкомату ДО создания признаков
            atm_raw_data = df[df["ATM_ID"] == atm_id].copy()

            # Проверяем, что данных достаточно
            if len(atm_raw_data) <= self.lags:
                self.logger.warning(
                    f"Недостаточно данных для банкомата {atm_id}: {len(atm_raw_data)} записей, требуется > {self.lags}"
                )
                continue

            # Создаем признаки для этого банкомата
            atm_data = self.pipeline.transform(atm_raw_data)
            atm_data = atm_data.sort_values("Date").reset_index(drop=True)

            # Проверяем, что после преобразований данные не потерялись
            if len(atm_data) == 0:
                self.logger.warning(
                    f"Нет данных после преобразований для банкомата {atm_id}"
                )
                continue

            # ПОДГОТОВКА ДАННЫХ ДЛЯ SKFORECAST (ТОЛЬКО ЦЕЛЕВАЯ ПЕРЕМЕННАЯ)
            y_series = pd.Series(
                atm_data["Cash_Demand_Next_Day"].values,
                index=pd.RangeIndex(start=0, stop=len(atm_data), name="index"),
            )

            forecaster = ForecasterRecursive(regressor=self.model, lags=self.lags)

            forecaster.fit(y=y_series)
            self.forecasters[atm_id] = forecaster

            self.logger.info(
                f"Форкастер для {atm_id} обучен на {len(atm_data)} записях"
            )

        self.logger.info(
            f"Обучение завершено. Создано {len(self.forecasters)} форкастеров"
        )
        return self.forecasters

    def predict_14_days(self, historical_data):
        """Прогноз на 14 дней вперед для всех банкоматов"""
        self.logger.info("Прогнозирование на 14 дней для всех банкоматов...")

        if not self.forecasters:
            raise ValueError("Форкастеры не обучены!")

        all_forecasts = {}

        for atm_id, forecaster in self.forecasters.items():
            try:
                # Прогноз на 14 дней
                predictions = forecaster.predict(steps=14)
                all_forecasts[atm_id] = [int(pred) for pred in predictions]

                self.logger.info(f"Прогноз для {atm_id} завершен")

            except Exception as e:
                self.logger.error(f"Ошибка прогноза для {atm_id}: {e}")
                all_forecasts[atm_id] = [0] * 14

        # Создаем DataFrame в нужном формате
        forecast_df = pd.DataFrame(all_forecasts)
        forecast_df.index = [f"Day_{i+1}" for i in range(14)]
        forecast_df = forecast_df.T

        self.logger.info("Прогноз для всех банкоматов завершен")
        return forecast_df

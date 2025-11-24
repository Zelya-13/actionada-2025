import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# 1. Функция для предобработки данных (заполнение пропусков + извлечение даты)
def preprocess_data(X):
    X_copy = X.copy()
    
    # Заполняем пропуски
    X_copy = X_copy.fillna('UNKNOWN')
    
    # Извлекаем признаки из даты
    X_copy['Date'] = pd.to_datetime(X_copy['Date'])
    X_copy['DayOfWeek'] = X_copy['Date'].dt.day_name()
    X_copy['Month'] = X_copy['Date'].dt.month_name()
    X_copy['IsWeekend'] = X_copy['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    return X_copy

# 2. Создаем пайплайн
def create_pipeline():
    
    # Предобработчик данных
    preprocessor = FunctionTransformer(preprocess_data)
    
    # ColumnTransformer для финальных преобразований
    feature_engineering = ColumnTransformer(
        transformers=[
            # Числовые признаки
            ('numeric', 'passthrough', ['Amount']),
            
            # Категориальные признаки (уже обработаны в preprocess_data)
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse=False), 
             ['Mode', 'Category', 'Subcategory', 'Income/Expense', 'Currency',
              'DayOfWeek', 'Month', 'IsWeekend']),
            
            # Текстовые поля (уже обработаны в preprocess_data)
            ('text', CountVectorizer(), 'Note'),
        ],
        remainder='drop'
    )
    
    # Финальный пайплайн
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('feature_engineering', feature_engineering)
    ])
    
    return pipeline

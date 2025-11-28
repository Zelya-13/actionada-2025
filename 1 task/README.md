# Data Processing Pipeline (1 задание Актионада 2025)

Пайплайн для обработки финансовых данных с пропусками, датами и текстом.

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Использование
from pipeline import create_pipeline
import pandas as pd

# Загрузка данных
df = pd.read_csv('Daily Household Transactions.csv')

# Создание и запуск пайплайна
pipeline = create_pipeline()
X_processed = pipeline.fit_transform(df)

print(f"Результат: {X_processed.shape}")
print(X_processed)

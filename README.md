# Data Processing Pipeline

Простой пайплайн для обработки финансовых данных с пропусками, датами и текстом.

## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Использование
from pipeline import create_pipeline
import pandas as pd

# Загрузка данных
df = pd.read_csv('your_data.csv')

# Создание и запуск пайплайна
pipeline = create_pipeline()
X_processed = pipeline.fit_transform(df)

print(f"Результат: {X_processed.shape}")

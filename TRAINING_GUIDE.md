# Руководство по обучению ML моделей

## 1. Локальное обучение (текущий метод)

### Запуск обучения:

```bash
# Обучить все модели (5m, 10m, 30m, 1h)
python -m moex_agent.advanced_train

# Или конкретный горизонт
python -m moex_agent.advanced_train --horizon 5m
```

### Параметры обучения:

```python
# В advanced_train.py
TRAIN_CONFIG = {
    "n_splits": 5,              # Walk-Forward splits
    "test_size": 0.2,           # 20% на тест
    "min_samples": 10000,       # Минимум свечей
    "horizons": ["5m", "10m", "30m", "1h"],
}

# Параметры модели
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_split": 50,
    "min_samples_leaf": 20,
}
```

---

## 2. Yandex DataSphere (облачное обучение)

### Преимущества:
- GPU для быстрого обучения
- Большие датасеты
- Автоматическое масштабирование
- MLflow для трекинга экспериментов

### Настройка:

1. **Создать проект в Yandex Cloud:**
   ```
   https://console.cloud.yandex.ru/
   → DataSphere → Создать проект
   ```

2. **Загрузить данные:**
   ```python
   # В DataSphere notebook
   import boto3

   # Подключение к Object Storage
   s3 = boto3.client(
       's3',
       endpoint_url='https://storage.yandexcloud.net',
       aws_access_key_id='YOUR_KEY',
       aws_secret_access_key='YOUR_SECRET'
   )

   # Загрузить данные
   s3.upload_file('data/moex_agent.sqlite', 'moex-bucket', 'data.sqlite')
   ```

3. **Notebook для обучения:**

```python
# DataSphere Notebook

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import mlflow

# Настройка MLflow
mlflow.set_tracking_uri("your_mlflow_uri")
mlflow.set_experiment("moex-agent-training")

# Загрузка данных
df = pd.read_sql("SELECT * FROM candles", conn)

# Подготовка фич
from features import build_feature_frame, FEATURE_COLS
features_df = build_feature_frame(df)

# Walk-Forward обучение
def train_walk_forward(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
        )

        with mlflow.start_run(nested=True):
            model.fit(X_train, y_train)

            # Метрики
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("fold", fold)

            scores.append(accuracy)
            models.append(model)

    return models[-1], np.mean(scores)

# Обучение для каждого горизонта
horizons = ["5m", "10m", "30m", "1h"]

for horizon in horizons:
    with mlflow.start_run(run_name=f"model_{horizon}"):
        # Подготовка целевой переменной
        y = create_labels(df, horizon)

        # Обучение
        model, score = train_walk_forward(features_df[FEATURE_COLS], y)

        # Логирование
        mlflow.log_param("horizon", horizon)
        mlflow.log_metric("mean_accuracy", score)

        # Сохранение модели
        joblib.dump(model, f"model_time_{horizon}.joblib")
        mlflow.log_artifact(f"model_time_{horizon}.joblib")

        print(f"{horizon}: accuracy={score:.3f}")
```

4. **Скачать обученные модели:**
   ```python
   # Скачать из Object Storage
   s3.download_file('moex-bucket', 'models/model_time_5m.joblib', 'models/model_time_5m.joblib')
   ```

---

## 3. Автоматическое переобучение

### Скрипт еженедельного переобучения:

```bash
# scripts/weekly_retrain.sh
#!/bin/bash

echo "Starting weekly retrain..."

# Обновить данные
python -m moex_agent.bootstrap --days 30

# Переобучить модели
python -m moex_agent.advanced_train

# Отправить отчёт в Telegram
python -c "
from moex_agent.telegram import send_telegram
send_telegram('✅ Модели переобучены!')
"
```

### Cron задача:

```bash
# Каждое воскресенье в 3:00
0 3 * * 0 /path/to/scripts/weekly_retrain.sh
```

---

## 4. Альтернативные модели

### 4.1 XGBoost (быстрее):

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
)
```

### 4.2 LightGBM (ещё быстрее):

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    num_leaves=31,
)
```

### 4.3 CatBoost (лучше с категориальными):

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    verbose=False,
)
```

### 4.4 Neural Network (PyTorch):

```python
import torch
import torch.nn as nn

class TradingNN(nn.Module):
    def __init__(self, input_size=29):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)

# Обучение
model = TradingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

---

## 5. Добавление новых фич

### В features.py добавить:

```python
# Новые индикаторы
def add_custom_features(df):
    # Supertrend
    df['supertrend'] = calculate_supertrend(df)

    # Ichimoku
    df['ichimoku_tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['ichimoku_kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2

    # Fear & Greed (если есть данные)
    # df['fear_greed'] = get_fear_greed_index()

    return df

# Обновить список фич
FEATURE_COLS = [
    # ... существующие 29 фич ...
    'supertrend',
    'ichimoku_tenkan',
    'ichimoku_kijun',
]
```

---

## 6. Yandex Cloud Secrets

В Replit добавить:

| Secret | Описание |
|--------|----------|
| `YANDEX_API_KEY` | API ключ Yandex Cloud |
| `YANDEX_FOLDER_ID` | ID папки в Yandex Cloud |
| `YANDEX_S3_KEY` | Ключ Object Storage |
| `YANDEX_S3_SECRET` | Секрет Object Storage |

---

## 7. Получение API ключей Yandex

1. **Yandex Cloud Console:**
   https://console.cloud.yandex.ru/

2. **Создать сервисный аккаунт:**
   IAM → Сервисные аккаунты → Создать

3. **Создать API ключ:**
   Сервисный аккаунт → Создать новый ключ → API-ключ

4. **Folder ID:**
   Находится в URL консоли: `console.cloud.yandex.ru/folders/YOUR_FOLDER_ID`

---

## Резюме

| Метод | Скорость | Стоимость | Сложность |
|-------|----------|-----------|-----------|
| Локальное | Средняя | Бесплатно | Низкая |
| DataSphere | Высокая | ~500₽/час GPU | Средняя |
| Replit | Низкая | Бесплатно/Pro | Низкая |

**Рекомендация:** Начните с локального обучения, затем переходите на DataSphere для больших экспериментов.

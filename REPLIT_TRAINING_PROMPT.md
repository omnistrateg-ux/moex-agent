# MOEX Agent — Промпт для обучения моделей в Replit

## Контекст

Это торговая система для Московской биржи с ML-прогнозированием. Модели уже обучены и работают, но нужна возможность **переобучать их прямо в Replit**.

---

## Текущее состояние

### Обученные модели (в папке `models/`):

| Модель | Win Rate | Profit Factor | Статус |
|--------|----------|---------------|--------|
| model_time_5m.joblib | 56.8% | 2.33 | ✅ Работает |
| model_time_10m.joblib | 56.0% | 2.31 | ✅ Работает |
| model_time_30m.joblib | 56.0% | 2.39 | ✅ Работает |
| model_time_1h.joblib | 55.4% | 2.39 | ✅ Работает |

### Что нужно для переобучения:

1. Данные с MOEX (свечи) — загружаются автоматически
2. 29 технических индикаторов — уже реализованы в `features.py`
3. Walk-Forward валидация — уже реализована в `advanced_train.py`

---

## Задача: Создать скрипт переобучения для Replit

### Требования:

1. **Простой запуск** одной командой в Shell:
   ```bash
   python scripts/retrain.py
   ```

2. **Этапы работы:**
   - Загрузить свежие данные с MOEX (последние 30 дней)
   - Построить фичи для каждого тикера
   - Обучить модели с Walk-Forward валидацией
   - Сохранить новые модели в `models/`
   - Отправить уведомление в Telegram

3. **Оптимизация для Replit:**
   - Использовать меньше памяти (обрабатывать по одному тикеру)
   - Показывать прогресс в консоли
   - Сохранять промежуточные результаты

---

## Существующий код для использования

### features.py — 29 индикаторов:

```python
from moex_agent.features import build_feature_frame, FEATURE_COLS

# FEATURE_COLS содержит список всех 29 фич:
# ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'log_ret_1',
#  'atr_14', 'atr_pct', 'bb_width', 'bb_pct',
#  'rsi_14', 'macd', 'macd_signal', 'macd_hist',
#  'stoch_k', 'stoch_d', 'willr_14', 'cci_20', 'mfi_14',
#  'adx_14', 'plus_di', 'minus_di', 'ema_ratio',
#  'obv_ratio', 'vwap_dist', 'volume_ratio',
#  'hour', 'minute', 'day_of_week']

# Использование:
candles = [{"ts": "...", "open": 100, "high": 101, ...}, ...]
df = build_feature_frame(candles)
X = df[FEATURE_COLS]
```

### labels.py — создание меток:

```python
from moex_agent.labels import create_labels

# horizon: "5m", "10m", "30m", "1h"
# Возвращает: 1 если цена выросла, 0 если упала
labels = create_labels(candles_df, horizon="5m")
```

### storage.py — работа с БД:

```python
from moex_agent.storage import connect

conn = connect("data/moex_agent.sqlite")
candles = conn.execute("SELECT * FROM candles WHERE secid = ?", (ticker,)).fetchall()
```

### bootstrap.py — загрузка данных:

```python
from moex_agent.bootstrap import bootstrap_recent
from moex_agent.config_schema import load_config

config = load_config()
conn = connect(config.sqlite_path)
bootstrap_recent(conn, config, days=30)  # Загрузить 30 дней
```

### telegram.py — уведомления:

```python
from moex_agent.telegram import send_telegram

send_telegram("✅ Модели переобучены!")
```

---

## Алгоритм обучения (Walk-Forward)

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

def train_model(X, y, n_splits=5):
    """
    Walk-Forward валидация:
    - Разбиваем данные на 5 временных окон
    - Обучаем на прошлых данных, тестируем на будущих
    - Нет утечки данных!
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_model = None
    best_score = 0

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        score = model.score(X.iloc[test_idx], y.iloc[test_idx])

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score
```

---

## Структура скрипта retrain.py

```python
#!/usr/bin/env python3
"""
Переобучение ML моделей в Replit.

Запуск:
    python scripts/retrain.py [--days 30] [--horizon 5m]
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Дней данных")
    parser.add_argument("--horizon", type=str, help="Конкретный горизонт (5m/10m/30m/1h)")
    args = parser.parse_args()

    # 1. Загрузка данных
    print("📊 Загрузка данных с MOEX...")
    # ... bootstrap_recent() ...

    # 2. Обучение моделей
    horizons = [args.horizon] if args.horizon else ["5m", "10m", "30m", "1h"]

    for horizon in horizons:
        print(f"\n🔄 Обучение модели {horizon}...")
        # ... train_model() ...
        # ... joblib.dump() ...

    # 3. Уведомление
    print("\n✅ Готово!")
    # ... send_telegram() ...

if __name__ == "__main__":
    main()
```

---

## Конфигурация моделей

```python
MODEL_PARAMS = {
    "n_estimators": 100,      # Количество деревьев
    "max_depth": 5,           # Глубина дерева
    "learning_rate": 0.1,     # Скорость обучения
    "min_samples_split": 50,  # Мин. samples для split
    "min_samples_leaf": 20,   # Мин. samples в листе
    "random_state": 42,       # Для воспроизводимости
}

HORIZONS = ["5m", "10m", "30m", "1h"]
N_SPLITS = 5  # Количество Walk-Forward splits
```

---

## Ожидаемый результат

После запуска `python scripts/retrain.py`:

```
📊 Загрузка данных с MOEX...
   Загружено: 150,000 свечей

🔄 Обучение модели 5m...
   Fold 1: accuracy=0.54, win_rate=0.56
   Fold 2: accuracy=0.55, win_rate=0.57
   Fold 3: accuracy=0.54, win_rate=0.55
   Fold 4: accuracy=0.55, win_rate=0.58
   Fold 5: accuracy=0.56, win_rate=0.57
   ✅ Модель сохранена: models/model_time_5m.joblib
   Метрики: WR=56.6%, Acc=54.8%

🔄 Обучение модели 10m...
   ...

🔄 Обучение модели 30m...
   ...

🔄 Обучение модели 1h...
   ...

✅ Все модели переобучены!
📱 Уведомление отправлено в Telegram
```

---

## Дополнительные требования

1. **Логирование** — записывать результаты в `data/retrain.log`

2. **Метаданные** — обновлять `models/meta.json`:
   ```json
   {
     "trained_at": "2026-01-22T15:30:00",
     "platform": "Replit",
     "horizons": {
       "5m": {"win_rate": 0.566, "accuracy": 0.548},
       ...
     }
   }
   ```

3. **Backup** — перед переобучением копировать старые модели в `models/backup/`

4. **Валидация** — проверять что новая модель лучше старой (иначе не заменять)

---

## Файлы для создания/изменения

1. **Создать:** `scripts/retrain.py` — основной скрипт переобучения
2. **Создать:** `scripts/validate_model.py` — валидация качества модели
3. **Изменить:** `moex_agent/advanced_train.py` — если нужны исправления

---

## Команды для Replit Shell

```bash
# Переобучить все модели
python scripts/retrain.py

# Переобучить только 5-минутную модель
python scripts/retrain.py --horizon 5m

# Переобучить с большим количеством данных
python scripts/retrain.py --days 60

# Проверить качество моделей
python scripts/validate_model.py
```

---

## Важно

- **Память:** Replit имеет ограниченную память. Обрабатывайте данные по частям.
- **Время:** Обучение занимает 10-30 минут. Показывайте прогресс.
- **Backup:** Всегда делайте backup перед заменой моделей.
- **Валидация:** Новая модель должна быть лучше старой!

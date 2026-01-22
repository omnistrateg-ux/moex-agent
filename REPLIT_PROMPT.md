# MOEX Agent — Промпт для Replit AI

## 🎯 Описание проекта

Это **торговая система для Московской биржи (MOEX)** с ML-прогнозированием и маржинальным риск-менеджментом.

**Цель:** Автоматическая генерация торговых сигналов и paper trading с контролем рисков.

---

## 🚀 Как запустить веб-продукт

### Вариант 1: Простой запуск (Dashboard + API)
```python
# main.py
import uvicorn
from moex_agent.webapp import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Вариант 2: Полный продукт (Trading + Web)
```python
# main.py
import threading
import uvicorn
from moex_agent.webapp import app

def run_trading():
    from moex_agent.margin_paper_trading import MarginPaperTrader
    trader = MarginPaperTrader(initial_capital=200000)
    trader.run(duration_hours=168)  # 1 week

# Start trading in background
trading_thread = threading.Thread(target=run_trading, daemon=True)
trading_thread.start()

# Start web server
uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## 📋 Что нужно сделать после загрузки

### 1. Установить зависимости
```bash
pip install -r requirements.txt
```

### 2. Настроить конфигурацию
```bash
mv config.yaml.replit config.yaml
```

### 3. Установить Secrets (в Replit UI)
- `TELEGRAM_BOT_TOKEN` — токен бота
- `TELEGRAM_CHAT_ID` — ID чата

### 4. Инициализировать базу данных
```bash
python -m moex_agent.bootstrap --days 7
```

### 5. Запустить
Нажать **Run** или:
```bash
python main.py
```

---

## 🏗️ Архитектура системы

```
Data Layer:
  moex_iss.py      → Получение данных с MOEX ISS API
  storage.py       → SQLite хранение свечей
  bootstrap.py     → Первоначальная загрузка данных

Feature Engineering:
  features.py      → 29 технических индикаторов
  labels.py        → Создание меток для обучения

ML Models:
  predictor.py     → Загрузка и предсказания
  advanced_train.py → Walk-Forward обучение

Signal Generation:
  anomaly.py       → Детекция аномальных движений
  engine.py        → Основной pipeline

Risk Management:
  margin_risk_engine.py → Kill-Switch, Dynamic Leverage
  bcs_broker.py    → Лимиты брокера

Trading:
  margin_paper_trading.py → Paper trading с риск-контролем

Interface:
  webapp.py        → FastAPI + Dashboard
  telegram.py      → Уведомления в Telegram
```

---

## 🔑 Ключевые файлы для понимания

### 1. `webapp.py` — Web Interface
- FastAPI приложение
- Endpoints: `/api/status`, `/api/signals`, `/api/alerts`
- HTML Dashboard на `/`

### 2. `margin_paper_trading.py` — Trading Logic
- Класс `MarginPaperTrader`
- Метод `run_cycle()` — один цикл торговли
- Метод `run()` — основной loop

### 3. `margin_risk_engine.py` — Risk Control
- `check_kill_switch()` — проверка лимитов
- `calculate_leverage()` — динамическое плечо
- `assess_trade()` — оценка сделки

### 4. `features.py` — Technical Indicators
- 29 индикаторов: RSI, MACD, ATR, OBV и др.
- Функция `build_feature_frame(candles)`

### 5. `predictor.py` — ML Models
- Загрузка `.joblib` моделей
- Метод `predict()` — вероятность роста

---

## 📊 API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | HTML Dashboard |
| `/api/health` | GET | Health check |
| `/api/status` | GET | Статус системы (candles, alerts, models) |
| `/api/signals` | GET | Запустить цикл и получить сигналы |
| `/api/alerts` | GET | Список последних сигналов |
| `/api/tickers` | GET | Список отслеживаемых тикеров |
| `/api/candles/{ticker}` | GET | Свечи по тикеру |

---

## 🛡️ Риск-параметры

```yaml
Max loss per trade: 0.5%
Max daily loss: 2%
Max weekly loss: 5%
Max drawdown: 10%
Kill after consecutive losses: 5
Disabled horizons: 1d, 1w
```

---

## 📁 Структура данных

### SQLite таблицы:
```sql
-- Свечи
CREATE TABLE candles (
    secid TEXT,
    ts TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    value REAL,
    volume INTEGER,
    interval INTEGER
);

-- Сигналы
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    created_ts TEXT,
    secid TEXT,
    horizon TEXT,
    p REAL,
    signal_type TEXT,
    entry REAL,
    take REAL,
    stop REAL,
    sent INTEGER
);
```

### State файл (`data/margin_paper_state.json`):
```json
{
    "initial_capital": 200000,
    "cash": 200000,
    "positions": {},
    "closed_trades": [],
    "consecutive_losses": 0,
    "kill_switch_active": false
}
```

---

## 🎨 Для улучшения Dashboard

Текущий dashboard в `webapp.py` показывает:
- Количество свечей, алертов, тикеров
- Загруженные модели
- Таблицу последних сигналов

**Можно добавить:**
1. Equity curve график
2. Открытые позиции
3. История сделок с PnL
4. Real-time обновления через WebSocket

---

## ⚠️ Важные ограничения Replit

1. **Storage:** ~1GB — не загружать большую БД, создавать на лету
2. **Memory:** 512MB-2GB — модели ~25MB, должно хватить
3. **Always On:** Платная фича для 24/7 работы
4. **No GPU:** Модели CPU-only, работает нормально

---

## 🔧 Troubleshooting

### Ошибка "No module named 'yaml'"
```bash
pip install pyyaml
```

### Ошибка "telegram.bot_token is required"
Установить Secrets в Replit UI или в config.yaml

### Ошибка "No candles in database"
```bash
python -m moex_agent.bootstrap --days 7
```

### Медленная загрузка
MOEX API может быть медленным. Использовать `--days 3` для быстрого старта.

---

## 📝 Пример main.py для Replit

```python
"""
MOEX Agent — Trading Signal Generator
Run on Replit with web dashboard
"""
import os
import sys
import threading
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize database if empty."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect

    config = load_config()
    conn = connect(config.sqlite_path)

    cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
    count = cur.fetchone()["cnt"]

    if count < 1000:
        logger.info("Database empty, bootstrapping...")
        from moex_agent.bootstrap import bootstrap_recent
        bootstrap_recent(conn, config, days=7)
        logger.info("Bootstrap complete!")
    else:
        logger.info(f"Database has {count:,} candles")

    conn.close()

def run_trading_background():
    """Run paper trading in background."""
    try:
        from moex_agent.margin_paper_trading import MarginPaperTrader
        trader = MarginPaperTrader(
            initial_capital=200000,
            max_leverage=3.0,
            resume=True
        )
        trader.run(duration_hours=168)
    except Exception as e:
        logger.error(f"Trading error: {e}")

def main():
    # Initialize
    logger.info("MOEX Agent starting...")
    init_database()

    # Start trading in background
    trading_thread = threading.Thread(
        target=run_trading_background,
        daemon=True
    )
    trading_thread.start()
    logger.info("Trading started in background")

    # Start web server
    import uvicorn
    from moex_agent.webapp import app

    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting web server on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
```

---

## 🎯 Цели для Replit AI

1. **Запустить веб-интерфейс** на порту 8080
2. **Показать dashboard** с текущим статусом
3. **Включить paper trading** в фоне
4. **Отправлять сигналы** в Telegram
5. **Обеспечить 24/7 работу** (с Always On)

Проект готов к использованию. Основная работа — настройка Secrets и запуск!

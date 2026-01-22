# MOEX Agent — AI-Powered Trading Signal Generator

## 🎯 Что это?

**MOEX Agent** — автоматизированная система генерации торговых сигналов для Московской биржи с маржинальным риск-менеджментом уровня проп-трейдинг деска.

### Ключевые возможности:
- 🤖 **ML-модели** для прогнозирования направления цены (4 горизонта: 5m, 10m, 30m, 1h)
- 📊 **Walk-Forward валидация** — честные метрики без data leakage
- ⚡ **Real-time мониторинг** 46 акций MOEX
- 🛡️ **Risk Engine** — Kill-Switch, Dynamic Leverage, Regime Detection
- 📱 **Telegram уведомления** о сигналах и сделках
- 🌐 **Web Dashboard** для мониторинга

---

## 📈 Метрики моделей (Walk-Forward)

| Горизонт | Win Rate | Profit Factor | Sharpe |
|----------|----------|---------------|--------|
| 5m       | 56.8%    | 2.33          | 3.44   |
| 10m      | 56.0%    | 2.31          | 3.74   |
| 30m      | 56.0%    | 2.39          | 4.33   |
| 1h       | 55.4%    | 2.39          | 4.90   |

**Данные:** 32.6M свечей, 4+ года истории

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                      MOEX Agent                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  MOEX    │───▶│ Feature  │───▶│    ML    │              │
│  │  ISS API │    │ Engine   │    │  Models  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │                     │
│       ▼               ▼               ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ SQLite   │    │  29+     │    │ Predict  │              │
│  │ Storage  │    │ Features │    │ Signals  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                       │                     │
│                                       ▼                     │
│                              ┌──────────────┐               │
│                              │  Risk Engine │               │
│                              │ ─────────────│               │
│                              │ • Kill-Switch│               │
│                              │ • Leverage   │               │
│                              │ • Regime     │               │
│                              └──────────────┘               │
│                                       │                     │
│                    ┌──────────────────┼──────────────────┐  │
│                    ▼                  ▼                  ▼  │
│             ┌──────────┐      ┌──────────┐      ┌────────┐ │
│             │ Telegram │      │  Paper   │      │  Web   │ │
│             │   Bot    │      │ Trading  │      │   UI   │ │
│             └──────────┘      └──────────┘      └────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Структура проекта

```
moex_agent/
├── moex_agent/                 # Основной пакет
│   ├── __init__.py
│   ├── __main__.py             # Entry point
│   │
│   │── # === DATA LAYER ===
│   ├── moex_iss.py             # API клиент MOEX ISS
│   ├── storage.py              # SQLite хранилище
│   ├── bootstrap.py            # Загрузка исторических данных
│   │
│   │── # === FEATURE ENGINEERING ===
│   ├── features.py             # 29 технических индикаторов
│   ├── labels.py               # Создание меток для обучения
│   │
│   │── # === ML MODELS ===
│   ├── predictor.py            # Загрузка и inference моделей
│   ├── train.py                # Базовое обучение
│   ├── advanced_train.py       # Walk-Forward обучение
│   │
│   │── # === SIGNAL GENERATION ===
│   ├── anomaly.py              # Детекция аномалий (сигналов)
│   ├── engine.py               # Основной pipeline
│   │
│   │── # === RISK MANAGEMENT ===
│   ├── margin_risk_engine.py   # Kill-Switch, Leverage, Regime
│   ├── bcs_broker.py           # Лимиты брокера БКС
│   ├── risk.py                 # Базовые риск-проверки
│   │
│   │── # === TRADING ===
│   ├── paper_trading.py        # Paper trading (базовый)
│   ├── margin_paper_trading.py # Margin paper trading
│   ├── live.py                 # Live режим
│   │
│   │── # === INTERFACE ===
│   ├── telegram.py             # Telegram уведомления
│   ├── webapp.py               # FastAPI web server
│   ├── dashboard.py            # Web dashboard
│   │
│   │── # === CONFIG ===
│   ├── config.py               # Legacy config
│   ├── config_schema.py        # Pydantic config schema
│   └── logging_config.py       # Логирование
│
├── models/                     # Обученные модели
│   ├── model_time_5m.joblib
│   ├── model_time_10m.joblib
│   ├── model_time_30m.joblib
│   └── model_time_1h.joblib
│
├── data/                       # Данные (создаётся автоматически)
│   ├── moex_agent.sqlite       # База свечей
│   └── margin_paper_state.json # Состояние paper trading
│
├── config.yaml                 # Конфигурация
├── requirements.txt            # Зависимости
└── pyproject.toml              # Python package config
```

---

## 🔄 Как работает система

### 1. Сбор данных
```
MOEX ISS API → 1-минутные свечи → SQLite
```
- Загружаем OHLCV данные для 46 тикеров
- Храним в SQLite для быстрого доступа

### 2. Feature Engineering (29 индикаторов)
```python
# Trend
- SMA 20, 50
- EMA 12, 26
- MACD, Signal, Histogram

# Momentum
- RSI 14
- Stochastic %K, %D
- ROC 10, 20
- Williams %R

# Volatility
- ATR 14
- Bollinger Bands (width, %B)
- Keltner Channels

# Volume
- OBV, OBV change
- VWAP
- Volume SMA ratio

# Price Action
- Candle body ratio
- Upper/Lower shadows
- Gap detection
```

### 3. ML Prediction
```
Features → HistGradientBoosting → Probability[0-1]
```
- 4 модели для разных горизонтов
- Калибровка вероятностей (Isotonic)
- Threshold: 54% для входа

### 4. Risk Assessment
```python
def assess_trade(signal):
    # 1. Regime Detection (BULL/BEAR/SIDEWAYS/HIGH_VOL)
    regime = detect_regime(candles)

    # 2. Kill-Switch Check
    if daily_loss >= 2% or weekly_loss >= 5%:
        return DISABLE

    # 3. Dynamic Leverage
    leverage = f(confidence, regime, volatility, drawdown)

    # 4. Position Sizing
    size = max_risk / (atr * leverage)

    return ALLOW if leverage > 0 else DISABLE
```

### 5. Execution (Paper Trading)
```
Signal + Risk OK → Open Position → Monitor → Close (Take/Stop/Timeout)
```

---

## 🛡️ Risk Management Rules

| Параметр | Лимит |
|----------|-------|
| Max loss per trade | 0.5% equity |
| Max daily loss | 2% |
| Max weekly loss | 5% |
| Max drawdown | 10% |
| Kill after losses | 5 подряд |
| Disabled horizons | 1d, 1w (gap risk) |

### Dynamic Leverage Formula
```
base_lev = HORIZON_MAX[horizon]  # 5m=3x, 10m=3x, 30m=2.5x, 1h=2x

multipliers:
  confidence: (prob - 0.54) / 0.1  # 0.54-0.64 → 0-1
  regime: BULL=1.0, SIDEWAYS=0.7, HIGH_VOL=0.3, BEAR=0.0
  volatility: 1.0 - vol_percentile * 0.5
  drawdown: 1.0 - (dd_pct / 10) * 0.5
  loss_streak: 1.0 - streak * 0.15

final_leverage = base_lev * product(multipliers)
```

---

## 🚀 Запуск

### Paper Trading (основной режим)
```bash
python -m moex_agent.margin_paper_trading
```

### Web Dashboard
```bash
python -m moex_agent.webapp
# → http://localhost:8080
```

### Bootstrap данных
```bash
python -m moex_agent.bootstrap --days 14
```

### Обучение моделей
```bash
python -m moex_agent.advanced_train --horizons 5m 10m 30m 1h
```

---

## 🌐 Web API Endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/` | GET | Dashboard HTML |
| `/api/status` | GET | Статус системы |
| `/api/signals` | GET | Текущие сигналы |
| `/api/positions` | GET | Открытые позиции |
| `/api/trades` | GET | История сделок |
| `/api/equity` | GET | Equity curve |
| `/health` | GET | Health check |

---

## ⚙️ Конфигурация (config.yaml)

```yaml
app:
  poll_seconds: 5          # Интервал опроса
  cooldown_minutes: 30     # Кулдаун между сигналами

storage:
  sqlite_path: "data/moex_agent.sqlite"

universe:
  tickers:
    - SBER
    - GAZP
    - LKOH
    # ... 46 тикеров

signals:
  horizons:
    - { name: "5m",  minutes: 5 }
    - { name: "10m", minutes: 10 }
    - { name: "30m", minutes: 30 }
    - { name: "1h",  minutes: 60 }
  p_threshold: 0.52
  price_exit:
    take_atr: 0.70
    stop_atr: 0.40

telegram:
  enabled: true
  # Set via environment: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

---

## 🔑 Environment Variables

| Variable | Описание |
|----------|----------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота |
| `TELEGRAM_CHAT_ID` | ID чата для уведомлений |

---

## 📊 Пример вывода

### Telegram сигнал:
```
🎰 MARGIN PAPER TRADING

📈 НОВАЯ ПОЗИЦИЯ
━━━━━━━━━━━━━━━━━━━━
📍 SBER | LONG
📊 Плечо: 2.1x | Режим: BULL
💵 Цена: 267.50 ₽
📦 Размер: 150 шт. (85,000 ₽)
🎯 Take: 269.80 | Stop: 266.10
━━━━━━━━━━━━━━━━━━━━
💵 EQUITY: 200,000 ₽
```

### Закрытие сделки:
```
🎰 MARGIN PAPER TRADING

✅ СДЕЛКА ЗАКРЫТА
━━━━━━━━━━━━━━━━━━━━
📍 SBER (LONG)
📊 Плечо: 2.1x
💰 ПРИБЫЛЬ: +1,450 ₽ (+1.71%)
📊 Причина: take
━━━━━━━━━━━━━━━━━━━━
💵 EQUITY: 201,450 ₽
📉 Drawdown: 0.0%
🔢 Loss Streak: 0
```

---

## 🎓 Для Replit AI

**Контекст:** Это production-ready торговая система для MOEX. Модели обучены на 32M свечей с Walk-Forward валидацией.

**Главные файлы:**
- `margin_paper_trading.py` — основной trading loop
- `margin_risk_engine.py` — риск-менеджмент
- `webapp.py` — web интерфейс
- `features.py` — feature engineering
- `predictor.py` — ML inference

**Для веб-продукта:**
1. Запустить `webapp.py` на порту 8080
2. Dashboard показывает: equity, позиции, сигналы, историю
3. API возвращает JSON для фронтенда

**Ограничения Replit:**
- Нет GPU (но модели CPU-only)
- Storage ~1GB (база создаётся на лету)
- Для 24/7 нужен Always On (платный)

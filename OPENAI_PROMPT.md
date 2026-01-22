# MOEX Agent — Полный контекст проекта

## Кто я и что делал

Я Claude (Opus 4.5), работал над этим проектом с пользователем. Ниже полное описание системы, чтобы ты мог продолжить работу.

---

## 🎯 Что это за проект

**MOEX Agent** — автоматизированная торговая система для Московской биржи (MOEX) с:
- ML-моделями для прогнозирования направления цены
- Маржинальным риск-менеджментом уровня проп-трейдинг деска
- Paper Trading для тестирования без реальных денег
- Web Dashboard для мониторинга
- Telegram уведомлениями о сделках

---

## 📊 Метрики обученных моделей

Модели обучены на **32.6 миллионов свечей** (4+ года данных) с Walk-Forward валидацией:

| Горизонт | Win Rate | Profit Factor | Sharpe |
|----------|----------|---------------|--------|
| 5m       | 56.8%    | 2.33          | 3.44   |
| 10m      | 56.0%    | 2.31          | 3.74   |
| 30m      | 56.0%    | 2.39          | 4.33   |
| 1h       | 55.4%    | 2.39          | 4.90   |

**Горизонты 1d и 1w ОТКЛЮЧЕНЫ** для маржинальной торговли из-за gap risk (риск гэпа на открытии).

---

## 🏗️ Архитектура системы

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
├── moex_agent/                 # Основной Python пакет
│   │
│   │── # === СЛОЙ ДАННЫХ ===
│   ├── moex_iss.py             # API клиент MOEX ISS (получение свечей)
│   ├── storage.py              # SQLite хранилище свечей
│   ├── bootstrap.py            # Загрузка исторических данных
│   │
│   │── # === FEATURE ENGINEERING ===
│   ├── features.py             # 29 технических индикаторов
│   ├── labels.py               # Создание меток для обучения
│   │
│   │── # === ML МОДЕЛИ ===
│   ├── predictor.py            # Загрузка моделей и предсказания
│   ├── train.py                # Базовое обучение
│   ├── advanced_train.py       # Walk-Forward обучение (основной)
│   │
│   │── # === ГЕНЕРАЦИЯ СИГНАЛОВ ===
│   ├── anomaly.py              # Детекция аномалий в цене/объёме
│   ├── engine.py               # Основной pipeline обработки
│   │
│   │── # === РИСК-МЕНЕДЖМЕНТ ===
│   ├── margin_risk_engine.py   # Kill-Switch, Dynamic Leverage, Regime
│   ├── bcs_broker.py           # Маржинальные лимиты брокера БКС
│   ├── risk.py                 # Базовые риск-проверки
│   │
│   │── # === ТОРГОВЛЯ ===
│   ├── paper_trading.py        # Базовый paper trading
│   ├── margin_paper_trading.py # Маржинальный paper trading (основной)
│   ├── live.py                 # Live режим (для будущего)
│   │
│   │── # === ИНТЕРФЕЙС ===
│   ├── webapp.py               # FastAPI веб-сервер + Dashboard
│   ├── telegram.py             # Отправка уведомлений в Telegram
│   │
│   │── # === КОНФИГУРАЦИЯ ===
│   ├── config.py               # Legacy конфиг
│   ├── config_schema.py        # Pydantic схема конфига
│   └── logging_config.py       # Настройка логирования
│
├── models/                     # Обученные модели
│   ├── model_time_5m.joblib    # Модель на 5 минут
│   ├── model_time_10m.joblib   # Модель на 10 минут
│   ├── model_time_30m.joblib   # Модель на 30 минут
│   └── model_time_1h.joblib    # Модель на 1 час
│
├── data/                       # Данные (создаётся автоматически)
│   ├── moex_agent.sqlite       # SQLite база со свечами
│   └── margin_paper_state.json # Состояние paper trading
│
├── config.yaml                 # Основной конфиг
├── main.py                     # Entry point для Replit
└── requirements.txt            # Python зависимости
```

---

## 🔧 Ключевые файлы и их содержимое

### 1. margin_risk_engine.py — Риск-менеджмент

```python
# Конфигурация Kill-Switch
@dataclass
class KillSwitchConfig:
    max_loss_per_trade_pct: float = 0.5    # Макс. убыток на сделку
    max_daily_loss_pct: float = 2.0        # Макс. дневной убыток
    max_weekly_loss_pct: float = 5.0       # Макс. недельный убыток
    max_consecutive_losses: int = 5         # Макс. убытков подряд
    max_drawdown_pct: float = 10.0         # Макс. просадка

# Режимы рынка
class MarketRegime(Enum):
    BULL = "BULL"           # Восходящий тренд
    BEAR = "BEAR"           # Нисходящий тренд
    SIDEWAYS = "SIDEWAYS"   # Боковик
    HIGH_VOL = "HIGH_VOL"   # Высокая волатильность
    UNKNOWN = "UNKNOWN"

# Основной класс
class MarginRiskEngine:
    DISABLED_HORIZONS = {"1d", "1w"}  # Отключены для маржи
    HORIZON_MAX_LEVERAGE = {
        "5m": 3.0,
        "10m": 3.0,
        "30m": 2.5,
        "1h": 2.0
    }

    def calculate_leverage(self, confidence, regime, volatility, horizon):
        """Динамический расчёт плеча."""
        if horizon in self.DISABLED_HORIZONS:
            return 0.0
        if confidence < 0.54:
            return 0.0

        base = self.HORIZON_MAX_LEVERAGE.get(horizon, 1.0)

        # Множители
        conf_mult = min((confidence - 0.54) / 0.1, 1.0)
        regime_mult = {"BULL": 1.0, "SIDEWAYS": 0.7, "HIGH_VOL": 0.3, "BEAR": 0.0}
        vol_mult = 1.0 - volatility * 0.5
        dd_mult = 1.0 - (drawdown / 10) * 0.5
        streak_mult = 1.0 - loss_streak * 0.15

        return base * conf_mult * regime_mult * vol_mult * dd_mult * streak_mult

    def check_kill_switch(self):
        """Проверка условий остановки торговли."""
        if self.state.daily_loss_pct >= self.kill_config.max_daily_loss_pct:
            return True, "Daily loss limit"
        if self.state.weekly_loss_pct >= self.kill_config.max_weekly_loss_pct:
            return True, "Weekly loss limit"
        if self.state.consecutive_losses >= self.kill_config.max_consecutive_losses:
            return True, "Consecutive losses"
        return False, None
```

### 2. margin_paper_trading.py — Paper Trading

```python
class MarginPaperTrader:
    def __init__(self, initial_capital=200000, max_leverage=3.0):
        self.account = MarginAccount(initial_capital)
        self.risk_engine = MarginRiskEngine(initial_capital)

    def run_cycle(self):
        """Один цикл торговли."""
        # 1. Проверить Kill-Switch
        kill_active, reason = self.risk_engine.check_kill_switch()
        if kill_active:
            return  # Торговля остановлена

        # 2. Получить данные с MOEX
        result = self.engine.run_cycle(self.conn)

        # 3. Проверить выходы из позиций
        for position in self.account.positions:
            if hit_take_profit or hit_stop_loss or timeout:
                self.close_position(position)

        # 4. Обработать новые сигналы
        for signal in result.signals:
            if signal.horizon in {"1d", "1w"}:
                continue  # Пропускаем длинные горизонты

            # Оценка риска
            assessment = self.risk_engine.assess_trade(
                ticker=signal.ticker,
                direction=signal.direction,
                horizon=signal.horizon,
                confidence=signal.probability
            )

            if assessment.decision == "ALLOW":
                self.open_position(signal, assessment.leverage)

    def run(self, duration_hours=168):
        """Основной цикл."""
        while not shutdown:
            self.run_cycle()
            time.sleep(5)  # Пауза между циклами
```

### 3. features.py — 29 технических индикаторов

```python
def build_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Строит 29 фич для каждой свечи."""

    # Trend индикаторы
    - SMA 20, 50 (скользящие средние)
    - EMA 12, 26 (экспоненциальные)
    - MACD, Signal, Histogram

    # Momentum
    - RSI 14
    - Stochastic %K, %D
    - ROC 10, 20 (Rate of Change)
    - Williams %R

    # Volatility
    - ATR 14 (Average True Range)
    - Bollinger Bands (width, %B)
    - Keltner Channels

    # Volume
    - OBV (On-Balance Volume)
    - OBV change (нормализованный)
    - VWAP
    - Volume SMA ratio

    # Price Action
    - Candle body ratio
    - Upper/Lower shadows
    - Gap detection

    return features_df
```

### 4. webapp.py — Web Dashboard

```python
# FastAPI приложение
app = FastAPI(title="MOEX Agent")

# API Endpoints
@app.get("/")                    # HTML Dashboard
@app.get("/api/health")          # Health check
@app.get("/api/status")          # Статус системы
@app.get("/api/equity")          # Equity и P&L
@app.get("/api/positions")       # Открытые позиции
@app.get("/api/trades")          # История сделок с "reason"
@app.get("/api/alerts")          # Последние сигналы
@app.get("/api/signals")         # Запустить цикл

# Каждая сделка содержит поле "reason":
def _get_trade_reason(trade):
    return (
        f"ML-модель предсказала {signal} на горизонте {horizon} | "
        f"Режим рынка: {regime} | "
        f"Динамическое плечо: {leverage}x"
    )
```

### 5. config.yaml — Конфигурация

```yaml
app:
  poll_seconds: 5              # Интервал опроса
  cooldown_minutes: 30         # Кулдаун между сигналами

storage:
  sqlite_path: "data/moex_agent.sqlite"

universe:
  board: "TQBR"
  tickers:
    - SBER      # Сбербанк
    - GAZP      # Газпром
    - LKOH      # Лукойл
    # ... 46 тикеров всего

signals:
  horizons:
    - { name: "5m",  minutes: 5 }
    - { name: "10m", minutes: 10 }
    - { name: "30m", minutes: 30 }
    - { name: "1h",  minutes: 60 }
  p_threshold: 0.52            # Порог вероятности
  price_exit:
    take_atr: 0.70             # Take Profit в ATR
    stop_atr: 0.40             # Stop Loss в ATR

telegram:
  enabled: true
  bot_token: ""                # Из env: TELEGRAM_BOT_TOKEN
  chat_id: ""                  # Из env: TELEGRAM_CHAT_ID
```

---

## 🛡️ Риск-менеджмент (детально)

### Kill-Switch условия:
| Условие | Лимит | Действие |
|---------|-------|----------|
| Убыток на сделку | > 0.5% капитала | Не открываем |
| Дневной убыток | > 2% | СТОП торговли |
| Недельный убыток | > 5% | СТОП торговли |
| Убытков подряд | > 5 | СТОП торговли |
| Просадка | > 10% | СТОП торговли |

### Динамическое плечо:
```
final_leverage = base_leverage × conf × regime × vol × dd × streak

где:
- base_leverage: 5m=3x, 10m=3x, 30m=2.5x, 1h=2x
- conf: (probability - 0.54) / 0.1  (0 до 1)
- regime: BULL=1.0, SIDEWAYS=0.7, HIGH_VOL=0.3, BEAR=0
- vol: 1.0 - volatility_percentile × 0.5
- dd: 1.0 - (drawdown% / 10) × 0.5
- streak: 1.0 - consecutive_losses × 0.15
```

### Режимы рынка:
- **BULL**: SMA20 > SMA50, цена > SMA20, RSI > 50
- **BEAR**: SMA20 < SMA50, цена < SMA20, RSI < 50
- **HIGH_VOL**: ATR > 90-й перцентиль
- **SIDEWAYS**: всё остальное

---

## 📱 Telegram уведомления

### Новая позиция:
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

## 🌐 Web Dashboard

Dashboard показывает:
1. **Статус системы**: свечей, сигналов, тикеров, моделей
2. **Equity & P&L**: капитал, изменение, daily/total P&L
3. **Win Rate**: процент выигрышей, profit factor
4. **Открытые позиции**: тикер, направление, плечо, take/stop
5. **История сделок**: время, P&L, результат, ОСНОВАНИЕ
6. **Kill-Switch статус**: активен/неактивен

При клике на сделку открывается модальное окно с полными деталями и объяснением почему была совершена сделка.

---

## 🚀 Запуск

### Локально:
```bash
# 1. Создать venv
python -m venv .venv && source .venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Загрузить данные
python -m moex_agent.bootstrap --days 14

# 4. Запустить paper trading
caffeinate -i python -m moex_agent.margin_paper_trading

# 5. Или запустить веб-сервер
python main.py
```

### На Replit:
```bash
# 1. Переименовать конфиг
mv config.yaml.replit config.yaml

# 2. Добавить Secrets
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID = "..."

# 3. Загрузить данные
python -m moex_agent.bootstrap --days 7

# 4. Нажать Run (запустит main.py)
```

---

## 📈 Текущее состояние

На момент передачи:
- Paper Trading запущен и работает
- Совершено несколько сделок
- Equity ~200,000 ₽
- Kill-Switch не активирован
- Модели обучены на свежих данных (32.6M свечей)

---

## 🎯 Что нужно сделать дальше

Пользователь хочет:
1. **Перенести проект на Replit** — архив готов (moex_agent_replit.zip)
2. **Веб-продукт** — Dashboard работает, показывает сделки
3. **Основание для сделок** — реализовано в поле "reason"

Возможные улучшения:
- Добавить график Equity Curve
- WebSocket для real-time обновлений
- Больше деталей в "основании" (конкретные индикаторы)
- Backtesting интерфейс

---

## 📦 Архив для Replit

Файл: `moex_agent_replit.zip` (9.3 MB)
Содержит всё необходимое для запуска на Replit.

---

## 🔑 Важные переменные окружения

| Переменная | Описание |
|------------|----------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота |
| `TELEGRAM_CHAT_ID` | ID чата для уведомлений |

---

Это полный контекст проекта. Можешь продолжать работу с этой информацией!

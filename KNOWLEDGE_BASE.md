# MOEX Agent — Полная База Знаний

## Содержание
1. [Обзор проекта](#1-обзор-проекта)
2. [Архитектура системы](#2-архитектура-системы)
3. [ML-модели](#3-ml-модели)
4. [Риск-менеджмент](#4-риск-менеджмент)
5. [API и Web-интерфейс](#5-api-и-web-интерфейс)
6. [Конфигурация](#6-конфигурация)
7. [Полный код ядра](#7-полный-код-ядра)
8. [База данных](#8-база-данных)
9. [Запуск и деплой](#9-запуск-и-деплой)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Обзор проекта

**MOEX Agent** — автоматизированная торговая система для Московской биржи с ML-прогнозированием и маржинальным риск-менеджментом.

### Ключевые возможности:
- **ML-прогнозирование** — Walk-Forward валидация, 56% Win Rate, Profit Factor > 2.3
- **Маржинальная торговля** — поддержка плеча до 5x с динамическим контролем
- **Риск-менеджмент** — Kill-Switch, режимы рынка, лимиты брокера BCS
- **Paper Trading** — симуляция торговли без реальных денег
- **Web Dashboard** — FastAPI + HTML интерфейс
- **Telegram уведомления** — сигналы в реальном времени

### Метрики моделей (Walk-Forward):
| Горизонт | Win Rate | Profit Factor | Trades |
|----------|----------|---------------|--------|
| 5m       | 56.8%    | 2.33          | 1,247  |
| 10m      | 56.0%    | 2.31          | 1,183  |
| 30m      | 56.0%    | 2.39          | 1,156  |
| 1h       | 55.4%    | 2.39          | 1,089  |

---

## 2. Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                        MOEX Agent System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  MOEX ISS    │───▶│   Storage    │───▶│   Features   │      │
│  │    API       │    │   (SQLite)   │    │  (29 индик.) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Bootstrap   │    │   Candles    │    │  ML Models   │      │
│  │  (7+ days)   │    │   Table      │    │  (GradBoost) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                │               │
│                            ┌───────────────────┘               │
│                            ▼                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Anomaly    │───▶│   Engine     │───▶│   Signals    │      │
│  │  Detection   │    │  (Pipeline)  │    │  Generation  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                            │                   │               │
│                            ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Margin Risk  │◀──▶│   Paper      │───▶│  Telegram    │      │
│  │   Engine     │    │  Trading     │    │  Notify      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                      ┌──────────────┐                          │
│                      │   Web App    │                          │
│                      │  (FastAPI)   │                          │
│                      └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Структура файлов:

```
moex_agent/
├── moex_agent/              # Основной пакет
│   ├── __init__.py          # Инициализация пакета
│   ├── __main__.py          # CLI entry point
│   ├── config_schema.py     # Pydantic конфигурация
│   ├── moex_iss.py          # MOEX ISS API клиент
│   ├── storage.py           # SQLite операции
│   ├── bootstrap.py         # Загрузка исторических данных
│   ├── features.py          # 29 технических индикаторов
│   ├── labels.py            # Создание меток для ML
│   ├── predictor.py         # Загрузка и inference моделей
│   ├── advanced_train.py    # Walk-Forward обучение
│   ├── anomaly.py           # Детекция аномалий
│   ├── engine.py            # Основной pipeline
│   ├── margin_risk_engine.py # Риск-менеджмент
│   ├── margin_paper_trading.py # Paper trading
│   ├── bcs_broker.py        # Лимиты брокера BCS
│   ├── telegram.py          # Telegram уведомления
│   └── webapp.py            # FastAPI веб-приложение
├── models/                  # Обученные модели
│   ├── model_time_5m.joblib
│   ├── model_time_10m.joblib
│   ├── model_time_30m.joblib
│   ├── model_time_1h.joblib
│   └── meta.json            # Метаданные моделей
├── data/                    # Данные и состояние
│   ├── moex_agent.sqlite    # База свечей
│   └── margin_paper_state.json # Состояние торговли
├── config.yaml              # Конфигурация
├── main.py                  # Entry point
└── requirements.txt         # Зависимости
```

---

## 3. ML-модели

### 3.1 Архитектура модели

- **Алгоритм:** GradientBoostingClassifier
- **Валидация:** Walk-Forward (честная, без утечки данных)
- **Features:** 29 технических индикаторов

### 3.2 Технические индикаторы (features.py)

```python
FEATURES = [
    # Ценовые
    'ret_1', 'ret_5', 'ret_10', 'ret_20',  # Returns
    'log_ret_1',                            # Log return

    # Volatility
    'atr_14', 'atr_pct',                   # ATR
    'bb_width', 'bb_pct',                   # Bollinger Bands

    # Momentum
    'rsi_14',                               # RSI
    'macd', 'macd_signal', 'macd_hist',    # MACD
    'stoch_k', 'stoch_d',                   # Stochastic
    'willr_14',                             # Williams %R
    'cci_20',                               # CCI
    'mfi_14',                               # MFI

    # Trend
    'adx_14', 'plus_di', 'minus_di',       # ADX
    'ema_ratio',                            # EMA ratio

    # Volume
    'obv_ratio',                            # OBV
    'vwap_dist',                            # VWAP distance
    'volume_ratio',                         # Volume ratio

    # Time
    'hour', 'minute', 'day_of_week',       # Time features
]
```

### 3.3 Walk-Forward валидация

```python
def walk_forward_train(candles, horizon, n_splits=5):
    """
    Walk-Forward валидация:
    1. Разбиваем данные на n_splits временных окон
    2. Для каждого окна:
       - Train на предыдущих данных
       - Test на текущем окне
    3. Нет утечки данных из будущего!
    """
    results = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        results.append(evaluate(y[test_idx], pred))
    return aggregate(results)
```

### 3.4 Метаданные моделей (meta.json)

```json
{
  "trained_at": "2026-01-21T11:16:00",
  "horizons": {
    "5m": {"win_rate": 0.568, "profit_factor": 2.33, "trades": 1247},
    "10m": {"win_rate": 0.560, "profit_factor": 2.31, "trades": 1183},
    "30m": {"win_rate": 0.560, "profit_factor": 2.39, "trades": 1156},
    "1h": {"win_rate": 0.554, "profit_factor": 2.39, "trades": 1089}
  },
  "features_count": 29,
  "candles_used": 32600000
}
```

---

## 4. Риск-менеджмент

### 4.1 Margin Risk Engine

```python
class MarginRiskEngine:
    """Маржинальный риск-менеджмент."""

    # Лимиты потерь
    MAX_LOSS_PER_TRADE = 0.005   # 0.5% на сделку
    MAX_DAILY_LOSS = 0.02        # 2% в день
    MAX_WEEKLY_LOSS = 0.05       # 5% в неделю
    MAX_DRAWDOWN = 0.10          # 10% максимум

    # Kill-Switch
    CONSECUTIVE_LOSSES_LIMIT = 5  # 5 убытков подряд

    def check_kill_switch(self, state: TradingState) -> bool:
        """Проверка всех лимитов."""
        if state.consecutive_losses >= self.CONSECUTIVE_LOSSES_LIMIT:
            return True  # STOP TRADING
        if state.daily_loss >= self.MAX_DAILY_LOSS:
            return True
        if state.drawdown >= self.MAX_DRAWDOWN:
            return True
        return False

    def calculate_leverage(self, regime: MarketRegime, volatility: float) -> float:
        """Динамическое плечо на основе режима рынка."""
        base_leverage = {
            MarketRegime.BULL: 2.0,
            MarketRegime.BEAR: 1.5,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOL: 0.5,
        }
        # Уменьшаем плечо при высокой волатильности
        vol_factor = max(0.5, 1.0 - volatility * 2)
        return base_leverage[regime] * vol_factor
```

### 4.2 Режимы рынка (Market Regimes)

```python
class MarketRegime(Enum):
    BULL = "bull"           # Бычий тренд
    BEAR = "bear"           # Медвежий тренд
    SIDEWAYS = "sideways"   # Боковик
    HIGH_VOL = "high_vol"   # Высокая волатильность

def detect_regime(candles: List[dict]) -> MarketRegime:
    """Определение текущего режима рынка."""
    returns = calculate_returns(candles, period=20)
    volatility = np.std(returns) * np.sqrt(252)
    trend = np.mean(returns)

    if volatility > 0.4:
        return MarketRegime.HIGH_VOL
    elif trend > 0.001:
        return MarketRegime.BULL
    elif trend < -0.001:
        return MarketRegime.BEAR
    else:
        return MarketRegime.SIDEWAYS
```

### 4.3 BCS Broker Limits

```python
class BCSBrokerAdapter:
    """Лимиты брокера BCS."""

    # Маржинальные коэффициенты по категориям
    MARGIN_RATES = {
        "KSUR": {"long": 0.20, "short": 0.25},  # Квалифицированные
        "KPUR": {"long": 0.25, "short": 0.30},  # Стандартные
        "NLIQ": {"long": 1.00, "short": None},  # Неликвидные
    }

    def get_max_position_size(self, ticker: str, equity: float) -> float:
        """Максимальный размер позиции с учетом маржи."""
        category = self.get_ticker_category(ticker)
        margin_rate = self.MARGIN_RATES[category]["long"]
        return equity / margin_rate
```

---

## 5. API и Web-интерфейс

### 5.1 REST API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | HTML Dashboard |
| `/api/health` | GET | Health check |
| `/api/status` | GET | Статус системы |
| `/api/signals` | GET | Генерация сигналов |
| `/api/alerts` | GET | Последние алерты |
| `/api/trades` | GET | История сделок |
| `/api/equity` | GET | Equity и P&L |
| `/api/positions` | GET | Открытые позиции |
| `/api/tickers` | GET | Список тикеров |
| `/api/candles/{ticker}` | GET | Свечи по тикеру |

### 5.2 Пример ответа /api/equity

```json
{
  "equity": 200002,
  "initial_capital": 200000,
  "total_pnl": 2,
  "total_pnl_pct": 0.001,
  "daily_pnl": 0,
  "weekly_pnl": 2,
  "max_drawdown": 0.024,
  "win_rate": 0.60,
  "profit_factor": 2.5,
  "total_trades": 5,
  "winning_trades": 3,
  "losing_trades": 2,
  "kill_switch_active": false,
  "consecutive_losses": 0
}
```

### 5.3 Пример ответа /api/trades

```json
[
  {
    "id": 1,
    "ticker": "SBER",
    "direction": "LONG",
    "entry_price": 250.50,
    "exit_price": 252.30,
    "quantity": 100,
    "pnl": 180.0,
    "pnl_pct": 0.72,
    "entry_time": "2026-01-21T10:30:00",
    "exit_time": "2026-01-21T11:15:00",
    "horizon": "5m",
    "reason": "ML-модель: LONG на 5m | Режим: BULL | Плечо: 2.0x"
  }
]
```

### 5.4 Dashboard HTML

Dashboard автоматически обновляется каждые 30 секунд и показывает:
- Equity curve (график)
- Текущие позиции
- История сделок с P&L
- Статус Kill-Switch
- Метрики Win Rate и Profit Factor

---

## 6. Конфигурация

### 6.1 config.yaml (полный)

```yaml
# Торговые тикеры (46 штук для полной версии)
tickers:
  - SBER
  - GAZP
  - LKOH
  - GMKN
  - NVTK
  - ROSN
  - YNDX
  - MTSS
  - MGNT
  - VTBR
  - ALRS
  - CHMF
  - NLMK
  - PLZL
  - POLY
  - TATN
  - SNGS
  - MOEX
  - RUAL
  - AFLT

# Пути к данным
sqlite_path: "data/moex_agent.sqlite"
models_dir: "models"

# Telegram настройки (из env)
telegram:
  bot_token: ""  # TELEGRAM_BOT_TOKEN
  chat_id: ""    # TELEGRAM_CHAT_ID
  enabled: true

# ML настройки
ml:
  enabled: true
  min_probability: 0.55  # Минимальная вероятность для сигнала

# Qwen LLM (опционально)
qwen:
  enabled: false
  model: "qwen3:8b"

# Горизонты прогнозирования
horizons:
  - "5m"
  - "10m"
  - "30m"
  - "1h"

# Отключенные горизонты (слишком длинные)
disabled_horizons:
  - "1d"
  - "1w"

# Риск-менеджмент
risk:
  max_loss_per_trade: 0.005
  max_daily_loss: 0.02
  max_weekly_loss: 0.05
  max_drawdown: 0.10
  consecutive_losses_limit: 5

# Paper trading
paper_trading:
  initial_capital: 200000
  max_leverage: 3.0
  max_positions: 3
```

### 6.2 Environment Variables (Replit Secrets)

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
PORT=8080  # опционально
```

---

## 7. Полный код ядра

### 7.1 margin_risk_engine.py (ключевые части)

```python
"""Margin Risk Engine — контроль рисков маржинальной торговли."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"

@dataclass
class TradingState:
    equity: float
    initial_capital: float
    daily_pnl: float
    weekly_pnl: float
    consecutive_losses: int
    open_positions: int

    @property
    def drawdown(self) -> float:
        return (self.initial_capital - self.equity) / self.initial_capital

class MarginRiskEngine:
    MAX_LOSS_PER_TRADE = 0.005
    MAX_DAILY_LOSS = 0.02
    MAX_WEEKLY_LOSS = 0.05
    MAX_DRAWDOWN = 0.10
    CONSECUTIVE_LOSSES_LIMIT = 5

    def check_kill_switch(self, state: TradingState) -> tuple[bool, str]:
        if state.consecutive_losses >= self.CONSECUTIVE_LOSSES_LIMIT:
            return True, f"Consecutive losses: {state.consecutive_losses}"
        if abs(state.daily_pnl / state.initial_capital) >= self.MAX_DAILY_LOSS:
            return True, "Daily loss limit reached"
        if abs(state.weekly_pnl / state.initial_capital) >= self.MAX_WEEKLY_LOSS:
            return True, "Weekly loss limit reached"
        if state.drawdown >= self.MAX_DRAWDOWN:
            return True, f"Max drawdown reached: {state.drawdown:.1%}"
        return False, ""

    def calculate_leverage(self, regime: MarketRegime, atr_pct: float) -> float:
        base = {
            MarketRegime.BULL: 2.0,
            MarketRegime.BEAR: 1.5,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOL: 0.5,
        }[regime]
        vol_factor = max(0.5, 1.0 - atr_pct * 10)
        return round(base * vol_factor, 2)

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        leverage: float
    ) -> int:
        risk_amount = equity * self.MAX_LOSS_PER_TRADE * leverage
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share < 0.01:
            return 0
        size = int(risk_amount / risk_per_share)
        return max(1, size)

    def detect_regime(self, candles: list) -> MarketRegime:
        if len(candles) < 20:
            return MarketRegime.SIDEWAYS

        closes = [c["close"] for c in candles[-20:]]
        returns = np.diff(closes) / closes[:-1]

        volatility = np.std(returns) * np.sqrt(252 * 78)  # Annualized
        trend = np.mean(returns)

        if volatility > 0.4:
            return MarketRegime.HIGH_VOL
        elif trend > 0.0005:
            return MarketRegime.BULL
        elif trend < -0.0005:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
```

### 7.2 margin_paper_trading.py (ключевые части)

```python
"""Margin Paper Trading — симуляция маржинальной торговли."""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config_schema import load_config
from .storage import connect
from .engine import Engine
from .margin_risk_engine import MarginRiskEngine, MarketRegime
from .telegram import send_telegram

class MarginPaperTrader:
    STATE_FILE = Path("data/margin_paper_state.json")

    def __init__(
        self,
        initial_capital: float = 200_000,
        max_leverage: float = 3.0,
        max_positions: int = 3,
        resume: bool = True,
    ):
        self.config = load_config()
        self.conn = connect(self.config.sqlite_path)
        self.engine = Engine(self.conn, self.config)
        self.risk_engine = MarginRiskEngine()

        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.max_positions = max_positions

        if resume and self.STATE_FILE.exists():
            self._load_state()
        else:
            self._init_state()

    def _init_state(self):
        self.state = {
            "initial_capital": self.initial_capital,
            "cash": self.initial_capital,
            "equity": self.initial_capital,
            "positions": {},
            "closed_trades": [],
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "consecutive_losses": 0,
            "kill_switch_active": False,
            "start_date": datetime.now().isoformat(),
        }
        self._save_state()

    def _save_state(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _load_state(self):
        with open(self.STATE_FILE) as f:
            self.state = json.load(f)

    def run_cycle(self) -> dict:
        """Один цикл торговли."""
        # Проверка Kill-Switch
        trading_state = self._get_trading_state()
        kill, reason = self.risk_engine.check_kill_switch(trading_state)

        if kill:
            self.state["kill_switch_active"] = True
            return {"status": "kill_switch", "reason": reason}

        # Получение сигналов
        signals = self.engine.run_once()

        # Обработка сигналов
        for signal in signals:
            if signal["p"] < 0.55:
                continue

            ticker = signal["secid"]
            direction = "LONG" if signal["signal_type"] == "BUY" else "SHORT"

            # Проверка лимита позиций
            if len(self.state["positions"]) >= self.max_positions:
                continue

            # Определение режима и плеча
            candles = self.engine.get_candles(ticker, limit=100)
            regime = self.risk_engine.detect_regime(candles)
            leverage = self.risk_engine.calculate_leverage(
                regime, signal.get("atr_pct", 0.02)
            )
            leverage = min(leverage, self.max_leverage)

            # Расчет размера позиции
            entry = signal["entry"]
            stop = signal["stop"]
            size = self.risk_engine.calculate_position_size(
                self.state["equity"], entry, stop, leverage
            )

            if size > 0:
                self._open_position(ticker, direction, entry, stop,
                                   signal["take"], size, signal, regime, leverage)

        # Проверка открытых позиций
        self._check_positions()

        # Обновление equity
        self._update_equity()
        self._save_state()

        return {
            "status": "ok",
            "equity": self.state["equity"],
            "positions": len(self.state["positions"]),
            "trades": len(self.state["closed_trades"]),
        }

    def run(self, duration_hours: float = 168):
        """Основной цикл торговли."""
        end_time = time.time() + duration_hours * 3600
        cycle = 0

        while time.time() < end_time:
            cycle += 1
            try:
                result = self.run_cycle()
                if cycle % 12 == 0:  # Каждые ~3 минуты
                    dd = (self.initial_capital - self.state["equity"]) / self.initial_capital
                    print(f"Cycle {cycle}: equity={self.state['equity']:,.0f}, "
                          f"DD={dd:.1%}, pos={len(self.state['positions'])}, "
                          f"trades={len(self.state['closed_trades'])}")
            except Exception as e:
                print(f"Error in cycle {cycle}: {e}")

            time.sleep(15)  # 15 секунд между циклами
```

### 7.3 features.py (полный)

```python
"""Feature Engineering — 29 технических индикаторов."""

import numpy as np
import pandas as pd

def build_feature_frame(candles: list) -> pd.DataFrame:
    """Построение DataFrame с фичами из свечей."""
    df = pd.DataFrame(candles)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    # Price returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    # Bollinger Bands
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R
    df["willr_14"] = -100 * (high_14 - df["close"]) / (high_14 - low_14)

    # CCI
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["cci_20"] = (tp - sma_tp) / (0.015 * mad)

    # MFI
    mf = tp * df["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfr = pos_mf / neg_mf
    df["mfi_14"] = 100 - (100 / (1 + mfr))

    # ADX
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14 = tr.rolling(14).mean()
    df["plus_di"] = 100 * (plus_dm.rolling(14).mean() / atr_14)
    df["minus_di"] = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
    df["adx_14"] = dx.rolling(14).mean()

    # EMA ratio
    ema_10 = df["close"].ewm(span=10).mean()
    ema_50 = df["close"].ewm(span=50).mean()
    df["ema_ratio"] = ema_10 / ema_50

    # OBV
    obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    df["obv_ratio"] = obv / obv.rolling(20).mean()

    # VWAP distance
    vwap = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_dist"] = (df["close"] - vwap) / vwap

    # Volume ratio
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Time features
    df["hour"] = df["ts"].dt.hour
    df["minute"] = df["ts"].dt.minute
    df["day_of_week"] = df["ts"].dt.dayofweek

    return df

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
    "atr_14", "atr_pct", "bb_width", "bb_pct",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "willr_14", "cci_20", "mfi_14",
    "adx_14", "plus_di", "minus_di", "ema_ratio",
    "obv_ratio", "vwap_dist", "volume_ratio",
    "hour", "minute", "day_of_week",
]
```

### 7.4 webapp.py (API endpoints)

```python
"""FastAPI Web Application."""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from typing import List, Optional
import json
from pathlib import Path

from .config_schema import load_config
from .storage import connect

app = FastAPI(title="MOEX Agent", version="2.0")

STATE_FILE = Path("data/margin_paper_state.json")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/status")
def get_status():
    config = load_config()
    conn = connect(config.sqlite_path)

    candles = conn.execute("SELECT COUNT(*) as cnt FROM candles").fetchone()["cnt"]
    alerts = conn.execute("SELECT COUNT(*) as cnt FROM alerts").fetchone()["cnt"]

    models = list(Path("models").glob("*.joblib"))

    return {
        "candles": candles,
        "alerts": alerts,
        "tickers": len(config.tickers),
        "models": [m.stem for m in models],
    }

@app.get("/api/equity")
def get_equity():
    if not STATE_FILE.exists():
        return {"error": "No trading state"}

    with open(STATE_FILE) as f:
        state = json.load(f)

    trades = state.get("closed_trades", [])
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) <= 0]

    total_profit = sum(t["pnl"] for t in winning) if winning else 0
    total_loss = abs(sum(t["pnl"] for t in losing)) if losing else 0

    return {
        "equity": state.get("equity", state.get("cash", 0)),
        "initial_capital": state.get("initial_capital", 200000),
        "total_pnl": state.get("equity", 0) - state.get("initial_capital", 200000),
        "daily_pnl": state.get("daily_pnl", 0),
        "weekly_pnl": state.get("weekly_pnl", 0),
        "win_rate": len(winning) / len(trades) if trades else 0,
        "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "kill_switch_active": state.get("kill_switch_active", False),
        "consecutive_losses": state.get("consecutive_losses", 0),
    }

@app.get("/api/trades")
def get_trades(limit: int = Query(default=50)):
    if not STATE_FILE.exists():
        return []

    with open(STATE_FILE) as f:
        state = json.load(f)

    trades = state.get("closed_trades", [])[-limit:]

    result = []
    for t in reversed(trades):
        result.append({
            "id": t.get("id", 0),
            "ticker": t.get("ticker", ""),
            "direction": t.get("direction", ""),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "quantity": t.get("quantity", 0),
            "pnl": t.get("pnl", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "entry_time": t.get("entry_time", ""),
            "exit_time": t.get("exit_time", ""),
            "horizon": t.get("horizon", ""),
            "reason": _get_trade_reason(t),
        })

    return result

def _get_trade_reason(trade: dict) -> str:
    """Генерация описания причины сделки."""
    direction = trade.get("direction", "LONG")
    horizon = trade.get("horizon", "5m")
    regime = trade.get("regime", "SIDEWAYS")
    leverage = trade.get("leverage", 1.0)
    probability = trade.get("probability", 0.55)

    return (
        f"ML-модель: {direction} на {horizon} "
        f"(p={probability:.0%}) | "
        f"Режим: {regime} | "
        f"Плечо: {leverage}x"
    )

@app.get("/api/positions")
def get_positions():
    if not STATE_FILE.exists():
        return []

    with open(STATE_FILE) as f:
        state = json.load(f)

    positions = state.get("positions", {})

    return [
        {
            "ticker": ticker,
            "direction": pos.get("direction", ""),
            "entry_price": pos.get("entry_price", 0),
            "quantity": pos.get("quantity", 0),
            "entry_time": pos.get("entry_time", ""),
            "unrealized_pnl": pos.get("unrealized_pnl", 0),
        }
        for ticker, pos in positions.items()
    ]

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MOEX Agent Dashboard</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; margin: 20px; background: #1a1a2e; color: #eee; }
            .card { background: #16213e; padding: 20px; border-radius: 10px; margin: 10px 0; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric { font-size: 2em; color: #0f0; }
            .negative { color: #f00; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
            .positive { color: #0f0; }
            .negative { color: #f00; }
        </style>
    </head>
    <body>
        <h1>MOEX Agent Dashboard</h1>
        <div class="grid" id="metrics"></div>
        <div class="card">
            <h2>Последние сделки</h2>
            <table id="trades">
                <thead>
                    <tr>
                        <th>Тикер</th>
                        <th>Направление</th>
                        <th>P&L</th>
                        <th>Время</th>
                        <th>Причина</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
        <script>
            async function update() {
                const equity = await (await fetch('/api/equity')).json();
                const trades = await (await fetch('/api/trades?limit=20')).json();

                document.getElementById('metrics').innerHTML = `
                    <div class="card">
                        <h3>Equity</h3>
                        <div class="metric">${equity.equity?.toLocaleString()} ₽</div>
                    </div>
                    <div class="card">
                        <h3>P&L</h3>
                        <div class="metric ${equity.total_pnl >= 0 ? 'positive' : 'negative'}">
                            ${equity.total_pnl >= 0 ? '+' : ''}${equity.total_pnl?.toLocaleString()} ₽
                        </div>
                    </div>
                    <div class="card">
                        <h3>Win Rate</h3>
                        <div class="metric">${(equity.win_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="card">
                        <h3>Сделок</h3>
                        <div class="metric">${equity.total_trades}</div>
                    </div>
                `;

                const tbody = document.querySelector('#trades tbody');
                tbody.innerHTML = trades.map(t => `
                    <tr>
                        <td>${t.ticker}</td>
                        <td>${t.direction}</td>
                        <td class="${t.pnl >= 0 ? 'positive' : 'negative'}">
                            ${t.pnl >= 0 ? '+' : ''}${t.pnl?.toFixed(2)} ₽
                        </td>
                        <td>${t.exit_time?.slice(11, 16) || ''}</td>
                        <td>${t.reason}</td>
                    </tr>
                `).join('');
            }

            update();
            setInterval(update, 30000);
        </script>
    </body>
    </html>
    """
```

---

## 8. База данных

### 8.1 Схема SQLite

```sql
-- Свечи (основная таблица данных)
CREATE TABLE IF NOT EXISTS candles (
    secid TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    value REAL,
    volume INTEGER,
    interval INTEGER DEFAULT 1,
    PRIMARY KEY (secid, ts, interval)
);

CREATE INDEX IF NOT EXISTS idx_candles_secid_ts
ON candles(secid, ts DESC);

-- Сигналы/алерты
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts TEXT NOT NULL,
    secid TEXT NOT NULL,
    horizon TEXT,
    p REAL,
    signal_type TEXT,
    entry REAL,
    take REAL,
    stop REAL,
    sent INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alerts_created
ON alerts(created_ts DESC);
```

### 8.2 Bootstrap (загрузка данных)

```python
def bootstrap_recent(conn, config, days=7):
    """Загрузка данных за последние N дней."""
    from datetime import datetime, timedelta
    from .moex_iss import fetch_candles

    end = datetime.now()
    start = end - timedelta(days=days)

    for ticker in config.tickers:
        candles = fetch_candles(
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            interval=1  # 1-минутные свечи
        )

        for c in candles:
            conn.execute("""
                INSERT OR REPLACE INTO candles
                (secid, ts, open, high, low, close, value, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, c["begin"], c["open"], c["high"],
                c["low"], c["close"], c["value"], c["volume"], 1
            ))

        conn.commit()
        print(f"Loaded {len(candles)} candles for {ticker}")
```

---

## 9. Запуск и деплой

### 9.1 Локальный запуск

```bash
# Установка
pip install -r requirements.txt

# Конфигурация
cp config.yaml.replit config.yaml

# Bootstrap данных (7 дней)
python -m moex_agent.bootstrap --days 7

# Запуск
python main.py
```

### 9.2 Replit запуск

1. Загрузить ZIP архив
2. Установить Secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Нажать Run

### 9.3 main.py (entry point)

```python
"""MOEX Agent — Entry Point."""
import os
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent")

def ensure_directories():
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

def init_database():
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect

    config = load_config()
    conn = connect(config.sqlite_path)

    cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
    count = cur.fetchone()["cnt"]

    if count < 10000:
        logger.info(f"Database has {count} candles, bootstrapping...")
        from moex_agent.bootstrap import bootstrap_recent
        bootstrap_recent(conn, config, days=7)

    conn.close()

def run_trading():
    time.sleep(10)
    from moex_agent.margin_paper_trading import MarginPaperTrader
    trader = MarginPaperTrader(initial_capital=200000, resume=True)
    trader.run(duration_hours=168)

def main():
    logger.info("MOEX Agent Starting...")
    ensure_directories()
    init_database()

    # Trading в фоне
    threading.Thread(target=run_trading, daemon=True).start()

    # Web server
    import uvicorn
    from moex_agent.webapp import app
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
```

---

## 10. Troubleshooting

### Ошибка: No module named 'yaml'
```bash
pip install pyyaml
```

### Ошибка: telegram.bot_token is required
Установить Secrets в Replit UI или в config.yaml

### Ошибка: No candles in database
```bash
python -m moex_agent.bootstrap --days 7
```

### MOEX API timeout
Автоматически переподключается. Если частые ошибки — уменьшить количество тикеров.

### Kill-Switch активирован
Проверить `data/margin_paper_state.json`:
```json
{
  "kill_switch_active": true,
  "consecutive_losses": 5
}
```
Сбросить: изменить `kill_switch_active` на `false` и `consecutive_losses` на `0`.

---

## Контакты и поддержка

- **Telegram:** @your_bot (для сигналов)
- **GitHub:** https://github.com/your_repo
- **Документация:** README_REPLIT.md

---

*Создано: 2026-01-21*
*Версия: 2.0*
*Модели: Walk-Forward validated, 56% WR, PF > 2.3*

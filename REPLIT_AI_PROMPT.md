# MOEX Agent — Промпт для Replit AI

## Состояние проекта: 85% готов

### Что СДЕЛАНО:

| Компонент | Статус | Описание |
|-----------|--------|----------|
| ML Модели | ✅ 100% | 4 модели обучены (5m, 10m, 30m, 1h), Walk-Forward валидация |
| Риск-менеджмент | ✅ 100% | Kill-Switch, Dynamic Leverage, Market Regimes |
| Paper Trading | ✅ 100% | Маржинальная торговля с контролем рисков |
| MOEX API | ✅ 100% | Получение свечей, auto-retry |
| Feature Engineering | ✅ 100% | 29 технических индикаторов |
| Telegram | ✅ 100% | Уведомления о сделках |
| Web API | ✅ 100% | FastAPI endpoints |
| Dashboard | ⚠️ 80% | Базовый HTML, нужна доработка |
| База данных | ✅ 100% | SQLite, auto-bootstrap |

### Что НЕ СДЕЛАНО / НУЖНО ДОРАБОТАТЬ:

| Задача | Приоритет | Описание |
|--------|-----------|----------|
| Dashboard улучшения | HIGH | Добавить графики equity curve, улучшить дизайн |
| WebSocket real-time | MEDIUM | Real-time обновления без перезагрузки |
| Бэктест визуализация | MEDIUM | Графики результатов бэктеста |
| Логирование в UI | LOW | Показ логов в dashboard |
| Mobile responsive | LOW | Адаптация под мобильные |

---

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                     MOEX Agent System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DATA LAYER                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  moex_iss.py │───▶│  storage.py  │───▶│ bootstrap.py │   │
│  │  MOEX API    │    │   SQLite     │    │  Load data   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  ML LAYER                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ features.py  │───▶│  labels.py   │───▶│ predictor.py │   │
│  │ 29 indicators│    │  Target var  │    │  Inference   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  SIGNAL LAYER                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  anomaly.py  │───▶│  engine.py   │───▶│   alerts     │   │
│  │  Detection   │    │  Pipeline    │    │  Generation  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  RISK LAYER                                                  │
│  ┌──────────────────────────┐    ┌──────────────────────┐   │
│  │  margin_risk_engine.py   │◄──▶│    bcs_broker.py     │   │
│  │  Kill-Switch, Leverage   │    │    Broker Limits     │   │
│  └──────────────────────────┘    └──────────────────────┘   │
│                                                              │
│  TRADING LAYER                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           margin_paper_trading.py                     │   │
│  │  Open/Close positions, Track P&L, State management   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  INTERFACE LAYER                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  webapp.py   │    │ telegram.py  │    │  main.py     │   │
│  │  FastAPI     │    │  Notify      │    │  Entry point │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ML Модели

### Характеристики:
- **Алгоритм:** GradientBoostingClassifier
- **Валидация:** Walk-Forward (честная, без утечки данных)
- **Features:** 29 технических индикаторов
- **Данные:** 32.6M свечей для обучения

### Метрики моделей:

| Модель | Win Rate | Profit Factor | Trades | Файл |
|--------|----------|---------------|--------|------|
| 5m | 56.8% | 2.33 | 1,247 | `models/model_time_5m.joblib` |
| 10m | 56.0% | 2.31 | 1,183 | `models/model_time_10m.joblib` |
| 30m | 56.0% | 2.39 | 1,156 | `models/model_time_30m.joblib` |
| 1h | 55.4% | 2.39 | 1,089 | `models/model_time_1h.joblib` |

### Отключенные модели (слишком длинный горизонт):
- `model_time_1d.joblib` — дневной (не используется)
- `model_time_1w.joblib` — недельный (не используется)

### Использование модели:

```python
from moex_agent.predictor import Predictor

predictor = Predictor("models")
probability = predictor.predict(ticker="SBER", horizon="5m", features=feature_dict)
# probability > 0.55 → сигнал на покупку
# probability < 0.45 → сигнал на продажу
```

---

## Движок (Engine)

### Файл: `moex_agent/engine.py`

### Основной pipeline:

```python
class Engine:
    def run_once(self) -> List[dict]:
        """Один цикл генерации сигналов."""
        signals = []

        for ticker in self.config.tickers:
            # 1. Получить свечи
            candles = self.fetch_candles(ticker)

            # 2. Построить фичи
            features = build_feature_frame(candles)

            # 3. Для каждого горизонта
            for horizon in ["5m", "10m", "30m", "1h"]:
                # 4. Предсказание ML
                prob = self.predictor.predict(ticker, horizon, features)

                # 5. Если вероятность высокая — сигнал
                if prob > 0.55:
                    signal = self.create_signal(ticker, horizon, prob, "BUY")
                    signals.append(signal)
                elif prob < 0.45:
                    signal = self.create_signal(ticker, horizon, 1-prob, "SELL")
                    signals.append(signal)

        return signals
```

### Ключевые методы:

| Метод | Описание |
|-------|----------|
| `run_once()` | Один цикл генерации сигналов |
| `fetch_candles(ticker)` | Получить свечи из БД/API |
| `create_signal(...)` | Создать объект сигнала |
| `save_alert(signal)` | Сохранить в БД |

---

## Ядро (Core Components)

### 1. Риск-менеджмент (`margin_risk_engine.py`)

```python
class MarginRiskEngine:
    # Лимиты
    MAX_LOSS_PER_TRADE = 0.005   # 0.5%
    MAX_DAILY_LOSS = 0.02        # 2%
    MAX_WEEKLY_LOSS = 0.05       # 5%
    MAX_DRAWDOWN = 0.10          # 10%
    CONSECUTIVE_LOSSES_LIMIT = 5

    def check_kill_switch(self, state) -> tuple[bool, str]:
        """Проверка всех лимитов. True = СТОП торговля."""

    def calculate_leverage(self, regime, volatility) -> float:
        """Динамическое плечо 0.5x - 2.0x."""

    def detect_regime(self, candles) -> MarketRegime:
        """BULL / BEAR / SIDEWAYS / HIGH_VOL"""
```

### 2. Paper Trading (`margin_paper_trading.py`)

```python
class MarginPaperTrader:
    STATE_FILE = "data/margin_paper_state.json"

    def run_cycle(self) -> dict:
        """Один цикл торговли."""
        # 1. Проверить Kill-Switch
        # 2. Получить сигналы от Engine
        # 3. Открыть позиции
        # 4. Проверить открытые позиции (TP/SL)
        # 5. Сохранить состояние

    def run(self, duration_hours=168):
        """Основной loop (по умолчанию 1 неделя)."""
```

### 3. Features (`features.py`)

29 индикаторов:
- Returns: ret_1, ret_5, ret_10, ret_20, log_ret_1
- Volatility: atr_14, atr_pct, bb_width, bb_pct
- Momentum: rsi_14, macd, macd_signal, macd_hist, stoch_k, stoch_d, willr_14, cci_20, mfi_14
- Trend: adx_14, plus_di, minus_di, ema_ratio
- Volume: obv_ratio, vwap_dist, volume_ratio
- Time: hour, minute, day_of_week

---

## Навигация по файлам

### Основные файлы (ЧИТАТЬ В ПЕРВУЮ ОЧЕРЕДЬ):

| Файл | Описание | Строк |
|------|----------|-------|
| `main.py` | Entry point, запуск всего | 145 |
| `moex_agent/webapp.py` | Web API + Dashboard | 850 |
| `moex_agent/margin_paper_trading.py` | Paper trading | 700 |
| `moex_agent/margin_risk_engine.py` | Риск-менеджмент | 650 |
| `moex_agent/engine.py` | Основной pipeline | 380 |

### Вспомогательные:

| Файл | Описание |
|------|----------|
| `moex_agent/features.py` | 29 индикаторов |
| `moex_agent/predictor.py` | Загрузка ML моделей |
| `moex_agent/storage.py` | SQLite операции |
| `moex_agent/moex_iss.py` | MOEX API клиент |
| `moex_agent/telegram.py` | Telegram уведомления |
| `moex_agent/config_schema.py` | Pydantic конфигурация |
| `moex_agent/bcs_broker.py` | Лимиты брокера BCS |

### Документация:

| Файл | Описание |
|------|----------|
| `KNOWLEDGE_BASE.md` | Полная база знаний (41 KB) |
| `README_REPLIT.md` | Документация для Replit |
| `ARCHITECTURE.md` | Архитектура системы |
| `QUICK_START.md` | Быстрый старт |

---

## API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | HTML Dashboard |
| `/api/health` | GET | Health check |
| `/api/status` | GET | Статус (candles, alerts, models) |
| `/api/signals` | GET | Генерация сигналов |
| `/api/alerts` | GET | Последние алерты |
| `/api/trades` | GET | История сделок с P&L |
| `/api/equity` | GET | Equity, Win Rate, PF |
| `/api/positions` | GET | Открытые позиции |
| `/api/tickers` | GET | Список тикеров |
| `/api/candles/{ticker}` | GET | Свечи по тикеру |

---

## Конфигурация

### Environment Variables (Secrets):
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
PORT=8080
```

### config.yaml:
```yaml
tickers:
  - SBER
  - GAZP
  - LKOH
  # ... 46 тикеров

sqlite_path: "data/moex_agent.sqlite"
models_dir: "models"

telegram:
  enabled: true

horizons:
  - "5m"
  - "10m"
  - "30m"
  - "1h"

disabled_horizons:
  - "1d"
  - "1w"
```

---

## Текущее состояние торговли

```json
{
  "equity": 200002,
  "initial_capital": 200000,
  "total_pnl": +2 RUB,
  "closed_trades": 5,
  "win_rate": 60%,
  "consecutive_losses": 0,
  "kill_switch_active": false
}
```

Последние сделки:
1. SFIN SHORT: -3.35 ₽ (stop)
2. MGNT SHORT: +4.68 ₽ (take)
3. SMLT SHORT: +2.83 ₽ (take)
4. SFIN SHORT: -4.10 ₽ (stop)
5. HEAD SHORT: +1.68 ₽ (timeout)

---

## Что нужно сделать в Replit

### Приоритет HIGH:

1. **Улучшить Dashboard** (`webapp.py`):
   - Добавить график Equity Curve (Chart.js или Plotly)
   - Показать открытые позиции в реальном времени
   - Улучшить таблицу сделок (фильтры, сортировка)

2. **Проверить запуск**:
   - Убедиться что bootstrap работает
   - Проверить Telegram уведомления

### Приоритет MEDIUM:

3. **WebSocket** для real-time обновлений
4. **Графики** результатов бэктеста
5. **Мобильная** адаптация CSS

### Приоритет LOW:

6. **Логи** в UI
7. **Настройки** через UI (изменение тикеров, лимитов)

---

## Запуск

```bash
# Установка зависимостей (автоматически в Replit)
pip install -r requirements.txt

# Запуск
python main.py
```

При первом запуске:
1. Создаются папки `data/`, `models/`
2. Bootstrap загружает 7 дней данных с MOEX
3. Запускается web server на порту 8080
4. Запускается paper trading в фоне

---

## Частые проблемы

| Проблема | Решение |
|----------|---------|
| No candles | `python -m moex_agent.bootstrap --days 7` |
| Telegram не работает | Проверить Secrets |
| MOEX timeout | Автоматически retry, подождать |
| Kill-Switch active | Сбросить в `data/margin_paper_state.json` |

---

## Контекст для AI

Это **торговая система для Московской биржи**. Цель — автоматическая генерация сигналов на покупку/продажу акций с ML-прогнозированием и контролем рисков.

**Ключевые принципы:**
- Никогда не рисковать более 0.5% капитала на сделку
- Kill-Switch при 5 убытках подряд или 10% просадке
- Только краткосрочные горизонты (5m, 10m, 30m, 1h)
- Paper trading (без реальных денег)

**Технологии:**
- Python 3.11+
- FastAPI + Uvicorn
- SQLite
- scikit-learn (GradientBoosting)
- pandas, numpy

---

*Обновлено: 2026-01-22*

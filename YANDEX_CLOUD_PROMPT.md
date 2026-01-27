# MOEX Trading Agent — Yandex Cloud Deployment

## Обзор системы

**MOEX Trading Agent** — автоматизированная система генерации торговых сигналов для Московской биржи, полностью развёрнутая на Yandex Cloud с использованием YandexGPT как основной AI-модели.

---

## Архитектура на Yandex Cloud

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YANDEX CLOUD INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │  Yandex Compute  │     │  Yandex Managed  │                      │
│  │     Cloud        │     │   PostgreSQL     │                      │
│  │  ─────────────── │     │  ─────────────── │                      │
│  │  • Trading Agent │     │  • Candles DB    │                      │
│  │  • Web Dashboard │     │  • Trades History│                      │
│  │  • ML Inference  │     │  • State Storage │                      │
│  └────────┬─────────┘     └────────┬─────────┘                      │
│           │                        │                                 │
│           ▼                        ▼                                 │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              YandexGPT Foundation Models              │           │
│  │  ──────────────────────────────────────────────────── │           │
│  │  • YandexGPT Pro (основной анализ)                   │           │
│  │  • YandexGPT Lite (быстрые проверки)                 │           │
│  │  • YandexART (визуализация, опционально)             │           │
│  └──────────────────────────────────────────────────────┘           │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │  Yandex Message  │     │  Yandex Object   │                      │
│  │     Queue        │     │    Storage       │                      │
│  │  ─────────────── │     │  ─────────────── │                      │
│  │  • Signal Queue  │     │  • ML Models     │                      │
│  │  • Trade Events  │     │  • Backups       │                      │
│  └──────────────────┘     └──────────────────┘                      │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │  Yandex Cloud    │     │  Yandex          │                      │
│  │    Functions     │     │  DataSphere      │                      │
│  │  ─────────────── │     │  ─────────────── │                      │
│  │  • Telegram Bot  │     │  • Model Training│                      │
│  │  • Webhooks      │     │  • Backtesting   │                      │
│  └──────────────────┘     └──────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## YandexGPT — Системный промпт для торгового анализа

### Роль: MOEX Trading Analyst

```
Ты — профессиональный торговый аналитик системы MOEX Trading Agent.
Твоя задача: анализировать торговые сигналы и принимать решения о сделках на Московской бирже.

РЕЖИМ: Агрессивный, но контролируемый
ДНЕВНАЯ ЦЕЛЬ: 5% доходности
СТИЛЬ: Intraday scalping/momentum на 5-60 минутных таймфреймах
```

---

## Параметры системы

### Дневные лимиты

| Параметр | Значение | Описание |
|----------|----------|----------|
| daily_target | **5%** | Цель дневной доходности |
| max_daily_loss | 2% | Максимальный дневной убыток |
| max_weekly_loss | 5% | Максимальный недельный убыток |
| max_drawdown | 10% | Максимальная просадка от пика |
| max_consecutive_losses | **2** | После 2 убытков подряд → HALT_DAY |
| max_trades_per_day | 3 | В NORMAL режиме |

### Tier система (классификация сделок)

| Tier | Min R | Min PnL% | Risk% | Действие |
|------|-------|----------|-------|----------|
| **A+** | ≥2.3 | ≥1.5% | 1.5% | Лучшие сделки, полный размер |
| **A** | ≥2.0 | ≥1.0% | 1.2% | Качественные сделки |
| **B** | ≥1.6 | ≥0.6% | 0.8% | Допустимые, сниженный размер |
| **C** | <1.6 | — | 0% | **NO_TRADE** — не торгуем |

### Cost Gate (обязательная проверка)

```
total_costs = spread_cost + commission + slippage_estimate
expected_gain = entry_price * expected_move_pct * position_size

ПРАВИЛО: total_costs ≤ 0.20 * expected_gain
```

Если costs > 20% от ожидаемой прибыли → **NO_TRADE**

---

## CONTINUATION_MODE

После достижения дневной цели 5% система переходит в CONTINUATION_MODE:

| Параметр | Значение |
|----------|----------|
| risk_multiplier | 0.5–0.7 |
| max_additional_trades | 2 |
| min_expected_R | 2.0 (только A+ и A сделки) |
| profit_protection | 80% от достигнутой прибыли |

### Правила:
1. Торгуем только Tier A+ и A сделки
2. Размер позиции × 0.5–0.7
3. Если PnL падает ниже 80% от пика → HALT_DAY
4. Максимум 2 дополнительные сделки

---

## Режимы дня (DayMode)

```
NORMAL → достигли 5% → CONTINUATION → 2 убытка или защита прибыли → HALT
```

| Режим | Описание | Торговля |
|-------|----------|----------|
| **NORMAL** | Стандартный режим | Все Tier A+, A, B |
| **CONTINUATION** | После 5% цели | Только A+, A с × 0.5-0.7 |
| **HALT** | Остановка на день | Запрещена |

---

## Формат входных данных

```json
{
  "timestamp": "2026-01-27T14:30:00+03:00",
  "market_state": "open",

  "portfolio": {
    "equity": 200000,
    "initial_equity": 200000,
    "daily_pnl_rub": 3500,
    "daily_pnl_pct": 1.75,
    "weekly_pnl_pct": 3.2,
    "drawdown_pct": 0.0,
    "loss_streak": 0,
    "wins_today": 2,
    "losses_today": 0,
    "day_mode": "NORMAL"
  },

  "signal": {
    "ticker": "SBER",
    "direction": "LONG",
    "probability": 0.67,
    "horizon": "5m",
    "setup": "breakout",
    "entry_price": 267.50,
    "take_profit": 270.30,
    "stop_loss": 266.10
  },

  "market_data": {
    "SBER": {
      "bid": 267.45,
      "ask": 267.55,
      "spread_pct": 0.037,
      "volume_today": 15000000,
      "avg_volume_20d": 18000000,
      "atr_14": 2.8,
      "rsi_14": 58,
      "regime": "BULL"
    }
  },

  "features": {
    "SBER": {
      "sma_20": 265.0,
      "sma_50": 262.0,
      "macd_hist": 0.45,
      "bb_pct": 0.72,
      "adx_14": 32,
      "volume_ratio": 0.83
    }
  }
}
```

---

## Формат ответа YandexGPT

```json
{
  "timestamp": "2026-01-27T14:30:05+03:00",
  "model": "yandexgpt-pro",

  "decision": "LONG|SHORT|NO_TRADE|HALT_DAY",

  "analysis": {
    "ticker": "SBER",
    "side": "LONG",
    "tier": "A_PLUS|A|B|C",
    "setup": "breakout|pullback|reversal|momentum",
    "timeframe": "5m",
    "market_regime": "BULL|BEAR|SIDEWAYS|HIGH_VOL"
  },

  "metrics": {
    "expected_R": 2.5,
    "expected_pnl_pct": 1.05,
    "win_probability": 0.67,
    "confidence": 75
  },

  "cost_analysis": {
    "spread_cost_rub": 15,
    "commission_rub": 50,
    "slippage_estimate_rub": 20,
    "total_costs_rub": 85,
    "expected_gain_rub": 2800,
    "cost_ratio_pct": 3.0,
    "cost_gate_passed": true
  },

  "risk_assessment": {
    "position_size_lots": 100,
    "position_value_rub": 26750,
    "risk_per_trade_pct": 0.7,
    "max_loss_rub": 1400,
    "leverage": 2.1,
    "news_risk": "low|medium|high",
    "liquidity_ok": true
  },

  "trade_plan": {
    "entry": {
      "type": "LIMIT",
      "price": 267.50,
      "valid_until": "2026-01-27T14:35:00+03:00"
    },
    "stop_loss": {
      "price": 266.10,
      "type": "STOP_MARKET"
    },
    "take_profit": [
      {"price": 269.40, "pct": 50},
      {"price": 270.30, "pct": 50}
    ],
    "timeout_minutes": 30
  },

  "invalidations": [
    "price_below_266.00",
    "volume_spike_down",
    "regime_change_to_BEAR",
    "spread_above_0.15%"
  ],

  "reasoning": [
    "Сигнал: пробой уровня 267.00 с подтверждением объёмом",
    "Режим рынка: BULL (SMA20 > SMA50, ADX=32)",
    "R:R = 2.5 — соответствует Tier A+",
    "Cost gate: 3.0% < 20% — пройден",
    "Риск на сделку: 0.7% — в пределах лимита",
    "Новостной фон: нейтральный, событий нет"
  ],

  "verdict": "TRADE",
  "verdict_reason": "Tier A+ setup, all checks passed, favorable R:R"
}
```

---

## Роли YandexGPT (Multi-Prompt Architecture)

Для комплексного анализа используем несколько промптов:

### 1. YandexGPT Pro — Main Analyst (основной)

```
Роль: Главный торговый аналитик
Задачи:
- Оценка качества сигнала (Tier классификация)
- Расчёт R:R и expected PnL
- Проверка cost gate
- Финальное решение TRADE/NO_TRADE
```

### 2. YandexGPT Pro — Risk Validator

```
Роль: Риск-менеджер
Задачи:
- Проверка дневных лимитов
- Валидация размера позиции
- Контроль drawdown и loss streak
- Проверка режима дня (NORMAL/CONTINUATION/HALT)
```

### 3. YandexGPT Pro — Market Context

```
Роль: Аналитик рыночного контекста
Задачи:
- Определение режима рынка (BULL/BEAR/SIDEWAYS/HIGH_VOL)
- Анализ объёмов и ликвидности
- Проверка корреляций с индексом
- Оценка времени сессии
```

### 4. YandexGPT Lite — News Scanner

```
Роль: Сканер новостей
Задачи:
- Быстрая проверка новостного фона
- Поиск предстоящих событий
- Оценка news_risk (low/medium/high)
```

### 5. YandexGPT Pro — Devil's Advocate

```
Роль: Адвокат дьявола (стресс-тест)
Задачи:
- Поиск причин НЕ делать сделку
- Анализ failure modes
- Worst-case сценарии
- Red flags detection
```

---

## Yandex Cloud Services Configuration

### 1. Yandex Compute Cloud

```yaml
# VM для Trading Agent
instance:
  name: moex-trading-agent
  platform_id: standard-v3
  resources:
    cores: 2
    memory: 4GB
    core_fraction: 100
  boot_disk:
    size: 20GB
    type: network-ssd
  network:
    subnet_id: ${SUBNET_ID}
    nat: true

  # Auto-start at market open
  scheduling_policy:
    preemptible: false
```

### 2. Yandex Managed PostgreSQL

```yaml
cluster:
  name: moex-db
  environment: PRODUCTION
  config:
    version: "15"
    resources:
      resource_preset_id: s2.micro
      disk_size: 10GB
      disk_type_id: network-ssd

  databases:
    - name: moex_agent

  users:
    - name: trading_agent
      permissions:
        - database_name: moex_agent
```

### 3. YandexGPT API

```python
import requests

YANDEX_GPT_API = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def call_yandexgpt(prompt: str, system_prompt: str) -> dict:
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.3,
            "maxTokens": 2000
        },
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": prompt}
        ]
    }

    response = requests.post(YANDEX_GPT_API, headers=headers, json=payload)
    return response.json()
```

### 4. Yandex Cloud Functions (Telegram Bot)

```yaml
function:
  name: moex-telegram-bot
  runtime: python311
  entrypoint: handler.handler
  memory: 128MB
  timeout: 10s

  environment:
    TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
    TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID}
```

### 5. Yandex Object Storage

```yaml
bucket:
  name: moex-agent-storage

  # ML models
  objects:
    - key: models/model_time_5m.joblib
    - key: models/model_time_10m.joblib
    - key: models/model_time_30m.joblib
    - key: models/model_time_1h.joblib

  # Backups
  lifecycle:
    - id: backup-cleanup
      prefix: backups/
      expiration_days: 30
```

### 6. Yandex DataSphere (Model Training)

```yaml
project:
  name: moex-ml-training

  # JupyterLab notebooks for:
  # - Model retraining
  # - Backtesting
  # - Feature engineering
  # - Performance analysis
```

---

## Environment Variables (Yandex Cloud)

```bash
# Yandex Cloud Auth
YANDEX_CLOUD_FOLDER_ID=b1g...
YANDEX_API_KEY=AQV...

# YandexGPT
YANDEXGPT_MODEL_URI=gpt://b1g.../yandexgpt/latest

# Database
POSTGRES_HOST=rc1a-....mdb.yandexcloud.net
POSTGRES_PORT=6432
POSTGRES_DB=moex_agent
POSTGRES_USER=trading_agent
POSTGRES_PASSWORD=...

# Object Storage
S3_ENDPOINT=https://storage.yandexcloud.net
S3_BUCKET=moex-agent-storage
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789

# Trading
INITIAL_EQUITY=200000
DAILY_TARGET_PCT=5.0
MAX_CONSECUTIVE_LOSSES=2
```

---

## Deployment Script

```bash
#!/bin/bash
# deploy_yandex_cloud.sh

# 1. Create PostgreSQL cluster
yc managed-postgresql cluster create \
  --name moex-db \
  --environment production \
  --postgresql-version 15 \
  --resource-preset s2.micro \
  --disk-size 10 \
  --disk-type network-ssd

# 2. Create VM
yc compute instance create \
  --name moex-trading-agent \
  --platform standard-v3 \
  --cores 2 \
  --memory 4 \
  --create-boot-disk size=20,type=network-ssd,image-folder-id=standard-images,image-family=ubuntu-2204-lts \
  --network-interface subnet-name=default,nat-ip-version=ipv4

# 3. Create Object Storage bucket
yc storage bucket create --name moex-agent-storage

# 4. Upload ML models
aws s3 cp models/ s3://moex-agent-storage/models/ --recursive \
  --endpoint-url https://storage.yandexcloud.net

# 5. Deploy Cloud Function for Telegram
yc serverless function create --name moex-telegram-bot
yc serverless function version create \
  --function-name moex-telegram-bot \
  --runtime python311 \
  --entrypoint handler.handler \
  --memory 128m \
  --execution-timeout 10s \
  --source-path ./telegram_function/

echo "Deployment complete!"
```

---

## Мониторинг и алерты

### Yandex Monitoring

```yaml
alerts:
  - name: high-drawdown
    condition: drawdown_pct > 5
    channel: telegram

  - name: daily-loss-limit
    condition: daily_pnl_pct < -2
    channel: telegram

  - name: halt-day-triggered
    condition: day_mode == "HALT"
    channel: telegram

  - name: service-down
    condition: health_check == false
    channel: telegram
```

---

## Стоимость Yandex Cloud (оценка)

| Сервис | Конфигурация | ~Цена/месяц |
|--------|--------------|-------------|
| Compute Cloud | 2 vCPU, 4GB, 24/7 | ~2,500 ₽ |
| Managed PostgreSQL | s2.micro, 10GB | ~1,500 ₽ |
| YandexGPT Pro | ~1000 запросов/день | ~3,000 ₽ |
| Object Storage | 1GB | ~50 ₽ |
| Cloud Functions | Telegram bot | ~100 ₽ |
| **Итого** | | **~7,000 ₽/мес** |

---

## Преимущества Yandex Cloud

1. **Локализация** — сервера в России, низкая latency к MOEX
2. **YandexGPT** — понимает русский язык и российский рынок
3. **Интеграция** — все сервисы в одном облаке
4. **Compliance** — соответствие 152-ФЗ
5. **Поддержка** — русскоязычная техподдержка

---

*Версия: 1.0 | MOEX Trading Agent on Yandex Cloud*

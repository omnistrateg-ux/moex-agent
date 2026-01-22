# MOEX Agent V1 — Architecture

## Overview

MOEX Agent is a real-time trading signal generator for the Moscow Exchange (MOEX).
It detects price/volume anomalies, predicts continuation probability via ML models,
filters signals through risk rules and optionally LLM analysis, then sends alerts to Telegram.

## Module Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MOEX ISS API                                   │
│                    https://iss.moex.com/iss/                                │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ HTTP (retry + backoff)
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           moex_iss.py                                       │
│                  fetch_candles() + fetch_quote()                            │
│                  Session with Retry(total=6, backoff=0.8)                   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           storage.py                                        │
│                    SQLite (WAL mode, 64MB cache)                            │
│              connect() + upsert_many() + get_state()                        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            live.py                                          │
│                     Main Pipeline Loop                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 1: Fetch candles (parallel, ThreadPoolExecutor)                       │
│  Step 2: Fetch quotes (parallel)                                            │
│  Step 3: Upsert to SQLite                                                   │
│  Step 4: Compute anomalies         → anomaly.py                             │
│  Step 5: Build features            → features.py                            │
│  Step 6: Predict P(success)        → models/*.joblib                        │
│  Step 7: Risk gatekeeper           → risk.py                                │
│  Step 8: Qwen LLM filter           → qwen.py (optional)                     │
│  Step 9: Send Telegram alert       → telegram.py                            │
│  Step 10: Update cooldown state                                             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Telegram Bot API                                    │
│                   @birge12_bot → chat_id: 120171956                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
moex_agent/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── ARCHITECTURE.md          # This file
│
├── db/
│   └── schema.sql           # SQLite schema (candles, quotes, alerts, state)
│
├── data/
│   └── moex_agent.sqlite    # SQLite database (~200MB)
│
├── models/
│   ├── meta.json            # Model metadata
│   ├── model_time_5m.joblib
│   ├── model_time_10m.joblib
│   ├── model_time_30m.joblib
│   ├── model_time_1h.joblib
│   ├── model_time_1d.joblib
│   └── model_time_1w.joblib
│
└── moex_agent/
    ├── __init__.py
    ├── __main__.py          # CLI entrypoint (TODO: unified)
    │
    │ # Core modules
    ├── config.py            # YAML config loader
    ├── moex_iss.py          # MOEX ISS API client
    ├── storage.py           # SQLite operations
    │
    │ # ML pipeline
    ├── features.py          # Feature engineering (returns, ATR, VWAP)
    ├── labels.py            # Label generation (time-exit)
    ├── train.py             # Model training (HistGradientBoosting + Isotonic)
    ├── anomaly.py           # Anomaly detection (robust z-score)
    │
    │ # Signal processing
    ├── risk.py              # Risk gatekeeper (spread, turnover)
    ├── qwen.py              # LLM analysis (Ollama/Qwen)
    │
    │ # Notifications
    ├── telegram.py          # Telegram bot integration
    │
    │ # Entry points
    ├── live.py              # Main loop
    ├── bootstrap.py         # Historical data loader
    └── init_db.py           # Database initialization
```

## SQLite Schema

```sql
-- candles: 1-minute OHLCV data
CREATE TABLE candles (
  secid TEXT NOT NULL,          -- Ticker (e.g., 'SBER')
  board TEXT NOT NULL,          -- Board (e.g., 'TQBR')
  interval INTEGER NOT NULL,    -- 1=1min, 10=10min, 60=1h
  ts TEXT NOT NULL,             -- ISO timestamp
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  value REAL,                   -- Turnover in RUB
  volume REAL,                  -- Volume in shares
  PRIMARY KEY (secid, board, interval, ts)
);

-- quotes: Real-time bid/ask/last
CREATE TABLE quotes (
  secid TEXT NOT NULL,
  board TEXT NOT NULL,
  ts TEXT NOT NULL,
  last REAL,
  bid REAL,
  ask REAL,
  numtrades REAL,
  voltoday REAL,
  valtoday REAL,
  PRIMARY KEY (secid, board, ts)
);

-- alerts: Generated signals (for history/web UI)
CREATE TABLE alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts TEXT NOT NULL,
  secid TEXT NOT NULL,
  horizon TEXT NOT NULL,
  p REAL NOT NULL,
  signal_type TEXT NOT NULL,    -- 'price-exit' or 'time-exit'
  entry REAL,
  take REAL,
  stop REAL,
  ttl_minutes INTEGER,
  anomaly_score REAL,
  payload_json TEXT,
  sent INTEGER DEFAULT 0        -- 1 if Telegram sent
);

-- state: Key-value store
CREATE TABLE state (
  key TEXT PRIMARY KEY,
  value TEXT
);
```

## Config Structure (config.yaml)

```yaml
app:
  poll_seconds: 5              # Main loop interval
  cooldown_minutes: 30         # Per-ticker cooldown
  top_n_anomalies: 10          # Max anomalies per cycle
  max_workers: 20              # Parallel HTTP workers

storage:
  sqlite_path: "data/moex_agent.sqlite"

universe:
  engine: "stock"
  market: "shares"
  board: "TQBR"
  tickers:                     # List of tickers to monitor
    - SBER
    - GAZP
    # ... 45 tickers total

signals:
  cooldown_minutes: 30         # (duplicate of app.cooldown_minutes - TODO: consolidate)
  top_n_anomalies: 5
  horizons:
    - { name: "5m",  minutes: 5 }
    - { name: "10m", minutes: 10 }
    - { name: "30m", minutes: 30 }
    - { name: "1h",  minutes: 60 }
    - { name: "1d",  minutes: 1440 }
    - { name: "1w",  minutes: 10080 }
  p_threshold: 0.35            # Min probability to generate signal
  price_exit:
    enabled: true

risk:
  max_spread_bps: 200          # Max spread in basis points
  min_turnover_rub_5m: 1000000 # Min 5-min turnover in RUB

qwen:
  enabled: true
  ollama_url: "http://localhost:11434"
  model: "qwen2.5:7b-instruct"
  max_tokens: 500
  temperature: 0.3

telegram:
  enabled: true
  bot_token: "..."
  chat_id: "..."
  send_recommendations:
    - STRONG_BUY
    - BUY
    - STRONG_SELL
    - SELL
```

## Data Flow Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  MOEX    │───▶│  SQLite  │───▶│ Anomaly  │───▶│ Features │───▶│    ML    │
│  ISS     │    │  Storage │    │ Detector │    │  Builder │    │  Models  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                     │                              │
                                     │ AnomalyResult                │ P(success)
                                     ▼                              ▼
                               ┌──────────┐                   ┌──────────┐
                               │   Risk   │◀──────────────────│  Select  │
                               │Gatekeeper│                   │  Best H  │
                               └──────────┘                   └──────────┘
                                     │
                                     │ pass/fail
                                     ▼
                               ┌──────────┐
                               │   Qwen   │ (optional LLM filter)
                               │ Analysis │
                               └──────────┘
                                     │
                                     │ skip/proceed
                                     ▼
                               ┌──────────┐
                               │ Telegram │
                               │  Alert   │
                               └──────────┘
```

## ML Models

**Algorithm**: HistGradientBoostingClassifier + IsotonicRegression (calibration)

**Features** (10 total):
- `r_1m`, `r_5m`, `r_10m`, `r_30m`, `r_60m` — returns over various windows
- `turn_1m`, `turn_5m`, `turn_10m` — turnover windows
- `atr_14` — 14-period Average True Range
- `dist_vwap_atr` — distance from 30-min VWAP normalized by ATR

**Labels**: Binary (1 = price increased after H minutes, net of 8bps fee)

**Horizons**: 5m, 10m, 30m, 1h, 1d, 1w

**Calibration**: Models are calibrated, max probability ≈ 0.60 (not 1.0!)

## Anomaly Detection

Uses **robust z-score** based on Median Absolute Deviation (MAD):

```
z = (value - median) / (1.4826 * MAD)
```

**Scoring formula**:
```
score = |z_ret| + 0.3*clip(z_vol, 0, 4) + 0.2*clip(volume_spike-1, 0, 5)
      - 1.5*(spread > max_spread)
      - 0.5*(turnover < min_turnover)
      - 0.8*(z_ret > 1 and z_vol < -1)  # counter-trend penalty
```

## Known Issues (Pre-V1)

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | `KeyError` on missing config keys | Crash | TODO: Pydantic schema |
| 2 | `IndexError` in predict_proba | Crash | Partially fixed (p_pos1) |
| 3 | Weekend empty windows | No signals | TODO: anchor by MAX(ts) |
| 4 | ISS timeouts | Missing data | Fixed (Retry) |
| 5 | `spread_bps=None` | Risk check fails | Fixed (None check) |
| 6 | ResourceWarning on Ctrl+C | Unclean exit | Partially fixed |
| 7 | Duplicate config keys | Confusion | TODO: consolidate |

## CLI Commands (Target V1)

```bash
# Initialize database
python -m moex_agent init-db

# Load historical data
python -m moex_agent bootstrap --days 180

# Train models
python -m moex_agent train --horizons 5m,10m,30m,1h,1d,1w

# Run live loop
python -m moex_agent live

# Start web dashboard
python -m moex_agent web --port 8000

# Test Telegram
python -m moex_agent telegram-test "Hello from MOEX Agent!"
```

## Dependencies

```
requests>=2.31.0      # HTTP client
pandas>=2.1.0         # Data manipulation
numpy>=1.26.0         # Numerical
PyYAML>=6.0.1         # Config parsing
pydantic>=2.6.0       # Config validation (TODO)
scikit-learn>=1.4.0   # ML models
joblib>=1.3.2         # Model serialization
fastapi>=0.109.0      # Web UI (TODO)
uvicorn>=0.27.0       # ASGI server (TODO)
click>=8.1.0          # CLI framework (TODO)
```

## Changelog

### V0.9 (Current)
- 45 tickers, 5.6M candles
- 6 ML models (calibrated)
- Telegram integration
- Qwen LLM filter
- Parallel HTTP fetching

### V1.0 (Target)
- [ ] Pydantic config schema
- [ ] Unified CLI
- [ ] FastAPI web dashboard
- [ ] Graceful shutdown
- [ ] Weekend anchor fix
- [ ] LaunchAgent for macOS

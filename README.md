# MOEX Agent V1

Real-time trading signal generator for the Moscow Exchange (MOEX).

Detects price/volume anomalies, predicts continuation probability via ML models,
filters signals through risk rules and optionally LLM analysis, then sends alerts to Telegram.

## Features

- **Real-time monitoring**: Polls MOEX ISS API every 5 seconds
- **Anomaly detection**: Robust z-score based on Median Absolute Deviation (MAD)
- **ML predictions**: HistGradientBoosting + Isotonic calibration for probability estimates
- **Multi-horizon**: 5m, 10m, 30m, 1h, 1d, 1w signal horizons
- **Risk management**: Spread and turnover filters
- **LLM analysis**: Optional Qwen integration for signal filtering
- **Telegram alerts**: Real-time notifications with entry/take/stop levels
- **Web dashboard**: FastAPI-based monitoring interface

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp config.example.yaml config.yaml
# Edit config.yaml: set tickers, Telegram credentials, etc.

# 3. Initialize database
python -m moex_agent init-db

# 4. Load historical data (180 days)
python -m moex_agent bootstrap --days 180

# 5. Train ML models
python -m moex_agent train

# 6. Run live signal loop
python -m moex_agent live

# Or start web dashboard
python -m moex_agent web --port 8000
```

## CLI Commands

```bash
python -m moex_agent init-db              # Initialize database schema
python -m moex_agent bootstrap --days 180 # Load historical candle data
python -m moex_agent train                # Train ML models
python -m moex_agent live                 # Run live signal generation
python -m moex_agent live --once          # Run one cycle and exit
python -m moex_agent web --port 8000      # Start web dashboard
python -m moex_agent telegram-test "msg"  # Test Telegram integration
python -m moex_agent status               # Show system status
```

## Configuration

Create `config.yaml` in the project root:

```yaml
app:
  poll_seconds: 5
  cooldown_minutes: 30
  top_n_anomalies: 10
  max_workers: 20

storage:
  sqlite_path: "data/moex_agent.sqlite"

universe:
  engine: "stock"
  market: "shares"
  board: "TQBR"
  tickers:
    - SBER
    - GAZP
    - LKOH
    - YNDX
    # ... add more tickers

signals:
  p_threshold: 0.35
  horizons:
    - { name: "5m", minutes: 5 }
    - { name: "10m", minutes: 10 }
    - { name: "30m", minutes: 30 }
    - { name: "1h", minutes: 60 }
    - { name: "1d", minutes: 1440 }
    - { name: "1w", minutes: 10080 }
  price_exit:
    enabled: true
    take_atr: 0.8
    stop_atr: 0.6

risk:
  max_spread_bps: 200
  min_turnover_rub_5m: 1000000

qwen:
  enabled: false
  ollama_url: "http://localhost:11434"
  model: "qwen2.5:7b-instruct"

telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
  send_recommendations:
    - STRONG_BUY
    - BUY
    - STRONG_SELL
    - SELL
```

## Web Dashboard

Access the dashboard at `http://localhost:8000` after starting the web server.

**API Endpoints:**
- `GET /` - HTML dashboard
- `GET /api/health` - Health check
- `GET /api/status` - System status
- `GET /api/alerts` - Recent alerts
- `GET /api/signals` - Run single cycle and return signals
- `GET /api/tickers` - List of monitored tickers
- `GET /api/candles/{secid}` - Candles for a ticker

## macOS LaunchAgent (Autostart)

To run MOEX Agent automatically on workdays:

```bash
# Create logs directory
mkdir -p logs

# Copy LaunchAgent plist
cp com.moex-agent.plist ~/Library/LaunchAgents/

# Edit paths in the plist if needed
nano ~/Library/LaunchAgents/com.moex-agent.plist

# Load the agent
launchctl load ~/Library/LaunchAgents/com.moex-agent.plist

# Check status
launchctl list | grep moex

# Unload if needed
launchctl unload ~/Library/LaunchAgents/com.moex-agent.plist
```

## Project Structure

```
moex_agent/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── ARCHITECTURE.md          # Detailed architecture docs
├── README.md                # This file
├── com.moex-agent.plist     # macOS LaunchAgent
│
├── db/
│   └── schema.sql           # SQLite schema
│
├── data/
│   └── moex_agent.sqlite    # SQLite database
│
├── models/
│   ├── meta.json            # Model metadata
│   └── model_time_*.joblib  # Trained models
│
└── moex_agent/
    ├── __init__.py
    ├── __main__.py          # CLI entrypoint
    ├── config_schema.py     # Pydantic config validation
    ├── moex_iss.py          # MOEX ISS API client
    ├── storage.py           # SQLite operations
    ├── features.py          # Feature engineering
    ├── labels.py            # Label generation
    ├── train.py             # Model training
    ├── predictor.py         # Safe ML inference
    ├── anomaly.py           # Anomaly detection
    ├── risk.py              # Risk gatekeeper
    ├── engine.py            # Pipeline engine
    ├── qwen.py              # LLM analysis
    ├── telegram.py          # Telegram notifications
    └── webapp.py            # FastAPI web dashboard
```

## ML Models

**Algorithm**: HistGradientBoostingClassifier + IsotonicRegression (calibration)

**Features** (10 total):
- `r_1m`, `r_5m`, `r_10m`, `r_30m`, `r_60m` — returns over various windows
- `turn_1m`, `turn_5m`, `turn_10m` — turnover windows
- `atr_14` — 14-period Average True Range
- `dist_vwap_atr` — distance from 30-min VWAP normalized by ATR

**Labels**: Binary (1 = price increased after H minutes, net of 8bps fee)

**Note**: Models are calibrated, max probability ≈ 0.60 (not 1.0!). This is expected behavior for a calibrated classifier on a balanced dataset.

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

## Requirements

```
requests>=2.31.0
pandas>=2.1.0
numpy>=1.26.0
PyYAML>=6.0.1
pydantic>=2.6.0
scikit-learn>=1.4.0
joblib>=1.3.2
fastapi>=0.109.0
uvicorn>=0.27.0
```

## Qwen (Optional LLM Analysis)

To enable LLM-based signal filtering:

```bash
# Install Ollama
brew install ollama

# Pull Qwen model
ollama pull qwen2.5:7b-instruct

# Start Ollama server
ollama serve
```

Then set `qwen.enabled: true` in config.yaml.

## License

MIT License

## Disclaimer

This project is for educational purposes only. It does not guarantee profits, and any trades you make are at your own risk. Always do your own research before making investment decisions.

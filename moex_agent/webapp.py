"""
MOEX Agent Web Dashboard (FastAPI)

Provides REST API and HTML dashboard for monitoring trading signals and paper trading.

Endpoints:
    GET /              - HTML dashboard
    GET /api/health    - Health check
    GET /api/status    - System status
    GET /api/alerts    - Recent alerts
    GET /api/signals   - Run single cycle and return signals
    GET /api/trades    - Completed trades history
    GET /api/equity    - Equity and P&L summary
    GET /api/positions - Open positions

Usage:
    uvicorn moex_agent.webapp:app --port 8000
    # or
    python -m moex_agent web --port 8000
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Lazy imports to avoid circular dependencies
_config = None
_conn = None
_engine = None


def get_config():
    """Lazy load config."""
    global _config
    if _config is None:
        from .config_schema import load_config
        _config = load_config()
    return _config


def get_conn():
    """Lazy load database connection."""
    global _conn
    if _conn is None:
        from .storage import connect
        config = get_config()
        _conn = connect(config.sqlite_path)
    return _conn


def get_engine():
    """Lazy load pipeline engine."""
    global _engine
    if _engine is None:
        from .engine import PipelineEngine
        _engine = PipelineEngine(get_config())
        _engine.load_models()
    return _engine


# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    # Startup
    logging.info("MOEX Agent Web Dashboard starting...")
    yield
    # Shutdown
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
    from .moex_iss import close_session
    close_session()
    logging.info("MOEX Agent Web Dashboard stopped")


# FastAPI app
app = FastAPI(
    title="MOEX Agent",
    description="Trading Signal Generator for Moscow Exchange",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    timestamp: str


class StatusResponse(BaseModel):
    candles_count: int
    alerts_count: int
    tickers_count: int
    date_range: Optional[Dict[str, str]]
    models_loaded: List[str]
    telegram_enabled: bool
    qwen_enabled: bool


class AlertResponse(BaseModel):
    id: int
    created_ts: str
    secid: str
    horizon: str
    p: float
    signal_type: str
    entry: Optional[float]
    take: Optional[float]
    stop: Optional[float]
    ttl_minutes: Optional[int]
    anomaly_score: Optional[float]
    sent: bool


class SignalResponse(BaseModel):
    ticker: str
    direction: str
    horizon: str
    p: float
    signal_type: str
    entry: Optional[float]
    take: Optional[float]
    stop: Optional[float]
    anomaly: Dict[str, Any]


class CycleResponse(BaseModel):
    signals: List[SignalResponse]
    anomalies_count: int
    duration_ms: float
    errors: List[str]


class TradeResponse(BaseModel):
    """Completed trade info."""
    ticker: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: int
    leverage: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    horizon: str
    regime: str
    reason: str  # Основание для сделки


class PositionResponse(BaseModel):
    """Open position info."""
    ticker: str
    direction: str
    entry_time: str
    entry_price: float
    size: int
    leverage: float
    take: Optional[float]
    stop: Optional[float]
    horizon: str
    regime: str
    unrealized_pnl: float


def _get_trade_reason(trade: dict) -> str:
    """Generate human-readable reason for trade."""
    direction = trade.get("direction", "LONG")
    horizon = trade.get("horizon", "5m")
    regime = trade.get("regime", "UNKNOWN")
    leverage = trade.get("leverage", 1.0)

    if direction == "LONG":
        signal = "рост цены"
    else:
        signal = "падение цены"

    regime_desc = {
        "BULL": "Восходящий тренд подтверждён",
        "BEAR": "Нисходящий тренд",
        "HIGH_VOL": "Высокая волатильность - плечо снижено",
        "SIDEWAYS": "Боковик - консервативный вход",
        "UNKNOWN": "Режим не определён",
    }

    reason_parts = [
        f"ML-модель предсказала {signal} на горизонте {horizon}",
        f"Режим рынка: {regime} ({regime_desc.get(regime, '')})",
        f"Динамическое плечо: {leverage:.1f}x",
    ]

    return " | ".join(reason_parts)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    """Get system status."""
    config = get_config()
    conn = get_conn()
    engine = get_engine()

    # Database stats
    cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
    candles_count = cur.fetchone()["cnt"]

    cur = conn.execute("SELECT COUNT(*) as cnt FROM alerts")
    alerts_count = cur.fetchone()["cnt"]

    cur = conn.execute("SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM candles")
    row = cur.fetchone()
    date_range = None
    if row["min_ts"] and row["max_ts"]:
        date_range = {"min": row["min_ts"], "max": row["max_ts"]}

    return StatusResponse(
        candles_count=candles_count,
        alerts_count=alerts_count,
        tickers_count=len(config.tickers),
        date_range=date_range,
        models_loaded=engine.models.horizons,
        telegram_enabled=config.telegram.enabled,
        qwen_enabled=config.qwen.enabled,
    )


@app.get("/api/alerts", response_model=List[AlertResponse])
def get_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    sent_only: bool = Query(default=False),
):
    """Get recent alerts."""
    from .storage import get_alerts

    conn = get_conn()
    rows = get_alerts(conn, limit=limit, sent_only=sent_only)

    return [
        AlertResponse(
            id=row["id"],
            created_ts=row["created_ts"],
            secid=row["secid"],
            horizon=row["horizon"],
            p=row["p"],
            signal_type=row["signal_type"],
            entry=row["entry"],
            take=row["take"],
            stop=row["stop"],
            ttl_minutes=row["ttl_minutes"],
            anomaly_score=row["anomaly_score"],
            sent=bool(row["sent"]),
        )
        for row in rows
    ]


@app.get("/api/signals", response_model=CycleResponse)
def run_cycle():
    """Run a single pipeline cycle and return signals."""
    conn = get_conn()
    engine = get_engine()

    try:
        result = engine.run_cycle(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CycleResponse(
        signals=[
            SignalResponse(
                ticker=s.secid,
                direction=s.direction.value if hasattr(s.direction, 'value') else s.direction,
                horizon=s.horizon,
                p=s.probability,
                signal_type=s.signal_type,
                entry=s.entry,
                take=s.take,
                stop=s.stop,
                anomaly={
                    "score": s.anomaly_score,
                    "z_ret_5m": s.z_ret_5m,
                    "z_vol_5m": s.z_vol_5m,
                    "volume_spike": s.volume_spike,
                },
            )
            for s in result.signals
        ],
        anomalies_count=result.anomalies_count,
        duration_ms=result.duration_ms,
        errors=result.errors,
    )


@app.get("/api/trades", response_model=List[TradeResponse])
def get_trades(limit: int = Query(default=50, ge=1, le=500)):
    """Get completed trades from paper trading state."""
    state_file = Path("data/margin_paper_state.json")
    if not state_file.exists():
        return []

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception:
        return []

    trades = state.get("closed_trades", [])[-limit:]

    return [
        TradeResponse(
            ticker=t["ticker"],
            direction=t["direction"],
            entry_time=t["entry_time"],
            exit_time=t["exit_time"],
            entry_price=t["entry_price"],
            exit_price=t["exit_price"],
            size=t["size"],
            leverage=t["leverage"],
            pnl=t["pnl"],
            pnl_pct=t["pnl_pct"],
            exit_reason=t["exit_reason"],
            horizon=t["horizon"],
            regime=t.get("regime", "UNKNOWN"),
            reason=_get_trade_reason(t),
        )
        for t in reversed(trades)
    ]


@app.get("/api/equity")
def get_equity():
    """Get equity and P&L summary."""
    state_file = Path("data/margin_paper_state.json")
    if not state_file.exists():
        return {
            "equity": 200000,
            "initial_capital": 200000,
            "total_pnl": 0,
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "trades_count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "consecutive_losses": 0,
            "kill_switch_active": False,
        }

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception:
        return {"error": "Failed to read state"}

    trades = state.get("closed_trades", [])

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    return {
        "equity": state.get("cash", 200000) + state.get("margin_used", 0),
        "initial_capital": state.get("initial_capital", 200000),
        "total_pnl": total_pnl,
        "daily_pnl": state.get("daily_pnl", 0),
        "weekly_pnl": state.get("weekly_pnl", 0),
        "trades_count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "consecutive_losses": state.get("consecutive_losses", 0),
        "kill_switch_active": state.get("kill_switch_active", False),
        "kill_switch_reason": state.get("kill_switch_reason", ""),
        "day_mode": state.get("day_mode", "NORMAL"),
        "daily_target_reached": state.get("daily_target_reached", False),
    }


@app.get("/api/positions")
def get_positions():
    """Get open positions."""
    state_file = Path("data/margin_paper_state.json")
    if not state_file.exists():
        return {"positions": []}

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception:
        return {"positions": []}

    positions = state.get("positions", {})

    return {
        "positions": [
            {
                "ticker": ticker,
                "direction": pos["direction"],
                "entry_time": pos["entry_time"],
                "entry_price": pos["entry_price"],
                "size": pos["size"],
                "leverage": pos["leverage"],
                "take": pos.get("take"),
                "stop": pos.get("stop"),
                "horizon": pos["horizon"],
                "regime": pos.get("regime", "UNKNOWN"),
            }
            for ticker, pos in positions.items()
        ]
    }


@app.get("/api/tickers")
def get_tickers():
    """Get list of monitored tickers."""
    config = get_config()
    return {"tickers": config.tickers}


@app.get("/api/candles/{secid}")
def get_candles(
    secid: str,
    days: int = Query(default=1, ge=1, le=30),
):
    """Get recent candles for a ticker."""
    from .storage import get_window

    conn = get_conn()
    df = get_window(conn, minutes=days * 24 * 60)

    ticker_df = df[df["secid"] == secid.upper()]
    if ticker_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker {secid}")

    return {
        "secid": secid.upper(),
        "count": len(ticker_df),
        "candles": ticker_df.to_dict(orient="records"),
    }


# HTML Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOEX Agent Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; margin-bottom: 10px; }
        h2 { color: #00d9ff; border-bottom: 1px solid #333; padding-bottom: 5px; margin-top: 30px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
        .card {
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .card-title { color: #888; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }
        .card-value { font-size: 24px; font-weight: bold; color: #00d9ff; }
        .card-subtitle { font-size: 12px; color: #666; margin-top: 5px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            font-size: 14px;
        }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #333; }
        th { background: #0f3460; color: #00d9ff; font-weight: 500; font-size: 12px; }
        tr:hover { background: #1f4068; }
        .badge {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            display: inline-block;
        }
        .badge-long { background: #0a6b50; color: #4ade80; }
        .badge-short { background: #6b0a0a; color: #f87171; }
        .badge-sent { background: #0a3d6b; color: #60a5fa; }
        .badge-win { background: #0a6b50; color: #4ade80; }
        .badge-loss { background: #6b0a0a; color: #f87171; }
        .badge-take { background: #0a4d6b; color: #60a5fa; }
        .badge-stop { background: #6b4d0a; color: #fbbf24; }
        .badge-timeout { background: #4a4a4a; color: #a0a0a0; }
        .pnl-positive { color: #4ade80; }
        .pnl-negative { color: #f87171; }
        .refresh-btn {
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 10px 5px 10px 0;
        }
        .refresh-btn:hover { background: #00b8d9; }
        .timestamp { color: #666; font-size: 12px; }
        #loading { display: none; color: #00d9ff; margin: 10px 0; }
        .error { color: #f87171; background: #3d0a0a; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .reason-cell {
            max-width: 300px;
            font-size: 11px;
            color: #888;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .reason-cell:hover { color: #00d9ff; }
        .kill-switch-warning {
            background: #6b0a0a;
            color: #f87171;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: bold;
        }
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal-content {
            background: #16213e;
            padding: 30px;
            border-radius: 10px;
            max-width: 600px;
            width: 90%;
            border: 1px solid #333;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-content h3 { color: #00d9ff; margin-top: 0; }
        .modal-content table { font-size: 13px; }
        .close-btn {
            float: right;
            font-size: 24px;
            cursor: pointer;
            color: #888;
        }
        .close-btn:hover { color: #fff; }
        .reason-box {
            background: #0f3460;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            line-height: 1.6;
        }
        /* Equity Chart */
        .chart-container {
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #333;
            margin: 20px 0;
        }
        .chart-container canvas {
            max-height: 300px;
        }
        /* Tier badges */
        .badge-tier-a-plus { background: #0a6b50; color: #4ade80; }
        .badge-tier-a { background: #0a4d6b; color: #60a5fa; }
        .badge-tier-b { background: #6b4d0a; color: #fbbf24; }
        .badge-tier-c { background: #4a4a4a; color: #a0a0a0; }
        /* Day mode */
        .day-mode-normal { color: #60a5fa; }
        .day-mode-continuation { color: #4ade80; }
        .day-mode-halt { color: #f87171; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>MOEX Agent Dashboard</h1>
        <p class="timestamp">Last updated: <span id="lastUpdate">-</span></p>

        <button class="refresh-btn" onclick="refresh()">Refresh</button>
        <button class="refresh-btn" onclick="runCycle()">Run Cycle</button>
        <div id="loading">Loading...</div>
        <div id="error"></div>
        <div id="killSwitch"></div>

        <!-- System Status -->
        <div class="cards" id="cards">
            <div class="card">
                <div class="card-title">Candles</div>
                <div class="card-value" id="candles">-</div>
            </div>
            <div class="card">
                <div class="card-title">Alerts</div>
                <div class="card-value" id="alertsCount">-</div>
            </div>
            <div class="card">
                <div class="card-title">Tickers</div>
                <div class="card-value" id="tickers">-</div>
            </div>
            <div class="card">
                <div class="card-title">Models</div>
                <div class="card-value" id="models">-</div>
            </div>
        </div>

        <!-- Equity & P&L -->
        <h2>Equity & P&L</h2>
        <div class="cards">
            <div class="card">
                <div class="card-title">Equity</div>
                <div class="card-value" id="equity">-</div>
                <div class="card-subtitle" id="equityChange">-</div>
            </div>
            <div class="card">
                <div class="card-title">Total P&L</div>
                <div class="card-value" id="totalPnl">-</div>
            </div>
            <div class="card">
                <div class="card-title">Daily P&L</div>
                <div class="card-value" id="dailyPnl">-</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value" id="winRate">-</div>
                <div class="card-subtitle" id="tradesCount">-</div>
            </div>
            <div class="card">
                <div class="card-title">Profit Factor</div>
                <div class="card-value" id="profitFactor">-</div>
            </div>
            <div class="card">
                <div class="card-title">Loss Streak</div>
                <div class="card-value" id="lossStreak">-</div>
            </div>
            <div class="card">
                <div class="card-title">Day Mode</div>
                <div class="card-value" id="dayMode">NORMAL</div>
            </div>
            <div class="card">
                <div class="card-title">Daily Target</div>
                <div class="card-value">5%</div>
                <div class="card-subtitle" id="targetProgress">-</div>
            </div>
        </div>

        <!-- Equity Curve Chart -->
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>

        <!-- Open Positions -->
        <h2>Open Positions</h2>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Direction</th>
                    <th>Entry</th>
                    <th>Size</th>
                    <th>Leverage</th>
                    <th>Take</th>
                    <th>Stop</th>
                    <th>Horizon</th>
                    <th>Regime</th>
                </tr>
            </thead>
            <tbody id="positionsBody"></tbody>
        </table>

        <!-- Trades History -->
        <h2>Trade History</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Ticker</th>
                    <th>Dir</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&L</th>
                    <th>Result</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody id="tradesBody"></tbody>
        </table>

        <!-- Recent Alerts -->
        <h2>Recent Signals</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Ticker</th>
                    <th>Direction</th>
                    <th>Horizon</th>
                    <th>Prob</th>
                    <th>Entry</th>
                    <th>Take</th>
                    <th>Stop</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="alertsBody"></tbody>
        </table>
    </div>

    <!-- Trade Details Modal -->
    <div id="tradeModal" class="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="closeModal()">&times;</span>
            <h3>Trade Details</h3>
            <div id="tradeDetails"></div>
        </div>
    </div>

    <script>
        async function refresh() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').innerHTML = '';

            try {
                // Fetch status
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();

                document.getElementById('candles').textContent = status.candles_count.toLocaleString();
                document.getElementById('alertsCount').textContent = status.alerts_count.toLocaleString();
                document.getElementById('tickers').textContent = status.tickers_count;
                document.getElementById('models').textContent = status.models_loaded.length;

                // Fetch equity
                const equityRes = await fetch('/api/equity');
                const equity = await equityRes.json();

                document.getElementById('equity').textContent = equity.equity.toLocaleString() + ' RUB';

                const change = equity.equity - equity.initial_capital;
                const changeEl = document.getElementById('equityChange');
                changeEl.textContent = (change >= 0 ? '+' : '') + change.toLocaleString() + ' RUB';
                changeEl.className = 'card-subtitle ' + (change >= 0 ? 'pnl-positive' : 'pnl-negative');

                const totalPnlEl = document.getElementById('totalPnl');
                totalPnlEl.textContent = (equity.total_pnl >= 0 ? '+' : '') + equity.total_pnl.toLocaleString();
                totalPnlEl.className = 'card-value ' + (equity.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative');

                const dailyPnlEl = document.getElementById('dailyPnl');
                dailyPnlEl.textContent = (equity.daily_pnl >= 0 ? '+' : '') + equity.daily_pnl.toLocaleString();
                dailyPnlEl.className = 'card-value ' + (equity.daily_pnl >= 0 ? 'pnl-positive' : 'pnl-negative');

                document.getElementById('winRate').textContent = equity.win_rate.toFixed(1) + '%';
                document.getElementById('tradesCount').textContent = equity.wins + 'W / ' + equity.losses + 'L';
                document.getElementById('profitFactor').textContent = equity.profit_factor.toFixed(2);
                document.getElementById('lossStreak').textContent = equity.consecutive_losses;

                // Day mode and target progress
                const dayModeEl = document.getElementById('dayMode');
                const dayMode = equity.day_mode || 'NORMAL';
                dayModeEl.textContent = dayMode;
                dayModeEl.className = 'card-value day-mode-' + dayMode.toLowerCase();

                const dailyPnlPct = (equity.daily_pnl / equity.initial_capital * 100) || 0;
                const targetProgressEl = document.getElementById('targetProgress');
                targetProgressEl.textContent = dailyPnlPct.toFixed(2) + '% / 5%';
                targetProgressEl.className = 'card-subtitle ' + (dailyPnlPct >= 5 ? 'pnl-positive' : '');

                // Update equity chart
                await updateEquityChart(equity);

                // Kill switch warning
                const killSwitchEl = document.getElementById('killSwitch');
                if (equity.kill_switch_active) {
                    killSwitchEl.innerHTML = '<div class="kill-switch-warning">KILL SWITCH ACTIVE: ' + equity.kill_switch_reason + '</div>';
                } else {
                    killSwitchEl.innerHTML = '';
                }

                // Fetch positions
                const posRes = await fetch('/api/positions');
                const posData = await posRes.json();
                const posBody = document.getElementById('positionsBody');
                posBody.innerHTML = '';

                if (posData.positions.length === 0) {
                    posBody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#666">No open positions</td></tr>';
                } else {
                    for (const p of posData.positions) {
                        const row = document.createElement('tr');
                        const dirClass = p.direction === 'LONG' ? 'badge-long' : 'badge-short';
                        row.innerHTML = `
                            <td><strong>${p.ticker}</strong></td>
                            <td><span class="badge ${dirClass}">${p.direction}</span></td>
                            <td>${p.entry_price.toFixed(2)}</td>
                            <td>${p.size}</td>
                            <td>${p.leverage.toFixed(1)}x</td>
                            <td>${p.take ? p.take.toFixed(2) : '-'}</td>
                            <td>${p.stop ? p.stop.toFixed(2) : '-'}</td>
                            <td>${p.horizon}</td>
                            <td>${p.regime}</td>
                        `;
                        posBody.appendChild(row);
                    }
                }

                // Fetch trades
                const tradesRes = await fetch('/api/trades?limit=20');
                const trades = await tradesRes.json();
                const tradesBody = document.getElementById('tradesBody');
                tradesBody.innerHTML = '';

                if (trades.length === 0) {
                    tradesBody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#666">No trades yet</td></tr>';
                } else {
                    for (const t of trades) {
                        const row = document.createElement('tr');
                        const pnlClass = t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                        const resultBadge = t.pnl >= 0 ? 'badge-win' : 'badge-loss';
                        const resultText = t.pnl >= 0 ? 'WIN' : 'LOSS';
                        const dirClass = t.direction === 'LONG' ? 'badge-long' : 'badge-short';

                        let exitBadge = 'badge-timeout';
                        if (t.exit_reason === 'take') exitBadge = 'badge-take';
                        if (t.exit_reason === 'stop') exitBadge = 'badge-stop';

                        const shortReason = t.reason.length > 40 ? t.reason.substring(0, 40) + '...' : t.reason;
                        const tradeJson = JSON.stringify(t).replace(/"/g, '&quot;');

                        row.innerHTML = `
                            <td>${new Date(t.exit_time).toLocaleString()}</td>
                            <td><strong>${t.ticker}</strong></td>
                            <td><span class="badge ${dirClass}">${t.direction}</span></td>
                            <td>${t.entry_price.toFixed(2)}</td>
                            <td>${t.exit_price.toFixed(2)}</td>
                            <td class="${pnlClass}">
                                <strong>${t.pnl >= 0 ? '+' : ''}${t.pnl.toLocaleString()}</strong>
                                <br><small>(${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%)</small>
                            </td>
                            <td>
                                <span class="badge ${resultBadge}">${resultText}</span>
                                <span class="badge ${exitBadge}">${t.exit_reason}</span>
                            </td>
                            <td class="reason-cell" onclick='showTradeDetails(${tradeJson})'>${shortReason}</td>
                        `;
                        tradesBody.appendChild(row);
                    }
                }

                // Fetch alerts
                const alertsRes = await fetch('/api/alerts?limit=15');
                const alerts = await alertsRes.json();
                const alertsBody = document.getElementById('alertsBody');
                alertsBody.innerHTML = '';

                for (const a of alerts) {
                    const row = document.createElement('tr');
                    const direction = a.signal_type.includes('price') && a.entry && a.take && a.take > a.entry ? 'LONG' : 'SHORT';
                    const dirClass = direction === 'LONG' ? 'badge-long' : 'badge-short';

                    row.innerHTML = `
                        <td>${new Date(a.created_ts).toLocaleString()}</td>
                        <td><strong>${a.secid}</strong></td>
                        <td><span class="badge ${dirClass}">${direction}</span></td>
                        <td>${a.horizon}</td>
                        <td>${(a.p * 100).toFixed(0)}%</td>
                        <td>${a.entry ? a.entry.toFixed(2) : '-'}</td>
                        <td>${a.take ? a.take.toFixed(2) : '-'}</td>
                        <td>${a.stop ? a.stop.toFixed(2) : '-'}</td>
                        <td>${a.sent ? '<span class="badge badge-sent">SENT</span>' : '-'}</td>
                    `;
                    alertsBody.appendChild(row);
                }

                document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
            } catch (e) {
                document.getElementById('error').innerHTML = '<div class="error">Error: ' + e.message + '</div>';
            }

            document.getElementById('loading').style.display = 'none';
        }

        function showTradeDetails(trade) {
            const modal = document.getElementById('tradeModal');
            const details = document.getElementById('tradeDetails');
            const pnlClass = trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

            details.innerHTML = `
                <table style="width:100%">
                    <tr><td>Ticker:</td><td><strong>${trade.ticker}</strong></td></tr>
                    <tr><td>Direction:</td><td>${trade.direction}</td></tr>
                    <tr><td>Horizon:</td><td>${trade.horizon}</td></tr>
                    <tr><td>Leverage:</td><td>${trade.leverage}x</td></tr>
                    <tr><td>Market Regime:</td><td>${trade.regime}</td></tr>
                    <tr><td>Entry:</td><td>${trade.entry_price.toFixed(2)} @ ${new Date(trade.entry_time).toLocaleString()}</td></tr>
                    <tr><td>Exit:</td><td>${trade.exit_price.toFixed(2)} @ ${new Date(trade.exit_time).toLocaleString()}</td></tr>
                    <tr><td>Size:</td><td>${trade.size} shares</td></tr>
                    <tr><td>Exit Reason:</td><td>${trade.exit_reason}</td></tr>
                    <tr><td>P&L:</td><td class="${pnlClass}"><strong>${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toLocaleString()} RUB (${trade.pnl_pct.toFixed(2)}%)</strong></td></tr>
                </table>
                <div class="reason-box">
                    <strong>Trade Basis:</strong><br>
                    ${trade.reason}
                </div>
            `;

            modal.style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('tradeModal').style.display = 'none';
        }

        async function runCycle() {
            document.getElementById('loading').style.display = 'block';
            try {
                const res = await fetch('/api/signals');
                const data = await res.json();
                if (data.signals.length > 0) {
                    alert('Found ' + data.signals.length + ' signals!\\n\\n' + data.signals.map(s => s.ticker + ' ' + s.direction + ' ' + s.horizon + ' p=' + (s.p*100).toFixed(0) + '%').join('\\n'));
                } else {
                    alert('Cycle complete: ' + data.anomalies_count + ' anomalies, no signals (duration: ' + data.duration_ms.toFixed(0) + 'ms)');
                }
                refresh();
            } catch (e) {
                document.getElementById('error').innerHTML = '<div class="error">Error: ' + e.message + '</div>';
            }
            document.getElementById('loading').style.display = 'none';
        }

        window.onclick = function(event) {
            const modal = document.getElementById('tradeModal');
            if (event.target === modal) closeModal();
        }

        // Equity Chart
        let equityChart = null;
        const equityHistory = [];
        const maxHistoryPoints = 100;

        async function updateEquityChart(equity) {
            // Add current point to history
            const now = new Date();
            equityHistory.push({
                time: now.toLocaleTimeString(),
                equity: equity.equity,
                pnl: equity.total_pnl
            });

            // Keep only last N points
            if (equityHistory.length > maxHistoryPoints) {
                equityHistory.shift();
            }

            const ctx = document.getElementById('equityChart').getContext('2d');

            if (equityChart) {
                // Update existing chart
                equityChart.data.labels = equityHistory.map(p => p.time);
                equityChart.data.datasets[0].data = equityHistory.map(p => p.equity);
                equityChart.update('none');
            } else {
                // Create new chart
                equityChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: equityHistory.map(p => p.time),
                        datasets: [{
                            label: 'Equity (RUB)',
                            data: equityHistory.map(p => p.equity),
                            borderColor: '#00d9ff',
                            backgroundColor: 'rgba(0, 217, 255, 0.1)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 2,
                            pointHoverRadius: 5
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return context.parsed.y.toLocaleString() + ' RUB';
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                },
                                ticks: {
                                    color: '#888',
                                    callback: function(value) {
                                        return value.toLocaleString();
                                    }
                                }
                            },
                            x: {
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.05)'
                                },
                                ticks: {
                                    color: '#888',
                                    maxTicksLimit: 10
                                }
                            }
                        },
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        }
                    }
                });
            }

            // Also try to load historical data from trades
            try {
                const tradesRes = await fetch('/api/trades?limit=50');
                const trades = await tradesRes.json();

                if (trades.length > 0 && equityHistory.length < 5) {
                    // Reconstruct equity history from trades
                    let runningEquity = equity.initial_capital;
                    const historicalPoints = [];

                    // Sort trades by exit time
                    const sortedTrades = [...trades].reverse();

                    for (const t of sortedTrades) {
                        runningEquity += t.pnl;
                        historicalPoints.push({
                            time: new Date(t.exit_time).toLocaleTimeString(),
                            equity: runningEquity,
                            pnl: t.pnl
                        });
                    }

                    // Add to history if we have historical data
                    if (historicalPoints.length > equityHistory.length) {
                        equityHistory.length = 0;
                        equityHistory.push(...historicalPoints);

                        equityChart.data.labels = equityHistory.map(p => p.time);
                        equityChart.data.datasets[0].data = equityHistory.map(p => p.equity);
                        equityChart.update('none');
                    }
                }
            } catch (e) {
                console.log('Could not load historical trades for chart:', e);
            }
        }

        refresh();
        setInterval(refresh, 30000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """HTML dashboard."""
    return DASHBOARD_HTML

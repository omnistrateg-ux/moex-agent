"""
MOEX Agent Web Dashboard

Веб-интерфейс для мониторинга торговли.

Usage:
    python -m moex_agent.dashboard --port 8080
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template_string, jsonify

logger = logging.getLogger("moex_agent.dashboard")

app = Flask(__name__)

STATE_FILE = Path("data/paper_trading_state.json")
LOG_FILE = Path("data/paper_trading.log")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOEX Agent Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #7c4dff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .simulation-badge {
            background: #ff6b6b;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 {
            font-size: 1rem;
            color: #888;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .big-number {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .positive { color: #00e676; }
        .negative { color: #ff5252; }
        .neutral { color: #888; }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: #888; }
        .stat-value { font-weight: 500; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        th {
            color: #888;
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
        }
        .refresh-btn {
            background: linear-gradient(90deg, #00d4ff, #7c4dff);
            border: none;
            padding: 10px 25px;
            border-radius: 25px;
            color: white;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.2s;
        }
        .refresh-btn:hover { transform: scale(1.05); }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }
        .last-update { color: #888; font-size: 0.9rem; }
        .chart-placeholder {
            height: 200px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        .metric-group {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            text-align: center;
        }
        .metric-item .value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .metric-item .label {
            font-size: 0.8rem;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <span class="simulation-badge">🎮 СИМУЛЯЦИЯ</span>
                <h1>MOEX Agent Dashboard</h1>
            </div>
            <div>
                <span class="last-update" id="lastUpdate">Загрузка...</span>
                <button class="refresh-btn" onclick="loadData()">Обновить</button>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>💰 Баланс</h2>
                <div class="big-number" id="equity">--</div>
                <div class="stat-row">
                    <span class="stat-label">Начальный капитал</span>
                    <span class="stat-value" id="initialCapital">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Свободные деньги</span>
                    <span class="stat-value" id="cash">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">В позициях</span>
                    <span class="stat-value" id="positionsValue">--</span>
                </div>
            </div>

            <div class="card">
                <h2>📈 Прибыль/Убыток</h2>
                <div class="big-number" id="totalPnl">--</div>
                <div class="stat-row">
                    <span class="stat-label">Процент</span>
                    <span class="stat-value" id="totalPnlPct">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Win Rate</span>
                    <span class="stat-value" id="winRate">--</span>
                </div>
            </div>

            <div class="card">
                <h2>📊 Статистика</h2>
                <div class="metric-group">
                    <div class="metric-item">
                        <div class="value" id="totalTrades">--</div>
                        <div class="label">Сделок</div>
                    </div>
                    <div class="metric-item">
                        <div class="value positive" id="wins">--</div>
                        <div class="label">Выигрышей</div>
                    </div>
                    <div class="metric-item">
                        <div class="value negative" id="losses">--</div>
                        <div class="label">Убытков</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card" style="grid-column: span 2;">
                <h2>📍 Открытые позиции</h2>
                <table id="positionsTable">
                    <thead>
                        <tr>
                            <th>Тикер</th>
                            <th>Направление</th>
                            <th>Цена входа</th>
                            <th>Количество</th>
                            <th>Сумма</th>
                            <th>Горизонт</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="card">
                <h2>⏱ Информация</h2>
                <div class="stat-row">
                    <span class="stat-label">Время работы</span>
                    <span class="stat-value" id="uptime">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Открытых позиций</span>
                    <span class="stat-value" id="openPositions">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Статус</span>
                    <span class="stat-value positive" id="status">Активен</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📋 Последние сделки</h2>
            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Время</th>
                        <th>Тикер</th>
                        <th>Направление</th>
                        <th>P&L</th>
                        <th>%</th>
                        <th>Причина</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <script>
        function formatNumber(num, decimals = 0) {
            return new Intl.NumberFormat('ru-RU', {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            }).format(num);
        }

        function formatPnl(pnl) {
            const prefix = pnl >= 0 ? '+' : '';
            return prefix + formatNumber(pnl, 0) + ' ₽';
        }

        function loadData() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    // Update balance
                    document.getElementById('equity').textContent = formatNumber(data.equity, 0) + ' ₽';
                    document.getElementById('initialCapital').textContent = formatNumber(data.initial_capital, 0) + ' ₽';
                    document.getElementById('cash').textContent = formatNumber(data.cash, 0) + ' ₽';
                    document.getElementById('positionsValue').textContent = formatNumber(data.positions_value, 0) + ' ₽';

                    // Update PnL
                    const pnlElement = document.getElementById('totalPnl');
                    pnlElement.textContent = formatPnl(data.total_pnl);
                    pnlElement.className = 'big-number ' + (data.total_pnl >= 0 ? 'positive' : 'negative');

                    document.getElementById('totalPnlPct').textContent = (data.total_pnl_pct >= 0 ? '+' : '') + data.total_pnl_pct.toFixed(2) + '%';
                    document.getElementById('winRate').textContent = data.win_rate.toFixed(1) + '%';

                    // Update stats
                    document.getElementById('totalTrades').textContent = data.total_trades;
                    document.getElementById('wins').textContent = data.wins;
                    document.getElementById('losses').textContent = data.losses;
                    document.getElementById('openPositions').textContent = Object.keys(data.positions).length;
                    document.getElementById('uptime').textContent = data.uptime;

                    // Update positions table
                    const positionsBody = document.getElementById('positionsTable').querySelector('tbody');
                    positionsBody.innerHTML = '';
                    for (const [ticker, pos] of Object.entries(data.positions)) {
                        const row = document.createElement('tr');
                        const dirClass = pos.direction === 'LONG' ? 'positive' : 'negative';
                        row.innerHTML = `
                            <td><strong>${ticker}</strong></td>
                            <td class="${dirClass}">${pos.direction === 'LONG' ? '📈 Покупка' : '📉 Продажа'}</td>
                            <td>${pos.entry_price.toFixed(2)} ₽</td>
                            <td>${pos.size}</td>
                            <td>${formatNumber(pos.entry_price * pos.size, 0)} ₽</td>
                            <td>${pos.horizon}</td>
                        `;
                        positionsBody.appendChild(row);
                    }
                    if (Object.keys(data.positions).length === 0) {
                        positionsBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666">Нет открытых позиций</td></tr>';
                    }

                    // Update trades table
                    const tradesBody = document.getElementById('tradesTable').querySelector('tbody');
                    tradesBody.innerHTML = '';
                    const recentTrades = data.closed_trades.slice(-10).reverse();
                    for (const trade of recentTrades) {
                        const row = document.createElement('tr');
                        const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
                        const exitTime = new Date(trade.exit_time).toLocaleString('ru-RU');
                        row.innerHTML = `
                            <td>${exitTime}</td>
                            <td><strong>${trade.ticker}</strong></td>
                            <td>${trade.direction === 'LONG' ? '📈' : '📉'}</td>
                            <td class="${pnlClass}">${formatPnl(trade.pnl)}</td>
                            <td class="${pnlClass}">${trade.pnl_pct >= 0 ? '+' : ''}${trade.pnl_pct.toFixed(2)}%</td>
                            <td>${trade.exit_reason}</td>
                        `;
                        tradesBody.appendChild(row);
                    }
                    if (recentTrades.length === 0) {
                        tradesBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666">Нет завершённых сделок</td></tr>';
                    }

                    // Update timestamp
                    document.getElementById('lastUpdate').textContent = 'Обновлено: ' + new Date().toLocaleTimeString('ru-RU');
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                    document.getElementById('status').textContent = 'Ошибка';
                    document.getElementById('status').className = 'stat-value negative';
                });
        }

        // Load data on page load
        loadData();

        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Главная страница dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def api_state():
    """API endpoint для получения состояния."""
    try:
        if not STATE_FILE.exists():
            return jsonify({"error": "State file not found"}), 404

        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        # Calculate derived values
        positions_value = sum(
            pos["entry_price"] * pos["size"]
            for pos in state.get("positions", {}).values()
        )
        equity = state["cash"] + positions_value
        total_pnl = sum(t["pnl"] for t in state.get("closed_trades", []))
        total_pnl_pct = (total_pnl / state["initial_capital"]) * 100 if state["initial_capital"] > 0 else 0

        closed_trades = state.get("closed_trades", [])
        wins = len([t for t in closed_trades if t["pnl"] > 0])
        losses = len([t for t in closed_trades if t["pnl"] <= 0])
        win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0

        # Calculate uptime
        start_time = datetime.fromisoformat(state["start_time"])
        uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime = f"{hours}ч {minutes}м"

        return jsonify({
            "initial_capital": state["initial_capital"],
            "cash": state["cash"],
            "equity": equity,
            "positions_value": positions_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_trades": len(closed_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "uptime": uptime,
            "positions": state.get("positions", {}),
            "closed_trades": closed_trades,
        })

    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs")
def api_logs():
    """API endpoint для получения последних логов."""
    try:
        if not LOG_FILE.exists():
            return jsonify({"logs": []})

        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-100:]  # Last 100 lines

        return jsonify({"logs": [line.strip() for line in lines]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="MOEX Agent Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting MOEX Agent Dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

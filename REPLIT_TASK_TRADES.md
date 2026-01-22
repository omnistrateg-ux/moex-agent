# Задача для Replit AI: Добавить панель сделок в Dashboard

## 🎯 Цель

Добавить в веб-dashboard (`webapp.py`) новую панель с историей сделок, которая показывает:
1. **Список совершённых сделок** (closed trades)
2. **Результат каждой сделки** (прибыль/убыток в рублях и %)
3. **Итоговый P&L** за день/неделю/всё время
4. **Основание для сделки** — почему модель решила купить/продать

---

## 📋 Что нужно сделать

### 1. Добавить API endpoint `/api/trades`

```python
# В webapp.py добавить:

class TradeResponse(BaseModel):
    """Completed trade info."""
    ticker: str
    direction: str           # LONG или SHORT
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: int                # количество акций
    leverage: float          # плечо
    pnl: float              # прибыль/убыток в рублях
    pnl_pct: float          # прибыль/убыток в %
    exit_reason: str         # take, stop, timeout
    horizon: str             # 5m, 10m, 30m, 1h
    regime: str              # BULL, BEAR, SIDEWAYS, HIGH_VOL
    # Основание для сделки:
    signal_basis: dict       # детали почему вошли

@app.get("/api/trades", response_model=List[TradeResponse])
def get_trades(limit: int = Query(default=50)):
    """Get completed trades from paper trading state."""
    import json
    from pathlib import Path

    state_file = Path("data/margin_paper_state.json")
    if not state_file.exists():
        return []

    with open(state_file) as f:
        state = json.load(f)

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
            signal_basis={
                "horizon": t["horizon"],
                "regime": t.get("regime", "UNKNOWN"),
                "leverage": t["leverage"],
                "reason": _get_trade_reason(t)
            }
        )
        for t in reversed(trades)  # newest first
    ]

def _get_trade_reason(trade: dict) -> str:
    """Generate human-readable reason for trade."""
    direction = trade["direction"]
    horizon = trade["horizon"]
    regime = trade.get("regime", "UNKNOWN")
    leverage = trade["leverage"]

    # Формируем описание
    if direction == "LONG":
        action = "ПОКУПКА"
        signal = "рост"
    else:
        action = "ПРОДАЖА"
        signal = "падение"

    reasons = []
    reasons.append(f"ML-модель предсказала {signal} на горизонте {horizon}")
    reasons.append(f"Режим рынка: {regime}")
    reasons.append(f"Плечо: {leverage:.1f}x (динамическое)")

    if regime == "BULL":
        reasons.append("Восходящий тренд подтверждён")
    elif regime == "BEAR":
        reasons.append("Нисходящий тренд")
    elif regime == "HIGH_VOL":
        reasons.append("Высокая волатильность — плечо снижено")
    elif regime == "SIDEWAYS":
        reasons.append("Боковик — консервативный вход")

    return " | ".join(reasons)
```

### 2. Добавить endpoint `/api/equity`

```python
@app.get("/api/equity")
def get_equity():
    """Get equity and P&L summary."""
    import json
    from pathlib import Path

    state_file = Path("data/margin_paper_state.json")
    if not state_file.exists():
        return {
            "equity": 200000,
            "initial_capital": 200000,
            "total_pnl": 0,
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "trades_count": 0,
            "win_rate": 0,
            "profit_factor": 0
        }

    with open(state_file) as f:
        state = json.load(f)

    trades = state.get("closed_trades", [])

    # Calculate metrics
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
        "profit_factor": profit_factor
    }
```

### 3. Обновить HTML Dashboard

Добавить в `DASHBOARD_HTML` после таблицы алертов:

```html
<!-- Equity Summary -->
<h2>💰 Equity & P&L</h2>
<div class="cards" id="equityCards">
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
</div>

<!-- Trades History -->
<h2>📊 История сделок</h2>
<table id="tradesTable">
    <thead>
        <tr>
            <th>Время</th>
            <th>Тикер</th>
            <th>Направление</th>
            <th>Вход</th>
            <th>Выход</th>
            <th>P&L</th>
            <th>Результат</th>
            <th>Основание</th>
        </tr>
    </thead>
    <tbody id="tradesBody"></tbody>
</table>

<!-- Trade Details Modal -->
<div id="tradeModal" class="modal" style="display:none;">
    <div class="modal-content">
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <h3>📋 Детали сделки</h3>
        <div id="tradeDetails"></div>
    </div>
</div>
```

### 4. Добавить CSS стили

```css
/* P&L colors */
.pnl-positive { color: #4ade80; }
.pnl-negative { color: #f87171; }

/* Trade result badges */
.badge-win { background: #0a6b50; color: #4ade80; }
.badge-loss { background: #6b0a0a; color: #f87171; }
.badge-take { background: #0a4d6b; color: #60a5fa; }
.badge-stop { background: #6b4d0a; color: #fbbf24; }
.badge-timeout { background: #4a4a4a; color: #a0a0a0; }

/* Modal */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.8);
    display: flex;
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
}
.close-btn {
    float: right;
    font-size: 24px;
    cursor: pointer;
    color: #888;
}
.close-btn:hover { color: #fff; }

/* Reason text */
.reason-text {
    font-size: 12px;
    color: #888;
    max-width: 300px;
    cursor: pointer;
}
.reason-text:hover { color: #00d9ff; }
```

### 5. Добавить JavaScript

```javascript
async function loadTrades() {
    try {
        // Load equity
        const equityRes = await fetch('/api/equity');
        const equity = await equityRes.json();

        document.getElementById('equity').textContent =
            equity.equity.toLocaleString() + ' ₽';

        const change = equity.equity - equity.initial_capital;
        const changeEl = document.getElementById('equityChange');
        changeEl.textContent = (change >= 0 ? '+' : '') + change.toLocaleString() + ' ₽';
        changeEl.className = change >= 0 ? 'pnl-positive' : 'pnl-negative';

        const totalPnlEl = document.getElementById('totalPnl');
        totalPnlEl.textContent = (equity.total_pnl >= 0 ? '+' : '') +
            equity.total_pnl.toLocaleString() + ' ₽';
        totalPnlEl.className = equity.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

        const dailyPnlEl = document.getElementById('dailyPnl');
        dailyPnlEl.textContent = (equity.daily_pnl >= 0 ? '+' : '') +
            equity.daily_pnl.toLocaleString() + ' ₽';
        dailyPnlEl.className = equity.daily_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

        document.getElementById('winRate').textContent =
            equity.win_rate.toFixed(1) + '%';
        document.getElementById('tradesCount').textContent =
            `${equity.wins}W / ${equity.losses}L из ${equity.trades_count}`;

        // Load trades
        const tradesRes = await fetch('/api/trades?limit=20');
        const trades = await tradesRes.json();

        const tbody = document.getElementById('tradesBody');
        tbody.innerHTML = '';

        for (const t of trades) {
            const row = document.createElement('tr');
            const pnlClass = t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const resultBadge = t.pnl >= 0 ? 'badge-win' : 'badge-loss';
            const resultText = t.pnl >= 0 ? '✅ WIN' : '❌ LOSS';

            let exitBadge = 'badge-timeout';
            if (t.exit_reason === 'take') exitBadge = 'badge-take';
            if (t.exit_reason === 'stop') exitBadge = 'badge-stop';

            const reason = t.signal_basis.reason || 'ML Signal';
            const shortReason = reason.length > 50 ? reason.substring(0, 50) + '...' : reason;

            row.innerHTML = `
                <td>${new Date(t.exit_time).toLocaleString()}</td>
                <td><strong>${t.ticker}</strong></td>
                <td>
                    <span class="badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}">
                        ${t.direction}
                    </span>
                </td>
                <td>${t.entry_price.toFixed(2)}</td>
                <td>${t.exit_price.toFixed(2)}</td>
                <td class="${pnlClass}">
                    <strong>${t.pnl >= 0 ? '+' : ''}${t.pnl.toLocaleString()} ₽</strong>
                    <br><small>(${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%)</small>
                </td>
                <td>
                    <span class="badge ${resultBadge}">${resultText}</span>
                    <br><span class="badge ${exitBadge}">${t.exit_reason}</span>
                </td>
                <td>
                    <span class="reason-text" onclick="showTradeDetails(${JSON.stringify(t).replace(/"/g, '&quot;')})">
                        ${shortReason}
                    </span>
                </td>
            `;
            tbody.appendChild(row);
        }
    } catch (e) {
        console.error('Error loading trades:', e);
    }
}

function showTradeDetails(trade) {
    const modal = document.getElementById('tradeModal');
    const details = document.getElementById('tradeDetails');

    const pnlClass = trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

    details.innerHTML = `
        <table style="width:100%">
            <tr><td>Тикер:</td><td><strong>${trade.ticker}</strong></td></tr>
            <tr><td>Направление:</td><td>${trade.direction}</td></tr>
            <tr><td>Горизонт:</td><td>${trade.horizon}</td></tr>
            <tr><td>Плечо:</td><td>${trade.leverage}x</td></tr>
            <tr><td>Режим рынка:</td><td>${trade.regime}</td></tr>
            <tr><td>Вход:</td><td>${trade.entry_price.toFixed(2)} ₽ @ ${new Date(trade.entry_time).toLocaleString()}</td></tr>
            <tr><td>Выход:</td><td>${trade.exit_price.toFixed(2)} ₽ @ ${new Date(trade.exit_time).toLocaleString()}</td></tr>
            <tr><td>Размер:</td><td>${trade.size} шт.</td></tr>
            <tr><td>Причина выхода:</td><td>${trade.exit_reason}</td></tr>
            <tr><td>P&L:</td><td class="${pnlClass}"><strong>${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toLocaleString()} ₽ (${trade.pnl_pct.toFixed(2)}%)</strong></td></tr>
        </table>

        <h4 style="margin-top:20px;">📋 Основание для сделки:</h4>
        <p style="background:#0f3460; padding:15px; border-radius:5px; line-height:1.6;">
            ${trade.signal_basis.reason}
        </p>
    `;

    modal.style.display = 'flex';
}

function closeModal() {
    document.getElementById('tradeModal').style.display = 'none';
}

// Add to refresh function
async function refresh() {
    // ... existing code ...
    await loadTrades();
}

// Close modal on click outside
window.onclick = function(event) {
    const modal = document.getElementById('tradeModal');
    if (event.target === modal) {
        closeModal();
    }
}
```

---

## 📊 Формат "Основание для сделки"

Каждая сделка должна показывать:

```
🔹 ML-модель предсказала рост на горизонте 30m
🔹 Режим рынка: BULL (восходящий тренд)
🔹 Плечо: 2.1x (динамическое на основе confidence)
🔹 Сигналы: RSI=32 (перепроданность), MACD crossover
🔹 Аномалия: объём +150% от среднего
```

### Данные для основания берём из:

1. **margin_paper_state.json** — closed_trades
2. **alerts таблица** — signal_type, anomaly_score
3. **margin_risk_engine.py** — regime, leverage calculation

---

## 🎨 Итоговый вид Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  MOEX Agent Dashboard                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Candles: 50K] [Alerts: 127] [Tickers: 20] [Models: 4]    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  💰 Equity & P&L                                            │
│                                                             │
│  [Equity]      [Total P&L]    [Daily P&L]   [Win Rate]     │
│  201,450 ₽    +1,450 ₽       +320 ₽        66.7%          │
│  +1,450 ₽                                   4W/2L          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  📊 История сделок                                          │
│                                                             │
│  Время    │ Тикер │ Dir  │ Вход   │ Выход │ P&L     │ Осн. │
│  ─────────┼───────┼──────┼────────┼───────┼─────────┼──────│
│  14:28    │ SMLT  │SHORT │ 992.50 │991.56 │ +3 ₽    │ ML...│
│  14:15    │ MGNT  │SHORT │ 5420   │ 5415  │ +5 ₽    │ ML...│
│  13:45    │ SFIN  │SHORT │ 1046   │ 1048  │ -3 ₽    │ ML...│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist

- [ ] Добавить `/api/trades` endpoint
- [ ] Добавить `/api/equity` endpoint
- [ ] Обновить HTML с панелью Equity
- [ ] Добавить таблицу сделок
- [ ] Добавить модальное окно с деталями
- [ ] Добавить CSS стили для P&L
- [ ] Добавить JavaScript для загрузки данных
- [ ] Добавить функцию генерации "основания"

---

## 🔧 Тестирование

1. Запустить `python main.py`
2. Открыть Dashboard на порту 8080
3. Дождаться нескольких сделок в paper trading
4. Проверить:
   - Equity обновляется
   - Сделки появляются в таблице
   - Клик по "Основание" открывает детали
   - P&L показывает правильные цвета

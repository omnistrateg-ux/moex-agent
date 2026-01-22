#!/bin/bash
# Ежедневный отчёт MOEX Agent в Telegram
# Запускать через cron: 0 21 * * * /path/to/daily_report.sh
# (21:00 - после закрытия биржи)

PROJECT_DIR="/Users/artempobedinskij/Desktop/Projects/moex_agent"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"

cd "$PROJECT_DIR"
source "$VENV_PATH"

python -c "
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from moex_agent.config_schema import load_config
from moex_agent.telegram import send_telegram

config = load_config('config.yaml')
state_file = Path('data/paper_trading_state.json')

if not state_file.exists():
    print('State file not found')
    exit(1)

state = json.loads(state_file.read_text())

# Расчёты
initial = state['initial_capital']
cash = state['cash']
positions = state.get('positions', {})
positions_value = sum(p['entry_price'] * p['size'] for p in positions.values())
equity = cash + positions_value
total_pnl = sum(t['pnl'] for t in state.get('closed_trades', []))
total_pnl_pct = (total_pnl / initial) * 100

# Сделки за сегодня
today = datetime.now(timezone.utc).date().isoformat()
today_trades = [
    t for t in state.get('closed_trades', [])
    if t['exit_time'].startswith(today)
]
today_pnl = sum(t['pnl'] for t in today_trades)
today_wins = len([t for t in today_trades if t['pnl'] > 0])

# Общая статистика
all_trades = state.get('closed_trades', [])
total_wins = len([t for t in all_trades if t['pnl'] > 0])
win_rate = (total_wins / len(all_trades) * 100) if all_trades else 0

# Время работы
start_time = datetime.fromisoformat(state['start_time'])
uptime = datetime.now(timezone.utc) - start_time
days = uptime.days
hours = uptime.seconds // 3600

# Формируем отчёт
pnl_emoji = '📈 ПРИБЫЛЬ' if total_pnl >= 0 else '📉 УБЫТОК'
today_emoji = '✅' if today_pnl >= 0 else '❌'

message = f'''🎮 СИМУЛЯЦИЯ - ВИРТУАЛЬНЫЙ СЧЁТ

📅 ЕЖЕДНЕВНЫЙ ОТЧЁТ
━━━━━━━━━━━━━━━━━━━━
Дата: {datetime.now().strftime('%d.%m.%Y')}
Время работы: {days} д. {hours} ч.

💰 СОСТОЯНИЕ СЧЁТА
━━━━━━━━━━━━━━━━━━━━
Начальный капитал: {initial:,.0f} ₽
Текущий баланс: {equity:,.0f} ₽
{pnl_emoji}: {abs(total_pnl):,.0f} ₽ ({total_pnl_pct:+.2f}%)

📊 ЗА СЕГОДНЯ
━━━━━━━━━━━━━━━━━━━━
Сделок: {len(today_trades)}
Выигрышных: {today_wins}
{today_emoji} Результат: {today_pnl:+,.0f} ₽

📈 ОБЩАЯ СТАТИСТИКА
━━━━━━━━━━━━━━━━━━━━
Всего сделок: {len(all_trades)}
Win Rate: {win_rate:.1f}%
Открытых позиций: {len(positions)}
'''

result = send_telegram(
    config.telegram.bot_token or '',
    config.telegram.chat_id or '',
    message
)
print('Отчёт отправлен:', result)
"

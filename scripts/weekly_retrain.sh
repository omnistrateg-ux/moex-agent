#!/bin/bash
# Еженедельное переобучение моделей MOEX Agent
# Запускать через cron: 0 3 * * 0 /path/to/weekly_retrain.sh

PROJECT_DIR="/Users/artempobedinskij/Desktop/Projects/moex_agent"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"
LOG_FILE="$PROJECT_DIR/data/retrain_$(date +%Y%m%d).log"

echo "=== Weekly Retrain Started: $(date) ===" >> "$LOG_FILE"

cd "$PROJECT_DIR"
source "$VENV_PATH"

# 1. Обновить данные (последние 7 дней)
echo "Updating data..." >> "$LOG_FILE"
python -m moex_agent bootstrap --days 7 >> "$LOG_FILE" 2>&1

# 2. Переобучить модели
echo "Retraining models..." >> "$LOG_FILE"
python -m moex_agent.optimize_train --skip-optimize >> "$LOG_FILE" 2>&1

# 3. Перезапустить paper trading
echo "Restarting paper trading..." >> "$LOG_FILE"
pkill -f "paper_trading" 2>/dev/null
sleep 5
nohup python -m moex_agent.paper_trading --capital 200000 --duration-days 7 >> data/paper_trading.log 2>&1 &

# 4. Уведомление в Telegram
python -c "
from moex_agent.config_schema import load_config
from moex_agent.telegram import send_telegram
import json
from pathlib import Path

config = load_config('config.yaml')
meta = json.loads(Path('models/meta.json').read_text())

lines = ['🔄 ЕЖЕНЕДЕЛЬНОЕ ПЕРЕОБУЧЕНИЕ', '', '✅ Модели обновлены:', '']
for h, info in meta.items():
    m = info.get('metrics', {})
    wr = m.get('win_rate', 0)
    pf = m.get('profit_factor', 0)
    lines.append(f'  {h}: WR={wr:.1f}%, PF={pf:.2f}')

send_telegram(config.telegram.bot_token, config.telegram.chat_id, '\n'.join(lines))
"

echo "=== Weekly Retrain Completed: $(date) ===" >> "$LOG_FILE"

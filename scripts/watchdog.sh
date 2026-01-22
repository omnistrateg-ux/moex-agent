#!/bin/bash
# MOEX Agent Watchdog
# Автоматический перезапуск при падении

PROJECT_DIR="/Users/artempobedinskij/Desktop/Projects/moex_agent"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"
LOG_FILE="$PROJECT_DIR/data/watchdog.log"
PID_FILE="$PROJECT_DIR/data/paper_trading.pid"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$LOG_FILE"
}

start_trading() {
    cd "$PROJECT_DIR"
    source "$VENV_PATH"
    nohup python -m moex_agent.paper_trading --capital 200000 --duration-days 7 >> data/paper_trading.log 2>&1 &
    echo $! > "$PID_FILE"
    log "Started paper trading with PID: $!"
}

check_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # Running
        fi
    fi
    return 1  # Not running
}

# Main loop
log "Watchdog started"

while true; do
    if ! check_process; then
        log "Paper trading not running, restarting..."
        start_trading
    fi
    sleep 60  # Check every minute
done

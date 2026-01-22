#!/bin/bash
# Установка MOEX Agent как системного сервиса

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

echo "=== MOEX Agent Service Installer ==="
echo "Project directory: $PROJECT_DIR"

# Определяем ОС
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected: macOS"

    # Делаем watchdog исполняемым
    chmod +x "$SCRIPTS_DIR/watchdog.sh"

    # Копируем plist
    PLIST_SRC="$SCRIPTS_DIR/com.moex-agent.plist"
    PLIST_DST="$HOME/Library/LaunchAgents/com.moex-agent.plist"

    # Останавливаем если запущен
    launchctl unload "$PLIST_DST" 2>/dev/null

    # Копируем
    cp "$PLIST_SRC" "$PLIST_DST"

    # Загружаем
    launchctl load "$PLIST_DST"

    echo "✅ Сервис установлен и запущен"
    echo ""
    echo "Команды управления:"
    echo "  Остановить:  launchctl unload ~/Library/LaunchAgents/com.moex-agent.plist"
    echo "  Запустить:   launchctl load ~/Library/LaunchAgents/com.moex-agent.plist"
    echo "  Статус:      launchctl list | grep moex"
    echo "  Логи:        tail -f $PROJECT_DIR/data/paper_trading.log"

elif [[ "$OSTYPE" == "linux"* ]]; then
    echo "Detected: Linux"

    SERVICE_SRC="$SCRIPTS_DIR/moex-agent.service"
    SERVICE_DST="/etc/systemd/system/moex-agent.service"

    # Требуется sudo
    if [ "$EUID" -ne 0 ]; then
        echo "Требуется sudo для установки systemd сервиса"
        sudo cp "$SERVICE_SRC" "$SERVICE_DST"
        sudo systemctl daemon-reload
        sudo systemctl enable moex-agent
        sudo systemctl start moex-agent
    else
        cp "$SERVICE_SRC" "$SERVICE_DST"
        systemctl daemon-reload
        systemctl enable moex-agent
        systemctl start moex-agent
    fi

    echo "✅ Сервис установлен и запущен"
    echo ""
    echo "Команды управления:"
    echo "  Статус:      sudo systemctl status moex-agent"
    echo "  Остановить:  sudo systemctl stop moex-agent"
    echo "  Запустить:   sudo systemctl start moex-agent"
    echo "  Логи:        journalctl -u moex-agent -f"

else
    echo "❌ Неподдерживаемая ОС: $OSTYPE"
    exit 1
fi

echo ""
echo "=== Готово ==="

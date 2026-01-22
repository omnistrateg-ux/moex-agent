# MOEX Agent - Инструкция по настройке

## Быстрый старт

```bash
# Активировать виртуальное окружение
source .venv/bin/activate

# Запустить paper trading
python -m moex_agent.paper_trading --capital 200000 --duration-days 7

# Запустить web dashboard
python -m moex_agent.dashboard --port 8080
```

## Компоненты системы

### 1. Автозапуск (macOS)

Установка сервиса для автоматического запуска:

```bash
# Установить сервис
./scripts/install_service.sh

# Проверить статус
launchctl list | grep moex

# Остановить
launchctl unload ~/Library/LaunchAgents/com.moex-agent.plist

# Запустить
launchctl load ~/Library/LaunchAgents/com.moex-agent.plist
```

### 2. Автозапуск (Linux)

```bash
# Установить systemd сервис
sudo cp scripts/moex-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable moex-agent
sudo systemctl start moex-agent

# Проверить статус
sudo systemctl status moex-agent
```

### 3. Cron Jobs

Добавить в crontab (`crontab -e`):

```cron
# Ежедневный отчёт в 21:00
0 21 * * * /Users/artempobedinskij/Desktop/Projects/moex_agent/scripts/daily_report.sh

# Еженедельное переобучение в воскресенье в 03:00
0 3 * * 0 /Users/artempobedinskij/Desktop/Projects/moex_agent/scripts/weekly_retrain.sh
```

### 4. Web Dashboard

```bash
# Запуск на порту 8080
python -m moex_agent.dashboard --port 8080

# Открыть в браузере
open http://localhost:8080
```

### 5. Переобучение моделей

```bash
# Полная оптимизация (с Optuna, долго)
python -m moex_agent.optimize_train --trials 100

# Быстрое переобучение с улучшенными параметрами
python -m moex_agent.optimize_train --skip-optimize
```

## Логи

- `data/paper_trading.log` - основной лог торговли
- `data/logs/moex_agent.log` - все логи (с ротацией)
- `data/logs/errors.log` - только ошибки
- `data/logs/trades.log` - лог сделок

## Риск-менеджмент

Настройки в `moex_agent/risk_manager.py`:

- `max_daily_loss_pct: 2.0` - стоп торговли при дневном убытке 2%
- `max_drawdown_pct: 10.0` - стоп при drawdown 10%
- `pause_after_losses: 3` - пауза после 3 убыточных сделок подряд

## Технические индикаторы

Модели используют следующие индикаторы:
- Returns (1m, 5m, 10m, 30m, 60m)
- RSI (7, 14)
- MACD (12/26/9)
- Bollinger Bands (20, 2σ)
- Stochastic (14/3)
- ADX (14)
- ATR (14)
- OBV
- Volatility
- Moving Averages (SMA20, SMA50)

## Telegram уведомления

Настройки в `config.yaml`:

```yaml
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

## Мониторинг

1. **Dashboard**: http://localhost:8080
2. **Логи**: `tail -f data/paper_trading.log`
3. **Статус**: `python -m moex_agent.paper_trading --status`

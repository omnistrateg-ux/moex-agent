# Настройка MOEX Agent на Replit

## Шаг 1: Создать Repl

1. Зайти на [replit.com](https://replit.com)
2. Create Repl → Import from GitHub
3. Или создать Python Repl и загрузить файлы

## Шаг 2: Загрузить файлы

Загрузить в Replit:
```
moex_agent/          # весь пакет
models/              # обученные модели (25MB)
config.yaml          # конфигурация
requirements.txt
pyproject.toml
.replit
replit.nix
```

**НЕ загружать:**
- `data/moex_agent.sqlite` (3.9GB - слишком большой)
- `.venv/`
- `__pycache__/`

## Шаг 3: Настроить Secrets

В Replit → Tools → Secrets добавить:

| Key | Value |
|-----|-------|
| `TELEGRAM_BOT_TOKEN` | ваш токен бота |
| `TELEGRAM_CHAT_ID` | ваш chat_id |

## Шаг 4: Обновить config.yaml

```yaml
telegram:
  enabled: true
  bot_token: ${TELEGRAM_BOT_TOKEN}
  chat_id: ${TELEGRAM_CHAT_ID}
```

Или использовать переменные окружения в коде.

## Шаг 5: Загрузить данные

В Shell выполнить:
```bash
python -m moex_agent.bootstrap --days 30
```

Это загрузит последние 30 дней свечей (~200MB).

## Шаг 6: Запустить

Нажать Run или:
```bash
python -m moex_agent.margin_paper_trading
```

## Always On (24/7)

Для работы 24/7 нужен Replit Core ($20/месяц):
- Deployments → Always On

**Альтернатива бесплатно:**
- Использовать UptimeRobot для пинга каждые 5 минут
- Добавить простой HTTP endpoint

## Структура после настройки

```
moex_agent/
├── moex_agent/
│   ├── margin_paper_trading.py
│   ├── margin_risk_engine.py
│   ├── bcs_broker.py
│   └── ...
├── models/
│   ├── model_time_5m.joblib
│   ├── model_time_10m.joblib
│   ├── model_time_30m.joblib
│   └── model_time_1h.joblib
├── data/
│   └── moex_agent.sqlite  # создастся автоматически
├── config.yaml
├── .replit
└── pyproject.toml
```

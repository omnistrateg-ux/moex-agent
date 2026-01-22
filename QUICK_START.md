# MOEX Agent — Quick Start для Replit

## 1. Загрузка проекта

1. Создайте новый Replit проект (Python)
2. Загрузите `moex_agent_full.zip`
3. Разархивируйте в корень проекта

## 2. Настройка Secrets

В Replit → Tools → Secrets добавьте:

```
TELEGRAM_BOT_TOKEN = ваш_токен_бота
TELEGRAM_CHAT_ID = ваш_chat_id
```

**Как получить:**
1. Создайте бота через @BotFather → получите токен
2. Напишите боту, затем откройте: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Найдите `chat.id` в ответе

## 3. Запуск

Нажмите **Run** или:
```bash
python main.py
```

## 4. Что происходит при запуске

1. **Bootstrap** — загрузка данных MOEX за 7 дней (~5 мин)
2. **Web Server** — запуск на порту 8080
3. **Paper Trading** — запуск в фоне через 10 сек

## 5. Dashboard

Откройте URL вашего Replit:
- `https://your-repl.replit.app/` — Dashboard
- `https://your-repl.replit.app/api/status` — API статус
- `https://your-repl.replit.app/api/equity` — Equity и P&L

## 6. Структура

```
moex_agent/          # Ядро системы
models/              # ML модели (4 шт, ~13MB)
data/                # Данные и состояние
config.yaml          # Конфигурация
main.py              # Entry point
```

## 7. Модели

| Горизонт | Win Rate | Profit Factor |
|----------|----------|---------------|
| 5m       | 56.8%    | 2.33          |
| 10m      | 56.0%    | 2.31          |
| 30m      | 56.0%    | 2.39          |
| 1h       | 55.4%    | 2.39          |

## 8. Troubleshooting

**Нет свечей:**
```bash
python -m moex_agent.bootstrap --days 3
```

**Telegram не работает:**
Проверьте Secrets → токен и chat_id

**Медленно:**
Уменьшите количество тикеров в `config.yaml`

---

**Готово!** Dashboard доступен на порту 8080.

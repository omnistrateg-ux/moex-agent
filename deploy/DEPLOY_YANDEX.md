# MOEX Agent — Деплой в Yandex Cloud

Инструкция по развёртыванию MOEX Agent в Yandex Cloud Serverless Containers с PostgreSQL.

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Yandex Cloud                            │
│  ┌─────────────────┐     ┌─────────────────────────────┐   │
│  │   Serverless    │     │   Managed PostgreSQL        │   │
│  │   Container     │────▶│   (moex-agent-db)           │   │
│  │   (moex-agent)  │     │   - Master: rc1b-...        │   │
│  └────────┬────────┘     │   - Replica: rc1d-...       │   │
│           │              └─────────────────────────────┘   │
│           │                                                 │
│  ┌────────▼────────┐                                       │
│  │   Container     │                                       │
│  │   Registry      │                                       │
│  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
            │
            │ HTTPS
            ▼
┌─────────────────────────┐
│   Telegram Bot API      │
│   (уведомления)         │
└─────────────────────────┘
```

## Инфраструктура (уже создана)

| Компонент | Значение |
|-----------|----------|
| **Organization** | cloud-omnistrateg |
| **Folder** | moex-agent (b1g58hgssbaevs2m924t) |
| **PostgreSQL Cluster** | moex-agent-db |
| **Master Host** | rc1b-5421i5tvv2p060mk.mdb.yandexcloud.net |
| **Replica Host** | rc1d-4o16qjh1tbppbt94.mdb.yandexcloud.net |
| **Database** | moexdb |
| **User** | moexagent |
| **Port** | 6432 |

## Быстрый старт

### 1. Установка Yandex Cloud CLI

```bash
# macOS / Linux
curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

# Перезапустите терминал или выполните:
source ~/.zshrc  # или ~/.bashrc
```

### 2. Авторизация

```bash
yc init
```

Выберите:
- Organization: `cloud-omnistrateg`
- Folder: `moex-agent`

### 3. Установка переменных окружения

```bash
# Обязательные
export DATABASE_URL="postgresql://moexagent:MoexAgent2026!Secure@rc1b-5421i5tvv2p060mk.mdb.yandexcloud.net:6432/moexdb?sslmode=require"

# Telegram (опционально, но рекомендуется)
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Yandex GPT (опционально)
export YANDEX_API_KEY="your_api_key"
```

### 4. Деплой

```bash
cd /Users/artempobedinskij/Desktop/Projects/moex_agent

# Сделать скрипт исполняемым
chmod +x deploy/yandex-deploy.sh

# Запустить деплой
./deploy/yandex-deploy.sh
```

## Локальное тестирование

Перед деплоем в облако рекомендуется протестировать локально:

```bash
cd /Users/artempobedinskij/Desktop/Projects/moex_agent

# Запуск с локальным PostgreSQL
docker-compose -f deploy/docker-compose.yml up --build

# Или с Yandex Cloud PostgreSQL (если есть доступ)
docker build -f deploy/Dockerfile -t moex-agent .
docker run -p 8080:8080 \
  -e DATABASE_URL="postgresql://moexagent:MoexAgent2026!Secure@rc1b-5421i5tvv2p060mk.mdb.yandexcloud.net:6432/moexdb?sslmode=require" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  moex-agent
```

Открыть: http://localhost:8080

## Переменные окружения

| Переменная | Обязательная | Описание |
|------------|--------------|----------|
| `DATABASE_URL` | Да | PostgreSQL connection string |
| `PORT` | Нет (8080) | Порт web-сервера |
| `TELEGRAM_BOT_TOKEN` | Нет | Токен Telegram бота |
| `TELEGRAM_CHAT_ID` | Нет | ID чата для уведомлений |
| `YANDEX_API_KEY` | Нет | API ключ для YandexGPT |

## Ресурсы контейнера

Настройки по умолчанию (можно изменить в `yandex-deploy.sh`):

| Параметр | Значение | Описание |
|----------|----------|----------|
| `MEMORY` | 1024Mi | RAM |
| `CORES` | 1 | vCPU |
| `CORE_FRACTION` | 100% | Гарантированная доля CPU |
| `EXECUTION_TIMEOUT` | 300s | Таймаут (для ML операций) |
| `CONCURRENCY` | 1 | Одновременных запросов |

## Endpoints

После деплоя доступны:

| Endpoint | Описание |
|----------|----------|
| `/` | Web Dashboard |
| `/api/health` | Health check |
| `/api/status` | Статус системы |
| `/api/positions` | Текущие позиции |
| `/api/signals` | История сигналов |
| `/api/equity` | График equity |

## Мониторинг

### Логи контейнера

```bash
# Получить ID контейнера
yc serverless container list

# Посмотреть логи
yc logging read --group-id <container-id> --limit 100
```

### PostgreSQL

```bash
# Подключение через psql
psql "postgresql://moexagent:MoexAgent2026!Secure@rc1b-5421i5tvv2p060mk.mdb.yandexcloud.net:6432/moexdb?sslmode=require"

# Проверка состояния
SELECT * FROM trading_state;
```

## Обновление

```bash
# Пересобрать и задеплоить новую версию
./deploy/yandex-deploy.sh
```

## Удаление

```bash
# Удалить контейнер
yc serverless container delete --name moex-agent

# Удалить registry (осторожно!)
yc container registry delete --name moex-agent-registry
```

## Troubleshooting

### Ошибка подключения к PostgreSQL

1. Проверьте, что IP разрешён в security group
2. Проверьте правильность DATABASE_URL
3. Проверьте статус кластера: `yc managed-postgresql cluster get moex-agent-db`

### Container не запускается

1. Проверьте логи: `yc logging read ...`
2. Проверьте health check: `curl https://<container-url>/api/health`
3. Убедитесь, что образ собрался без ошибок

### Telegram не работает

1. Проверьте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID
2. Убедитесь, что бот добавлен в чат
3. Проверьте логи на ошибки отправки

## Стоимость

Примерная стоимость при активном использовании:

| Сервис | Стоимость |
|--------|-----------|
| Serverless Containers | ~500-1500 ₽/мес |
| Managed PostgreSQL (b2.micro) | ~2000 ₽/мес |
| Container Registry | ~50 ₽/мес |
| **Итого** | **~2500-3500 ₽/мес** |

*Цены могут меняться, см. https://cloud.yandex.ru/prices*

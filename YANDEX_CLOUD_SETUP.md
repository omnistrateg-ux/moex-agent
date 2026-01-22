# Настройка Yandex Cloud для MOEX Agent

## 1. Создание аккаунта

1. Зарегистрируйтесь: https://console.cloud.yandex.ru/
2. Привяжите карту (дают 4000₽ на старт)
3. Создайте каталог (folder)

---

## 2. Yandex DataSphere (обучение моделей)

### 2.1 Создание проекта

1. https://console.cloud.yandex.ru/datasphere
2. **Создать проект** → укажите имя "moex-agent"
3. Выберите каталог

### 2.2 Конфигурация вычислений

| Конфигурация | CPU | RAM | GPU | Цена/час |
|--------------|-----|-----|-----|----------|
| c1.4 | 4 | 16GB | - | ~15₽ |
| g1.1 | 8 | 48GB | 1x V100 | ~150₽ |
| g2.1 | 28 | 119GB | 1x A100 | ~400₽ |

**Рекомендация:** `g1.1` для быстрого обучения с GPU.

### 2.3 Загрузка notebook

1. В проекте нажмите **JupyterLab**
2. File → Upload → выберите `notebooks/train_yandex_datasphere.ipynb`
3. Или: Terminal → `git clone https://github.com/omnistrateg-ux/moex-agent.git`

### 2.4 Запуск обучения

1. Откройте notebook
2. Kernel → Change Kernel → выберите с GPU
3. Run All Cells

---

## 3. Yandex Object Storage (хранение моделей)

### 3.1 Создание bucket

1. https://console.cloud.yandex.ru/storage
2. **Создать бакет**
3. Имя: `moex-agent-models`
4. Доступ: **Ограниченный**

### 3.2 Создание ключей доступа

1. https://console.cloud.yandex.ru/iam/service-accounts
2. **Создать сервисный аккаунт**
3. Имя: `moex-agent-storage`
4. Роль: `storage.editor`
5. **Создать статический ключ доступа**
6. Сохраните:
   - Access Key ID → `YANDEX_S3_KEY`
   - Secret Key → `YANDEX_S3_SECRET`

### 3.3 Добавление в Replit Secrets

| Secret | Значение |
|--------|----------|
| `YANDEX_S3_KEY` | ваш Access Key ID |
| `YANDEX_S3_SECRET` | ваш Secret Key |

---

## 4. YandexGPT (анализ сигналов)

### 4.1 Получение API ключа

1. https://console.cloud.yandex.ru/iam/service-accounts
2. Выберите сервисный аккаунт
3. **Создать новый ключ** → **API-ключ**
4. Сохраните ключ → `YANDEX_API_KEY`

### 4.2 Folder ID

Находится в URL консоли:
```
https://console.cloud.yandex.ru/folders/YOUR_FOLDER_ID
                                      ^^^^^^^^^^^^^^
```

### 4.3 Добавление в Replit Secrets

| Secret | Значение |
|--------|----------|
| `YANDEX_API_KEY` | ваш API ключ |
| `YANDEX_FOLDER_ID` | ID каталога |

---

## 5. Автоматизация переобучения

### 5.1 Yandex Cloud Functions

Создайте функцию для автоматического переобучения:

```python
# index.py для Cloud Functions
import boto3
import subprocess
import os

def handler(event, context):
    # Клонируем репо
    subprocess.run(["git", "clone", "https://github.com/omnistrateg-ux/moex-agent.git"])
    os.chdir("moex-agent")

    # Устанавливаем зависимости
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

    # Обучаем
    subprocess.run(["python", "-m", "moex_agent.advanced_train"])

    # Загружаем в S3
    s3 = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=os.environ['YANDEX_S3_KEY'],
        aws_secret_access_key=os.environ['YANDEX_S3_SECRET'],
    )

    for h in ['5m', '10m', '30m', '1h']:
        s3.upload_file(
            f'models/model_time_{h}.joblib',
            'moex-agent-models',
            f'models/model_time_{h}.joblib'
        )

    return {"status": "ok", "message": "Models trained and uploaded"}
```

### 5.2 Триггер по расписанию

1. https://console.cloud.yandex.ru/functions
2. Выберите функцию → **Триггеры**
3. **Создать триггер** → **Таймер**
4. Расписание: `0 3 ? * SUN *` (каждое воскресенье в 3:00)

---

## 6. Скрипт синхронизации моделей

Добавьте в Replit для автоматической загрузки новых моделей:

```python
# scripts/sync_models_yandex.py
import boto3
import os
from datetime import datetime

def sync_models():
    """Синхронизация моделей из Yandex Object Storage."""

    s3 = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=os.environ.get('YANDEX_S3_KEY'),
        aws_secret_access_key=os.environ.get('YANDEX_S3_SECRET'),
    )

    BUCKET = 'moex-agent-models'
    HORIZONS = ['5m', '10m', '30m', '1h']

    print(f"Синхронизация моделей из Yandex Cloud...")

    for h in HORIZONS:
        remote_key = f'models/model_time_{h}.joblib'
        local_path = f'models/model_time_{h}.joblib'

        try:
            # Проверяем дату изменения
            response = s3.head_object(Bucket=BUCKET, Key=remote_key)
            remote_modified = response['LastModified']

            # Проверяем локальную версию
            if os.path.exists(local_path):
                local_modified = datetime.fromtimestamp(
                    os.path.getmtime(local_path),
                    tz=remote_modified.tzinfo
                )

                if remote_modified <= local_modified:
                    print(f"  {h}: актуальна")
                    continue

            # Скачиваем
            s3.download_file(BUCKET, remote_key, local_path)
            print(f"  {h}: обновлена ✓")

        except Exception as e:
            print(f"  {h}: ошибка - {e}")

    # Meta.json
    try:
        s3.download_file(BUCKET, 'models/meta.json', 'models/meta.json')
        print("  meta.json: обновлен ✓")
    except:
        pass

    print("Готово!")

if __name__ == "__main__":
    sync_models()
```

---

## 7. Итоговые Secrets для Replit

| Secret | Описание |
|--------|----------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота |
| `TELEGRAM_CHAT_ID` | Chat ID (120171956) |
| `YANDEX_API_KEY` | API ключ для YandexGPT |
| `YANDEX_FOLDER_ID` | ID каталога Yandex Cloud |
| `YANDEX_S3_KEY` | Access Key для Object Storage |
| `YANDEX_S3_SECRET` | Secret Key для Object Storage |

---

## 8. Стоимость

| Сервис | Использование | Примерная стоимость |
|--------|---------------|---------------------|
| DataSphere g1.1 | 1 час/неделю | ~600₽/мес |
| Object Storage | 100 MB | ~5₽/мес |
| YandexGPT | 1000 запросов | ~100₽/мес |
| Cloud Functions | 1 запуск/неделю | ~10₽/мес |
| **Итого** | | **~715₽/мес** |

**Бесплатный грант:** 4000₽ на 60 дней для новых пользователей.

---

## Быстрый старт

```bash
# 1. В Replit Shell установите boto3
pip install boto3

# 2. Синхронизируйте модели
python scripts/sync_models_yandex.py

# 3. Перезапустите приложение
python main.py
```

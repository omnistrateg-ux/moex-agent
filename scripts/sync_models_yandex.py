#!/usr/bin/env python3
"""
Синхронизация ML моделей из Yandex Object Storage.

Использование:
    python scripts/sync_models_yandex.py

Требуются Secrets:
    YANDEX_S3_KEY - Access Key ID
    YANDEX_S3_SECRET - Secret Access Key
"""

import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import boto3
except ImportError:
    print("Установите boto3: pip install boto3")
    sys.exit(1)


BUCKET = "moex-agent-models"
HORIZONS = ["5m", "10m", "30m", "1h"]
MODELS_DIR = Path("models")


def get_s3_client():
    """Создать S3 клиент для Yandex Object Storage."""
    key = os.environ.get("YANDEX_S3_KEY")
    secret = os.environ.get("YANDEX_S3_SECRET")

    if not key or not secret:
        print("❌ Ошибка: установите YANDEX_S3_KEY и YANDEX_S3_SECRET")
        print("\nВ Replit: Tools → Secrets → добавьте ключи")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def sync_models():
    """Синхронизировать модели из Yandex Cloud."""
    print("=" * 50)
    print("Синхронизация моделей из Yandex Object Storage")
    print("=" * 50)

    s3 = get_s3_client()

    # Создаём папку если нет
    MODELS_DIR.mkdir(exist_ok=True)

    updated = 0
    skipped = 0
    errors = 0

    for horizon in HORIZONS:
        remote_key = f"models/model_time_{horizon}.joblib"
        local_path = MODELS_DIR / f"model_time_{horizon}.joblib"

        try:
            # Проверяем дату на сервере
            response = s3.head_object(Bucket=BUCKET, Key=remote_key)
            remote_modified = response["LastModified"]
            remote_size = response["ContentLength"]

            # Проверяем локальную версию
            if local_path.exists():
                local_modified = datetime.fromtimestamp(
                    local_path.stat().st_mtime, tz=remote_modified.tzinfo
                )
                local_size = local_path.stat().st_size

                # Если размеры совпадают и локальный новее - пропускаем
                if local_size == remote_size and remote_modified <= local_modified:
                    print(f"  {horizon}: ✓ актуальна ({remote_size / 1024 / 1024:.1f} MB)")
                    skipped += 1
                    continue

            # Скачиваем
            print(f"  {horizon}: загрузка ({remote_size / 1024 / 1024:.1f} MB)...", end=" ")
            s3.download_file(BUCKET, remote_key, str(local_path))
            print("✓")
            updated += 1

        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"  {horizon}: ⚠ не найдена на сервере")
            else:
                print(f"  {horizon}: ❌ ошибка - {e}")
            errors += 1
        except Exception as e:
            print(f"  {horizon}: ❌ ошибка - {e}")
            errors += 1

    # Meta.json
    try:
        meta_remote = "models/meta.json"
        meta_local = MODELS_DIR / "meta.json"

        s3.download_file(BUCKET, meta_remote, str(meta_local))
        print(f"  meta.json: ✓")
    except Exception as e:
        print(f"  meta.json: ⚠ {e}")

    # Итог
    print("\n" + "=" * 50)
    print(f"Обновлено: {updated}, Актуальны: {skipped}, Ошибки: {errors}")
    print("=" * 50)

    return errors == 0


def upload_models():
    """Загрузить локальные модели в Yandex Cloud."""
    print("=" * 50)
    print("Загрузка моделей в Yandex Object Storage")
    print("=" * 50)

    s3 = get_s3_client()

    for horizon in HORIZONS:
        local_path = MODELS_DIR / f"model_time_{horizon}.joblib"

        if not local_path.exists():
            print(f"  {horizon}: ⚠ не найдена локально")
            continue

        remote_key = f"models/model_time_{horizon}.joblib"
        size = local_path.stat().st_size

        print(f"  {horizon}: загрузка ({size / 1024 / 1024:.1f} MB)...", end=" ")
        s3.upload_file(str(local_path), BUCKET, remote_key)
        print("✓")

    # Meta.json
    meta_local = MODELS_DIR / "meta.json"
    if meta_local.exists():
        s3.upload_file(str(meta_local), BUCKET, "models/meta.json")
        print(f"  meta.json: ✓")

    print("\n✅ Загрузка завершена!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Синхронизация моделей с Yandex Cloud")
    parser.add_argument("--upload", action="store_true", help="Загрузить модели в облако")
    args = parser.parse_args()

    if args.upload:
        upload_models()
    else:
        sync_models()

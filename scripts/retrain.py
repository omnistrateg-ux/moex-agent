#!/usr/bin/env python3
"""
Переобучение ML моделей в Replit.

Запуск:
    python scripts/retrain.py [--days 30] [--horizon 5m]

Примеры:
    python scripts/retrain.py                    # Все модели, 30 дней
    python scripts/retrain.py --days 60          # Все модели, 60 дней
    python scripts/retrain.py --horizon 5m       # Только 5-минутная модель
    python scripts/retrain.py --no-backup        # Без бэкапа старых моделей
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "data" / "retrain.log"),
    ],
)
logger = logging.getLogger(__name__)

# Конфигурация
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_split": 50,
    "min_samples_leaf": 20,
    "random_state": 42,
}

HORIZONS = ["5m", "10m", "30m", "1h"]
N_SPLITS = 5
MODELS_DIR = ROOT / "models"
BACKUP_DIR = MODELS_DIR / "backup"


def load_data(days: int) -> pd.DataFrame:
    """Загрузить данные с MOEX."""
    from moex_agent.bootstrap import bootstrap_recent
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect

    logger.info(f"📊 Загрузка данных с MOEX ({days} дней)...")

    config = load_config()
    conn = connect(config.sqlite_path)

    # Проверяем текущее количество
    cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
    before = cur.fetchone()["cnt"]

    # Загружаем свежие данные
    try:
        bootstrap_recent(conn, config, days=days)
    except Exception as e:
        logger.warning(f"Ошибка загрузки: {e}")

    # Проверяем после
    cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
    after = cur.fetchone()["cnt"]

    logger.info(f"   Было: {before:,} → Стало: {after:,} свечей")

    # Загружаем все данные
    df = pd.read_sql("SELECT * FROM candles ORDER BY secid, ts", conn)
    conn.close()

    return df


def build_features_for_ticker(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """Построить фичи для одного тикера."""
    from moex_agent.features import FEATURE_COLS, build_feature_frame

    candles = ticker_df.to_dict("records")
    features_df = build_feature_frame(candles)

    return features_df[FEATURE_COLS]


def create_labels_for_horizon(df: pd.DataFrame, horizon: str) -> pd.Series:
    """Создать метки для горизонта."""
    # Определяем период в минутах
    periods = {"5m": 5, "10m": 10, "30m": 30, "1h": 60}
    period = periods.get(horizon, 5)

    # Будущая доходность
    future_return = df["close"].shift(-period) / df["close"] - 1

    # 1 если рост, 0 если падение
    labels = (future_return > 0).astype(int)

    return labels


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> tuple:
    """Walk-Forward обучение модели."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []
    best_model = None
    best_score = 0

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = GradientBoostingClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Метрики
        accuracy = (y_pred == y_test).mean()

        # Win Rate для высокой уверенности
        high_conf = y_prob > 0.55
        if high_conf.sum() > 0:
            wins = ((y_pred == y_test) & high_conf).sum()
            win_rate = wins / high_conf.sum()
        else:
            win_rate = 0.5

        results.append({
            "fold": fold + 1,
            "accuracy": accuracy,
            "win_rate": win_rate,
            "trades": int(high_conf.sum()),
        })

        logger.info(f"   Fold {fold + 1}: acc={accuracy:.3f}, wr={win_rate:.3f}, trades={high_conf.sum()}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = model

    # Средние метрики
    avg_metrics = {
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "win_rate": np.mean([r["win_rate"] for r in results]),
        "trades": int(np.mean([r["trades"] for r in results])),
    }

    return best_model, avg_metrics


def backup_models():
    """Создать бэкап текущих моделей."""
    BACKUP_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(exist_ok=True)

    for model_file in MODELS_DIR.glob("*.joblib"):
        shutil.copy(model_file, backup_path / model_file.name)

    meta_file = MODELS_DIR / "meta.json"
    if meta_file.exists():
        shutil.copy(meta_file, backup_path / "meta.json")

    logger.info(f"💾 Бэкап создан: {backup_path}")
    return backup_path


def send_notification(message: str):
    """Отправить уведомление в Telegram."""
    try:
        from moex_agent.telegram import send_telegram
        send_telegram(message)
        logger.info("📱 Уведомление отправлено в Telegram")
    except Exception as e:
        logger.warning(f"Telegram: {e}")


def main():
    parser = argparse.ArgumentParser(description="Переобучение ML моделей")
    parser.add_argument("--days", type=int, default=30, help="Дней данных для обучения")
    parser.add_argument("--horizon", type=str, help="Конкретный горизонт (5m/10m/30m/1h)")
    parser.add_argument("--no-backup", action="store_true", help="Не создавать бэкап")
    parser.add_argument("--no-notify", action="store_true", help="Не отправлять в Telegram")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("🚀 MOEX Agent — Переобучение моделей")
    logger.info("=" * 50)

    # Создаём папки
    MODELS_DIR.mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    # Бэкап
    if not args.no_backup:
        backup_models()

    # Загрузка данных
    df = load_data(args.days)

    if len(df) < 10000:
        logger.error("❌ Недостаточно данных для обучения!")
        sys.exit(1)

    # Определяем горизонты
    horizons = [args.horizon] if args.horizon else HORIZONS

    # Метаданные
    meta = {
        "trained_at": datetime.now().isoformat(),
        "platform": "Replit",
        "days_of_data": args.days,
        "candles_used": len(df),
        "horizons": {},
    }

    # Обучение
    for horizon in horizons:
        logger.info(f"\n🔄 Обучение модели {horizon}...")

        all_X = []
        all_y = []

        # Собираем данные по тикерам
        for ticker in df["secid"].unique():
            ticker_df = df[df["secid"] == ticker].copy()

            if len(ticker_df) < 500:
                continue

            try:
                # Фичи
                X = build_features_for_ticker(ticker_df)

                # Метки
                y = create_labels_for_horizon(ticker_df, horizon)

                # Выравниваем
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]

                all_X.append(X)
                all_y.append(y)

            except Exception as e:
                logger.warning(f"   {ticker}: ошибка - {e}")
                continue

        if not all_X:
            logger.error(f"   ❌ Нет данных для {horizon}")
            continue

        # Объединяем
        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)

        # Убираем NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        logger.info(f"   Данных для обучения: {len(X):,}")

        # Обучение
        model, metrics = train_model(X, y, N_SPLITS)

        # Сохранение
        model_path = MODELS_DIR / f"model_time_{horizon}.joblib"
        joblib.dump(model, model_path)

        logger.info(f"   ✅ Сохранено: {model_path.name}")
        logger.info(f"   📈 WR={metrics['win_rate']:.1%}, Acc={metrics['accuracy']:.1%}")

        meta["horizons"][horizon] = {
            "win_rate": round(metrics["win_rate"], 3),
            "accuracy": round(metrics["accuracy"], 3),
            "trades": metrics["trades"],
        }

    # Сохраняем метаданные
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Итог
    logger.info("\n" + "=" * 50)
    logger.info("✅ Переобучение завершено!")
    logger.info("=" * 50)

    for h, m in meta["horizons"].items():
        logger.info(f"   {h}: WR={m['win_rate']:.1%}, Acc={m['accuracy']:.1%}")

    # Уведомление
    if not args.no_notify:
        results = ", ".join([f"{h}={m['win_rate']:.0%}" for h, m in meta["horizons"].items()])
        send_notification(f"✅ Модели переобучены!\n{results}")


if __name__ == "__main__":
    main()

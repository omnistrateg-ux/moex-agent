"""
MOEX Agent - Продвинутое обучение на 4 годах данных

Использует:
- Расширенный набор фич (29+)
- Walk-forward validation
- Ensemble моделей
- Оптимизация под profit factor

Usage:
    python -m moex_agent.advanced_train
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .features import build_feature_frame
from .labels import make_time_exit_labels
from .predictor import FEATURE_COLS
from .storage import connect

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.advanced_train")


def walk_forward_backtest(
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    model_params: Dict,
    n_splits: int = 5,
    p_threshold: float = 0.52,
    take_atr: float = 0.7,
    stop_atr: float = 0.4,
) -> Dict:
    """
    Walk-forward validation для оценки реальной производительности.

    Симулирует реальную торговлю: обучение на прошлом, тест на будущем.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        "trades": [],
        "pnl": [],
        "win_rates": [],
        "profit_factors": [],
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        prices_test = prices[test_idx]
        atr_test = atr[test_idx]

        # Обучение модели
        model = HistGradientBoostingClassifier(**model_params, random_state=42)
        model.fit(X_train, y_train)

        # Предсказания
        y_proba = model.predict_proba(X_test)[:, 1]

        # Симуляция сделок
        fold_trades = []
        for i in range(len(y_proba)):
            if y_proba[i] < p_threshold:
                continue
            if prices_test[i] <= 0 or atr_test[i] <= 0:
                continue

            # Рассчёт PnL
            if y_test[i] == 1:
                pnl_pct = take_atr * atr_test[i] / prices_test[i] * 100
            else:
                pnl_pct = -stop_atr * atr_test[i] / prices_test[i] * 100

            fold_trades.append(pnl_pct)

        if fold_trades:
            wins = [t for t in fold_trades if t > 0]
            losses = [t for t in fold_trades if t <= 0]

            win_rate = len(wins) / len(fold_trades) * 100
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

            results["trades"].append(len(fold_trades))
            results["pnl"].append(sum(fold_trades))
            results["win_rates"].append(win_rate)
            results["profit_factors"].append(profit_factor)

            logger.debug(f"Fold {fold+1}: trades={len(fold_trades)}, WR={win_rate:.1f}%, PF={profit_factor:.2f}")

    if results["win_rates"]:
        return {
            "avg_trades": np.mean(results["trades"]),
            "avg_pnl": np.mean(results["pnl"]),
            "avg_win_rate": np.mean(results["win_rates"]),
            "avg_profit_factor": np.mean(results["profit_factors"]),
            "std_win_rate": np.std(results["win_rates"]),
            "std_profit_factor": np.std(results["profit_factors"]),
            "pnl": results["pnl"],  # Для расчёта Sharpe
        }

    return {"avg_win_rate": 0, "avg_profit_factor": 0, "pnl": []}


def create_ensemble_model(X: np.ndarray, y: np.ndarray) -> object:
    """
    Создание ensemble модели из нескольких алгоритмов.
    """
    # Базовые модели
    hgb = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    # Масштабирование для логистической регрессии
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
    )

    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("hgb", hgb),
            ("rf", rf),
        ],
        voting="soft",
        weights=[0.6, 0.4],  # HGB важнее
    )

    # Калибровка
    calibrated = CalibratedClassifierCV(
        ensemble,
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=3),
    )

    return calibrated


def train_horizon_model(
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    horizon: str,
    use_ensemble: bool = False,
) -> Tuple[object, Dict]:
    """
    Обучение модели для одного горизонта.
    """
    logger.info(f"Обучение модели {horizon}...")
    logger.info(f"  Данных: {len(X):,}, позитивных: {y.sum():,} ({y.mean()*100:.1f}%)")

    # Оптимальные параметры
    model_params = {
        "max_depth": 7,
        "learning_rate": 0.05,
        "max_iter": 300,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1,
    }

    # Walk-forward validation
    logger.info("  Walk-forward validation...")
    wf_results = walk_forward_backtest(
        X, y, prices, atr,
        model_params=model_params,
        n_splits=5,
    )

    logger.info(f"  WF Results: WR={wf_results.get('avg_win_rate', 0):.1f}%, PF={wf_results.get('avg_profit_factor', 0):.2f}")

    # Финальное обучение на всех данных
    if use_ensemble:
        logger.info("  Training ensemble model...")
        model = create_ensemble_model(X, y)
    else:
        base = HistGradientBoostingClassifier(**model_params, random_state=42)
        model = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=3))

    model.fit(X, y)

    # ВАЖНО: Используем Walk-Forward метрики как финальные (честная оценка)
    # In-sample метрики на тренировочных данных переоценены из-за data leakage
    wf_wr = wf_results.get("avg_win_rate", 0)
    wf_pf = wf_results.get("avg_profit_factor", 0)

    # Sharpe оцениваем по walk-forward PnL
    wf_pnl_list = wf_results.get("pnl", [])
    if wf_pnl_list and len(wf_pnl_list) > 1:
        sharpe = np.mean(wf_pnl_list) / (np.std(wf_pnl_list) + 1e-9)
    else:
        sharpe = 0

    metrics = {
        "win_rate": wf_wr,  # Walk-forward WR (честная оценка)
        "profit_factor": wf_pf,  # Walk-forward PF (честная оценка)
        "sharpe": sharpe,
        "total_trades": int(wf_results.get("avg_trades", 0) * 5),  # ~5 folds
        "wf_win_rate": wf_wr,
        "wf_profit_factor": wf_pf,
    }

    logger.info(f"  Final: WR={wf_wr:.1f}%, PF={wf_pf:.2f}, Sharpe={sharpe:.2f}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Продвинутое обучение MOEX Agent")
    parser.add_argument("--ensemble", action="store_true", help="Использовать ensemble моделей")
    parser.add_argument("--horizons", nargs="+", default=None, help="Горизонты для обучения")
    args = parser.parse_args()

    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    # Загрузка данных
    logger.info("Загрузка данных...")
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = 1
    ORDER BY secid, ts
    """
    candles = pd.read_sql_query(q, conn)
    logger.info(f"Загружено {len(candles):,} свечей")

    if len(candles) < 1_000_000:
        logger.warning("Мало данных! Рекомендуется минимум 1M свечей.")

    # Построение фич
    logger.info("Построение фич (29 индикаторов)...")
    feats = build_feature_frame(candles)

    # Создание меток
    horizons = [(h.name, h.minutes) for h in cfg.horizons]
    if args.horizons:
        horizons = [(h, m) for h, m in horizons if h in args.horizons]

    logger.info("Создание меток...")
    labels = make_time_exit_labels(candles, horizons=horizons)

    # Объединение
    df = feats.merge(labels, on=["secid", "ts"], how="inner")

    # Проверка наличия всех фич
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Отсутствующие фичи: {missing_cols}")
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
    else:
        available_cols = FEATURE_COLS

    df = df.dropna(subset=available_cols)
    logger.info(f"Данных для обучения: {len(df):,}")

    # Подготовка данных
    X = df[available_cols].to_numpy(dtype=float)
    prices = df["close"].to_numpy(dtype=float) if "close" in df.columns else np.ones(len(df))
    atr = df["atr_14"].to_numpy(dtype=float) if "atr_14" in df.columns else np.ones(len(df)) * 0.01

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    meta = {}

    # Обучение для каждого горизонта
    for name, minutes in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Горизонт: {name} ({minutes} мин)")
        logger.info(f"{'='*60}")

        ycol = f"y_time_{name}"
        if ycol not in df.columns:
            logger.warning(f"Метка {ycol} не найдена")
            continue

        y = df[ycol].to_numpy(dtype=int)

        model, metrics = train_horizon_model(
            X, y, prices, atr,
            horizon=name,
            use_ensemble=args.ensemble,
        )

        # Сохранение
        model_path = models_dir / f"model_time_{name}.joblib"
        joblib.dump(model, model_path)

        meta[name] = {
            "type": "advanced-4y",
            "path": str(model_path),
            "features": available_cols,
            "metrics": metrics,
            "trained_at": datetime.now().isoformat(),
            "data_size": len(df),
        }

        logger.info(f"Модель сохранена: {model_path}")

    # Сохранение метаданных
    meta_path = models_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Итоговый отчёт
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    logger.info("=" * 60)

    for horizon, info in meta.items():
        m = info["metrics"]
        logger.info(f"{horizon}: WR={m['win_rate']:.1f}%, PF={m['profit_factor']:.2f}, "
                   f"Sharpe={m['sharpe']:.2f}, WF_WR={m['wf_win_rate']:.1f}%")

    logger.info("\n✅ Обучение завершено!")

    conn.close()


if __name__ == "__main__":
    main()

"""
MOEX Agent - Оптимизация моделей для максимизации прибыли

Использует Optuna для поиска лучших гиперпараметров ML моделей
и оптимальных уровней take-profit/stop-loss.

Usage:
    python -m moex_agent.optimize_train --trials 100
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .config import load_config
from .features import build_feature_frame
from .labels import make_time_exit_labels
from .storage import connect

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.optimize")

# Import feature columns from predictor for consistency
from .predictor import FEATURE_COLS


@dataclass
class BacktestResult:
    """Результаты бэктеста."""
    total_trades: int
    wins: int
    losses: int
    total_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    sharpe: float
    win_rate: float


def simulate_trades(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    p_threshold: float = 0.5,
    take_atr: float = 0.5,
    stop_atr: float = 0.3,
) -> BacktestResult:
    """
    Симуляция сделок для оценки прибыльности модели.

    Args:
        y_true: Истинные метки (1 = рост, 0 = падение)
        y_pred_proba: Предсказанные вероятности
        prices: Цены закрытия
        atr: ATR для расчёта take/stop
        p_threshold: Порог вероятности для открытия сделки
        take_atr: Множитель ATR для take-profit
        stop_atr: Множитель ATR для stop-loss

    Returns:
        BacktestResult с метриками
    """
    trades_pnl = []

    for i in range(len(y_pred_proba)):
        if y_pred_proba[i] < p_threshold:
            continue

        if prices[i] <= 0 or atr[i] <= 0:
            continue

        # Симулируем сделку
        entry_price = prices[i]
        take_price = entry_price * (1 + take_atr * atr[i] / entry_price)
        stop_price = entry_price * (1 - stop_atr * atr[i] / entry_price)

        # Упрощённая симуляция: если y_true=1, цена пошла вверх
        if y_true[i] == 1:
            # Цена пошла вверх - take profit
            pnl_pct = (take_price - entry_price) / entry_price * 100
        else:
            # Цена пошла вниз - stop loss
            pnl_pct = (stop_price - entry_price) / entry_price * 100

        trades_pnl.append(pnl_pct)

    if not trades_pnl:
        return BacktestResult(
            total_trades=0, wins=0, losses=0,
            total_pnl_pct=0, avg_win_pct=0, avg_loss_pct=0,
            profit_factor=0, sharpe=0, win_rate=0
        )

    trades_pnl = np.array(trades_pnl)
    wins = trades_pnl[trades_pnl > 0]
    losses = trades_pnl[trades_pnl <= 0]

    total_pnl = float(np.sum(trades_pnl))
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    # Sharpe ratio (упрощённый)
    sharpe = float(np.mean(trades_pnl) / np.std(trades_pnl)) if np.std(trades_pnl) > 0 else 0

    win_rate = len(wins) / len(trades_pnl) * 100

    return BacktestResult(
        total_trades=len(trades_pnl),
        wins=len(wins),
        losses=len(losses),
        total_pnl_pct=total_pnl,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        sharpe=sharpe,
        win_rate=win_rate,
    )


def objective(
    trial: "optuna.Trial",
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    horizon_name: str,
) -> float:
    """
    Целевая функция для Optuna.

    Оптимизирует комбинацию гиперпараметров модели + take/stop уровней
    для максимизации profit factor.
    """
    # Гиперпараметры модели
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 500)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 100)
    l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)

    # Параметры торговли
    p_threshold = trial.suggest_float("p_threshold", 0.45, 0.65)
    take_atr = trial.suggest_float("take_atr", 0.3, 1.5)
    stop_atr = trial.suggest_float("stop_atr", 0.2, 1.0)

    # Time series split для валидации
    tscv = TimeSeriesSplit(n_splits=3)
    results = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        prices_val = prices[val_idx]
        atr_val = atr[val_idx]

        # Обучение модели
        model = HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=42,
        )

        try:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
        except Exception:
            return -1000  # Penalty for failed training

        # Симуляция сделок
        result = simulate_trades(
            y_val, y_proba, prices_val, atr_val,
            p_threshold=p_threshold,
            take_atr=take_atr,
            stop_atr=stop_atr,
        )

        if result.total_trades >= 10:
            results.append(result)

    if not results:
        return -1000

    # Усреднённые метрики по фолдам
    avg_profit_factor = np.mean([r.profit_factor for r in results])
    avg_sharpe = np.mean([r.sharpe for r in results])
    avg_win_rate = np.mean([r.win_rate for r in results])
    avg_trades = np.mean([r.total_trades for r in results])

    # Комбинированная метрика: profit_factor + sharpe + штраф за малое число сделок
    score = avg_profit_factor * 0.5 + avg_sharpe * 0.3 + (avg_win_rate / 100) * 0.2

    # Штраф за слишком мало сделок
    if avg_trades < 50:
        score *= (avg_trades / 50)

    return score


def optimize_horizon(
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    horizon_name: str,
    n_trials: int = 100,
) -> Tuple[Dict, "optuna.Study"]:
    """
    Оптимизация параметров для одного горизонта.

    Returns:
        Tuple[best_params, study]
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for optimization. Install: pip install optuna")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"moex_agent_{horizon_name}",
    )

    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda trial: objective(trial, X, y, prices, atr, horizon_name),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study.best_params, study


def train_with_params(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
) -> object:
    """
    Обучение модели с заданными параметрами + калибровка.
    """
    model_params = {
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.07),
        "max_iter": params.get("max_iter", 250),
        "min_samples_leaf": params.get("min_samples_leaf", 20),
        "l2_regularization": params.get("l2_regularization", 0.0),
        "random_state": 42,
    }

    base = HistGradientBoostingClassifier(**model_params)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
    clf.fit(X, y)

    return clf


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загрузка и подготовка данных."""
    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    logger.info("Загрузка данных...")
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval=1
    ORDER BY secid, ts
    """
    candles = pd.read_sql_query(q, conn)
    logger.info(f"Загружено {len(candles):,} свечей")

    logger.info("Построение признаков...")
    feats = build_feature_frame(candles)

    horizons = [(h.name, h.minutes) for h in cfg.horizons]
    logger.info("Создание меток...")
    labels = make_time_exit_labels(candles, horizons=horizons)

    df = feats.merge(labels, on=["secid", "ts"], how="inner")
    df = df.dropna(subset=FEATURE_COLS)

    conn.close()

    return df, candles, cfg


def main():
    parser = argparse.ArgumentParser(description="Оптимизация моделей MOEX Agent")
    parser.add_argument("--trials", type=int, default=100, help="Число trials для Optuna")
    parser.add_argument("--horizons", nargs="+", default=None, help="Горизонты для оптимизации")
    parser.add_argument("--skip-optimize", action="store_true", help="Пропустить оптимизацию, использовать дефолты")
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE and not args.skip_optimize:
        logger.error("Optuna не установлена. Установите: pip install optuna")
        logger.info("Или запустите с --skip-optimize для использования улучшенных дефолтных параметров")
        return

    df, candles, cfg = load_data()

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    horizons = [(h.name, h.minutes) for h in cfg.horizons]
    if args.horizons:
        horizons = [(h, m) for h, m in horizons if h in args.horizons]

    meta: Dict[str, Dict] = {}
    optimal_params: Dict[str, Dict] = {}

    for name, minutes in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Обработка горизонта: {name}")
        logger.info(f"{'='*60}")

        ycol = f"y_time_{name}"
        if ycol not in df.columns:
            logger.warning(f"Метка {ycol} не найдена, пропускаем")
            continue

        # Подготовка данных
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=int)
        prices = df["close"].to_numpy(dtype=float) if "close" in df.columns else np.ones(len(df))
        atr = df["atr_14"].to_numpy(dtype=float) if "atr_14" in df.columns else np.ones(len(df)) * 0.01

        logger.info(f"Данных: {len(X):,} строк, позитивных: {y.sum():,} ({y.mean()*100:.1f}%)")

        if args.skip_optimize:
            # Улучшенные дефолтные параметры
            best_params = {
                "max_depth": 7,
                "learning_rate": 0.05,
                "max_iter": 300,
                "min_samples_leaf": 30,
                "l2_regularization": 0.1,
                "p_threshold": 0.52,
                "take_atr": 0.7,
                "stop_atr": 0.4,
            }
            logger.info("Используем улучшенные дефолтные параметры")
        else:
            # Оптимизация
            logger.info(f"Запуск оптимизации ({args.trials} trials)...")
            best_params, study = optimize_horizon(X, y, prices, atr, name, n_trials=args.trials)

            logger.info(f"Лучший результат: {study.best_value:.4f}")
            logger.info(f"Лучшие параметры: {best_params}")

        optimal_params[name] = best_params

        # Обучение финальной модели
        logger.info("Обучение финальной модели...")
        clf = train_with_params(X, y, best_params)

        # Оценка на тестовой выборке
        y_proba = clf.predict_proba(X)[:, 1]
        result = simulate_trades(
            y, y_proba, prices, atr,
            p_threshold=best_params.get("p_threshold", 0.5),
            take_atr=best_params.get("take_atr", 0.5),
            stop_atr=best_params.get("stop_atr", 0.3),
        )

        logger.info(f"Результаты на всех данных:")
        logger.info(f"  Сделок: {result.total_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.1f}%")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"  Sharpe: {result.sharpe:.2f}")
        logger.info(f"  Total PnL: {result.total_pnl_pct:+.2f}%")

        # Сохранение модели
        path = models_dir / f"model_time_{name}.joblib"
        joblib.dump(clf, path)

        meta[name] = {
            "type": "time-exit-optimized",
            "path": str(path),
            "features": FEATURE_COLS,
            "params": best_params,
            "metrics": {
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "sharpe": result.sharpe,
            }
        }

        logger.info(f"Модель сохранена: {path}")

    # Сохранение метаданных
    (models_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Сохранение оптимальных параметров
    (models_dir / "optimal_params.json").write_text(
        json.dumps(optimal_params, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Рекомендации для config.yaml
    logger.info("\n" + "="*60)
    logger.info("РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ для config.yaml:")
    logger.info("="*60)

    # Усреднённые оптимальные параметры
    avg_p_threshold = np.mean([p.get("p_threshold", 0.5) for p in optimal_params.values()])
    avg_take_atr = np.mean([p.get("take_atr", 0.5) for p in optimal_params.values()])
    avg_stop_atr = np.mean([p.get("stop_atr", 0.3) for p in optimal_params.values()])

    logger.info(f"""
signals:
  p_threshold: {avg_p_threshold:.2f}
  price_exit:
    enabled: true
    take_atr: {avg_take_atr:.2f}
    stop_atr: {avg_stop_atr:.2f}
""")

    logger.info("Оптимизация завершена!")


if __name__ == "__main__":
    main()

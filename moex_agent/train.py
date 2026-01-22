from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

from .config import load_config
from .features import build_feature_frame
from .labels import make_time_exit_labels
from .storage import connect


# Import feature columns from predictor for consistency
from .predictor import FEATURE_COLS


def load_candles(conn) -> pd.DataFrame:
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval=1
    ORDER BY secid, ts
    """
    return pd.read_sql_query(q, conn)


def main() -> None:
    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    candles = load_candles(conn)
    feats = build_feature_frame(candles)

    horizons = [(h.name, h.minutes) for h in cfg.horizons]
    labels = make_time_exit_labels(candles, horizons=horizons)

    df = feats.merge(labels, on=["secid","ts"], how="inner")
    df = df.dropna(subset=FEATURE_COLS)

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    meta: Dict[str, Dict] = {}

    # Simple per-horizon model
    for name, _minutes in horizons:
        ycol = f"y_time_{name}"
        if ycol not in df.columns:
            continue

        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=int)

        # time-series CV for calibration
        base = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.07, max_iter=250)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
        clf.fit(X, y)

        path = models_dir / f"model_time_{name}.joblib"
        joblib.dump(clf, path)
        meta[name] = {"type": "time-exit", "path": str(path), "features": FEATURE_COLS}
        print(f"Saved: {path}")

    (models_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()

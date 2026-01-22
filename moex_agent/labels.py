from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def make_time_exit_labels(candles_1m: pd.DataFrame, horizons: List[Tuple[str, int]], fee_bps: float = 8.0) -> pd.DataFrame:
    """Label = 1 if return after H minutes > 0 after a fee slippage proxy.

    fee_bps is an approximate round-trip cost in bps.
    """
    df = candles_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["secid", "ts"])

    labels = []
    cost = fee_bps / 10000.0

    for secid, g in df.groupby("secid", sort=False):
        g = g.set_index("ts")
        close = g["close"].astype(float)
        y = pd.DataFrame(index=g.index)
        y["secid"] = secid
        for name, minutes in horizons:
            fut = close.shift(-minutes)
            ret = (fut / close) - 1.0
            ret_net = ret - cost
            y[f"y_time_{name}"] = (ret_net > 0).astype(int)
        labels.append(y.reset_index().rename(columns={"index": "ts"}))

    return pd.concat(labels, ignore_index=True)

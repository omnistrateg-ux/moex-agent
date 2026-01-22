from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from .config import load_config
from .moex_iss import fetch_candles
from .storage import connect, upsert_many


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=240)
    args = ap.parse_args()

    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=args.days)).isoformat()
    till_date = today.isoformat()

    rows_total = 0
    for secid in cfg.tickers:
        candles = fetch_candles(cfg.engine, cfg.market, cfg.board, secid, interval=1, from_date=from_date, till_date=till_date)
        rows = [(secid, cfg.board, 1, c.ts, c.open, c.high, c.low, c.close, c.value, c.volume) for c in candles]
        rows_total += upsert_many(
            conn,
            table="candles",
            columns=("secid","board","interval","ts","open","high","low","close","value","volume"),
            rows=rows,
        )
        print(f"{secid}: {len(rows)} candles")

    print(f"Upserted rows: {rows_total}")


if __name__ == "__main__":
    main()

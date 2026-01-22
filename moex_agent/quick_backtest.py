"""
Quick Backtester - Fast validation of trading strategy.

Uses 15-minute intervals and strict signal filtering.
Focuses on high-quality signals only (STRONG_BUY/STRONG_SELL).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .anomaly import compute_anomalies, Direction
from .config_schema import load_config
from .features import build_feature_frame
from .predictor import ModelRegistry, FEATURE_COLS
from .qwen import _rule_based_analysis, _get_ticker_liquidity
from .risk import RiskParams, pass_gatekeeper
from .storage import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.quick_backtest")


@dataclass
class Trade:
    ticker: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_pct: float
    horizon: str
    probability: float
    recommendation: str
    exit_reason: str


def run_quick_backtest(
    start_date: str,
    end_date: str,
    config_path: str = "config.yaml",
    signal_interval_minutes: int = 15,  # Check for signals every 15 min
    max_daily_trades: int = 5,  # Max trades per day
    only_strong_signals: bool = True,  # Only STRONG_BUY/STRONG_SELL
) -> List[Trade]:
    """
    Run a quick backtest with strict filtering.

    Returns list of trades.
    """
    config = load_config(config_path)
    conn = connect(config.sqlite_path)

    models = ModelRegistry()
    models.load()

    risk_params = RiskParams(
        max_spread_bps=config.risk.max_spread_bps,
        min_turnover_rub_5m=config.risk.min_turnover_rub_5m,
    )

    # Load candles
    logger.info(f"Loading candles {start_date} to {end_date}...")
    query = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = 1 AND ts >= ? AND ts <= ?
    ORDER BY ts, secid
    """
    all_candles = pd.read_sql_query(
        query, conn,
        params=(f"{start_date} 00:00:00", f"{end_date} 23:59:59")
    )
    all_candles["ts"] = pd.to_datetime(all_candles["ts"], utc=True)
    logger.info(f"Loaded {len(all_candles):,} candles")

    # Build features
    logger.info("Building features...")
    features_df = build_feature_frame(all_candles)
    features_df = features_df.dropna()

    # Get unique timestamps at signal_interval
    timestamps = sorted(all_candles["ts"].unique())
    signal_timestamps = [ts for i, ts in enumerate(timestamps) if i >= 200 and i % signal_interval_minutes == 0]
    logger.info(f"Processing {len(signal_timestamps)} signal timestamps...")

    trades: List[Trade] = []
    positions: Dict[str, dict] = {}  # ticker -> position
    cooldown_map: Dict[str, datetime] = {}
    daily_trades: Dict[str, int] = {}  # date -> count

    for ts_idx, ts in enumerate(signal_timestamps):
        current_date = ts.date().isoformat()

        # Limit daily trades
        if daily_trades.get(current_date, 0) >= max_daily_trades:
            continue

        # Get recent candles for this timestamp
        recent_candles = all_candles[all_candles["ts"] <= ts].groupby("secid").tail(2000).reset_index(drop=True)
        current_row = all_candles[all_candles["ts"] == ts]

        # Check exits for positions
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            ticker_row = current_row[current_row["secid"] == ticker]

            if ticker_row.empty:
                continue

            high = ticker_row.iloc[0]["high"]
            low = ticker_row.iloc[0]["low"]
            close = ticker_row.iloc[0]["close"]

            exit_reason = None
            exit_price = None

            # Check take profit
            if pos["take"] is not None:
                if pos["direction"] == "LONG" and high >= pos["take"]:
                    exit_reason = "take"
                    exit_price = pos["take"]
                elif pos["direction"] == "SHORT" and low <= pos["take"]:
                    exit_reason = "take"
                    exit_price = pos["take"]

            # Check stop loss
            if exit_reason is None and pos["stop"] is not None:
                if pos["direction"] == "LONG" and low <= pos["stop"]:
                    exit_reason = "stop"
                    exit_price = pos["stop"]
                elif pos["direction"] == "SHORT" and high >= pos["stop"]:
                    exit_reason = "stop"
                    exit_price = pos["stop"]

            # Check timeout
            if exit_reason is None:
                elapsed = (ts - pos["entry_time"]).total_seconds() / 60
                if elapsed >= pos["ttl"]:
                    exit_reason = "timeout"
                    exit_price = close

            if exit_reason:
                if pos["direction"] == "LONG":
                    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
                else:
                    pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * 100

                trades.append(Trade(
                    ticker=ticker,
                    direction=pos["direction"],
                    entry_time=pos["entry_time"],
                    exit_time=ts,
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    horizon=pos["horizon"],
                    probability=pos["probability"],
                    recommendation=pos["recommendation"],
                    exit_reason=exit_reason,
                ))
                del positions[ticker]

        # Build quotes
        quotes = {}
        for _, row in current_row.iterrows():
            quotes[row["secid"]] = {
                "secid": row["secid"],
                "last": row["close"],
                "bid": row["close"] * 0.9999,
                "ask": row["close"] * 1.0001,
            }

        # Detect anomalies
        anomalies = compute_anomalies(
            candles_1m=recent_candles[["secid", "ts", "close", "value", "volume"]],
            quotes=quotes,
            min_turnover_rub_5m=risk_params.min_turnover_rub_5m,
            max_spread_bps=risk_params.max_spread_bps,
            top_n=config.top_n_anomalies,
        )

        # Get latest features
        current_features = features_df[features_df["ts"] <= ts]
        latest_features = current_features.groupby("secid").tail(1)

        # Process anomalies
        for anomaly in anomalies:
            ticker = anomaly.secid

            # Skip if already have position
            if ticker in positions:
                continue

            # Check cooldown
            if ticker in cooldown_map and ts < cooldown_map[ticker]:
                continue

            # Get features
            row = latest_features[latest_features["secid"] == ticker]
            if row.empty:
                continue

            try:
                X = row[FEATURE_COLS].to_numpy(dtype=float)
            except KeyError:
                continue

            # Predict
            best_h, best_p = models.best_horizon(X)
            if best_h is None:
                continue

            # Risk gatekeeper
            if not pass_gatekeeper(
                p=best_p,
                p_threshold=config.p_threshold,
                turnover_5m=anomaly.turnover_5m,
                spread=anomaly.spread_bps,
                risk=risk_params,
            ):
                continue

            # Apply Qwen rules
            payload = {
                "ticker": ticker,
                "direction": anomaly.direction.value,
                "horizon": best_h,
                "p": best_p,
                "anomaly": {
                    "z_ret_5m": anomaly.z_ret_5m,
                    "z_vol_5m": anomaly.z_vol_5m,
                    "spread_bps": anomaly.spread_bps or 0,
                    "volume_spike": anomaly.volume_spike,
                },
                "market_context": {
                    "is_opening": False,
                    "ticker_liquidity": _get_ticker_liquidity(ticker),
                },
            }

            analysis = _rule_based_analysis(payload)
            if analysis.skip:
                continue

            # Filter by recommendation
            if only_strong_signals:
                if analysis.recommendation not in ["STRONG_BUY", "STRONG_SELL"]:
                    continue
            else:
                if analysis.recommendation not in ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]:
                    continue

            # Calculate price targets
            if ticker not in quotes:
                continue

            last_price = quotes[ticker]["last"]
            atr = float(row["atr_14"].iloc[0]) if "atr_14" in row.columns else None

            take = None
            stop = None
            if atr and atr > 0:
                take_atr = config.signals.price_exit.take_atr
                stop_atr = config.signals.price_exit.stop_atr

                if anomaly.direction == Direction.LONG:
                    take = last_price + take_atr * atr
                    stop = last_price - stop_atr * atr
                else:
                    take = last_price - take_atr * atr
                    stop = last_price + stop_atr * atr

            ttl = next((h.minutes for h in config.horizons if h.name == best_h), 60)

            # Open position
            positions[ticker] = {
                "entry_time": ts,
                "entry_price": last_price,
                "direction": anomaly.direction.value,
                "take": take,
                "stop": stop,
                "ttl": ttl,
                "horizon": best_h,
                "probability": best_p,
                "recommendation": analysis.recommendation,
            }

            cooldown_map[ticker] = ts + timedelta(minutes=config.cooldown_minutes)
            daily_trades[current_date] = daily_trades.get(current_date, 0) + 1

            logger.debug(f"OPEN: {ticker} {anomaly.direction.value} @ {last_price:.2f} "
                        f"H={best_h} p={best_p:.2f} {analysis.recommendation}")

        # Progress
        if ts_idx % 100 == 0:
            logger.info(f"Processed {ts_idx}/{len(signal_timestamps)}, {len(trades)} trades completed")

    conn.close()
    return trades


def analyze_results(trades: List[Trade]) -> None:
    """Print analysis of backtest results."""
    if not trades:
        print("No trades executed.")
        return

    print("\n" + "=" * 60)
    print("QUICK BACKTEST RESULTS")
    print("=" * 60)

    total = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    win_rate = len(wins) / total * 100
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    total_pnl = sum(t.pnl_pct for t in trades)

    print(f"\nTOTAL TRADES: {total}")
    print(f"  Wins: {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses: {len(losses)}")
    print(f"\nAVERAGE P&L:")
    print(f"  Avg Win: {avg_win:+.2f}%")
    print(f"  Avg Loss: {avg_loss:+.2f}%")
    print(f"  Total: {total_pnl:+.2f}%")

    # By direction
    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    long_wins = len([t for t in longs if t.pnl_pct > 0])
    short_wins = len([t for t in shorts if t.pnl_pct > 0])

    print(f"\nBY DIRECTION:")
    print(f"  LONG: {len(longs)} trades, {long_wins/len(longs)*100 if longs else 0:.1f}% win rate")
    print(f"  SHORT: {len(shorts)} trades, {short_wins/len(shorts)*100 if shorts else 0:.1f}% win rate")

    # By exit reason
    takes = [t for t in trades if t.exit_reason == "take"]
    stops = [t for t in trades if t.exit_reason == "stop"]
    timeouts = [t for t in trades if t.exit_reason == "timeout"]

    print(f"\nBY EXIT REASON:")
    print(f"  Take Profit: {len(takes)} ({np.mean([t.pnl_pct for t in takes]):+.2f}% avg)" if takes else "  Take Profit: 0")
    print(f"  Stop Loss: {len(stops)} ({np.mean([t.pnl_pct for t in stops]):+.2f}% avg)" if stops else "  Stop Loss: 0")
    print(f"  Timeout: {len(timeouts)} ({np.mean([t.pnl_pct for t in timeouts]):+.2f}% avg)" if timeouts else "  Timeout: 0")

    # By horizon
    print(f"\nBY HORIZON:")
    for h in ["5m", "10m", "30m", "1h", "1d", "1w"]:
        h_trades = [t for t in trades if t.horizon == h]
        if h_trades:
            h_wins = len([t for t in h_trades if t.pnl_pct > 0])
            h_pnl = sum(t.pnl_pct for t in h_trades)
            print(f"  {h}: {len(h_trades)} trades, {h_wins/len(h_trades)*100:.1f}% WR, {h_pnl:+.2f}% total")

    # Top 5 best and worst trades
    sorted_trades = sorted(trades, key=lambda t: t.pnl_pct, reverse=True)
    print(f"\nTOP 5 BEST TRADES:")
    for t in sorted_trades[:5]:
        print(f"  {t.ticker} {t.direction} {t.horizon}: {t.pnl_pct:+.2f}% ({t.exit_reason})")

    print(f"\nTOP 5 WORST TRADES:")
    for t in sorted_trades[-5:]:
        print(f"  {t.ticker} {t.direction} {t.horizon}: {t.pnl_pct:+.2f}% ({t.exit_reason})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Quick Backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=15, help="Signal check interval (minutes)")
    parser.add_argument("--max-daily", type=int, default=5, help="Max daily trades")
    parser.add_argument("--all-signals", action="store_true", help="Include BUY/SELL (not just STRONG)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")

    args = parser.parse_args()

    trades = run_quick_backtest(
        start_date=args.start,
        end_date=args.end,
        config_path=args.config,
        signal_interval_minutes=args.interval,
        max_daily_trades=args.max_daily,
        only_strong_signals=not args.all_signals,
    )

    analyze_results(trades)


if __name__ == "__main__":
    main()

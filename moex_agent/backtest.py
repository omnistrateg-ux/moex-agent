"""
MOEX Agent Backtester / Simulator

Simulates trading on historical data to evaluate strategy performance.
Uses the same signal generation pipeline as live trading.

Usage:
    python -m moex_agent.backtest --start 2025-06-01 --end 2025-12-31 --capital 1000000

Features:
    - Virtual trading with configurable initial capital
    - Realistic execution with spread simulation
    - Position sizing based on risk
    - Comprehensive performance metrics
    - Trade log export
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .anomaly import compute_anomalies, Direction
from .config_schema import load_config
from .features import build_feature_frame
from .predictor import ModelRegistry, FEATURE_COLS
from .qwen import _rule_based_analysis  # Use same rules as live trading
from .risk import RiskParams, pass_gatekeeper
from .storage import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.backtest")


@dataclass
class Trade:
    """Single completed trade."""
    ticker: str
    direction: str  # LONG or SHORT
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: int  # number of shares
    pnl: float  # profit/loss in RUB
    pnl_pct: float  # profit/loss percentage
    horizon: str
    probability: float
    exit_reason: str  # 'take', 'stop', 'timeout'


@dataclass
class Position:
    """Open position."""
    ticker: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    size: int
    take: Optional[float]
    stop: Optional[float]
    ttl_minutes: int
    horizon: str
    probability: float


@dataclass
class BacktestResult:
    """Backtesting results."""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    avg_win: float
    avg_loss: float
    profit_factor: float

    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    trades: List[Trade]
    equity_curve: List[Tuple[datetime, float]]

    # By direction
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float

    # By horizon
    horizon_stats: Dict[str, Dict[str, float]]


class Backtester:
    """
    Strategy backtester using historical data.

    Simulates the full signal generation and trade execution pipeline.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        initial_capital: float = 1_000_000,
        position_size_pct: float = 0.05,  # 5% of capital per trade
        spread_bps: float = 10,  # simulated spread
        commission_pct: float = 0.0003,  # 0.03% commission
    ):
        self.config = load_config(config_path)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.spread_bps = spread_bps
        self.commission_pct = commission_pct

        self.models = ModelRegistry()
        self.models.load()

        self.risk_params = RiskParams(
            max_spread_bps=self.config.risk.max_spread_bps,
            min_turnover_rub_5m=self.config.risk.min_turnover_rub_5m,
        )

        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.cooldown_map: Dict[str, datetime] = {}

    def _apply_spread(self, price: float, direction: Direction, is_entry: bool) -> float:
        """Apply simulated spread to price."""
        spread_mult = self.spread_bps / 10000

        if direction == Direction.LONG:
            if is_entry:
                return price * (1 + spread_mult / 2)  # Buy at ask
            else:
                return price * (1 - spread_mult / 2)  # Sell at bid
        else:  # SHORT
            if is_entry:
                return price * (1 - spread_mult / 2)  # Sell at bid
            else:
                return price * (1 + spread_mult / 2)  # Buy at ask

    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size in shares."""
        position_value = self.capital * self.position_size_pct
        shares = int(position_value / price)
        return max(1, shares)

    def _open_position(
        self,
        ticker: str,
        direction: Direction,
        price: float,
        timestamp: datetime,
        take: Optional[float],
        stop: Optional[float],
        ttl_minutes: int,
        horizon: str,
        probability: float,
    ) -> None:
        """Open a new position."""
        if ticker in self.positions:
            return  # Already have position

        entry_price = self._apply_spread(price, direction, is_entry=True)
        size = self._calculate_position_size(entry_price)

        # Deduct commission
        commission = entry_price * size * self.commission_pct
        self.capital -= commission

        self.positions[ticker] = Position(
            ticker=ticker,
            direction=direction,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            take=take,
            stop=stop,
            ttl_minutes=ttl_minutes,
            horizon=horizon,
            probability=probability,
        )

        logger.debug(f"OPEN: {ticker} {direction.value} @ {entry_price:.2f} x{size}")

    def _close_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Close an existing position."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        exit_price = self._apply_spread(price, pos.direction, is_entry=False)

        # Calculate P&L
        if pos.direction == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * pos.size
        else:  # SHORT
            pnl = (pos.entry_price - exit_price) * pos.size

        # Deduct commission
        commission = exit_price * pos.size * self.commission_pct
        pnl -= commission

        pnl_pct = pnl / (pos.entry_price * pos.size) * 100

        self.capital += pnl

        trade = Trade(
            ticker=ticker,
            direction=pos.direction.value,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            horizon=pos.horizon,
            probability=pos.probability,
            exit_reason=reason,
        )
        self.trades.append(trade)

        del self.positions[ticker]

        logger.debug(f"CLOSE: {ticker} @ {exit_price:.2f} PnL={pnl:+.2f} ({pnl_pct:+.2f}%) [{reason}]")

    def _check_exits(self, ticker: str, high: float, low: float, close: float, timestamp: datetime) -> None:
        """Check if any position needs to be closed."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]

        # Check take profit
        if pos.take is not None:
            if pos.direction == Direction.LONG and high >= pos.take:
                self._close_position(ticker, pos.take, timestamp, "take")
                return
            elif pos.direction == Direction.SHORT and low <= pos.take:
                self._close_position(ticker, pos.take, timestamp, "take")
                return

        # Check stop loss
        if pos.stop is not None:
            if pos.direction == Direction.LONG and low <= pos.stop:
                self._close_position(ticker, pos.stop, timestamp, "stop")
                return
            elif pos.direction == Direction.SHORT and high >= pos.stop:
                self._close_position(ticker, pos.stop, timestamp, "stop")
                return

        # Check timeout
        elapsed = (timestamp - pos.entry_time).total_seconds() / 60
        if elapsed >= pos.ttl_minutes:
            self._close_position(ticker, close, timestamp, "timeout")

    def run(
        self,
        start_date: str,
        end_date: str,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            verbose: Print progress

        Returns:
            BacktestResult with performance metrics
        """
        conn = connect(self.config.sqlite_path)

        # Load all candles for the period
        query = """
        SELECT secid, ts, open, high, low, close, value, volume
        FROM candles
        WHERE interval = 1
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts, secid
        """

        if verbose:
            logger.info(f"Loading candles from {start_date} to {end_date}...")

        all_candles = pd.read_sql_query(
            query, conn,
            params=(f"{start_date} 00:00:00", f"{end_date} 23:59:59")
        )
        all_candles["ts"] = pd.to_datetime(all_candles["ts"], utc=True)

        if all_candles.empty:
            raise ValueError(f"No candles found for period {start_date} to {end_date}")

        if verbose:
            logger.info(f"Loaded {len(all_candles):,} candles")

        # Build features for the entire period
        if verbose:
            logger.info("Building features...")
        features_df = build_feature_frame(all_candles)
        features_df = features_df.dropna()

        # Get unique timestamps
        timestamps = sorted(all_candles["ts"].unique())

        if verbose:
            logger.info(f"Processing {len(timestamps):,} timestamps...")

        # Pre-index data for O(1) access instead of O(n) filtering
        # Build index: ts -> rows with that timestamp
        ts_to_idx = all_candles.groupby("ts").indices
        feat_ts_to_idx = features_df.groupby("ts").indices

        # Pre-compute last N candles per ticker for each timestamp
        # Using rolling window approach for efficiency
        ticker_history = {secid: [] for secid in all_candles["secid"].unique()}

        # Process each timestamp
        processed = 0
        signal_count = 0

        for i, ts in enumerate(timestamps):
            # Skip first 200 timestamps for warmup
            if i < 200:
                continue

            # Get current rows using pre-built index (O(1) instead of O(n))
            current_idx = ts_to_idx.get(ts, [])
            current_row = all_candles.iloc[current_idx] if len(current_idx) > 0 else pd.DataFrame()

            # Check exits for all positions
            for _, row in current_row.iterrows():
                self._check_exits(
                    row["secid"],
                    row["high"],
                    row["low"],
                    row["close"],
                    ts,
                )

            # Only check for new signals every 5 minutes
            if i % 5 != 0:
                continue

            # Get recent candles efficiently using window slicing
            # Take last 2000 timestamps worth of candles
            start_idx = max(0, i - 2000)
            recent_timestamps = timestamps[start_idx:i+1]
            recent_idx = []
            for rts in recent_timestamps:
                recent_idx.extend(ts_to_idx.get(rts, []))
            recent_candles = all_candles.iloc[recent_idx] if recent_idx else pd.DataFrame()

            # Mock quotes (use current close as last price)
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
                min_turnover_rub_5m=self.risk_params.min_turnover_rub_5m,
                max_spread_bps=self.risk_params.max_spread_bps,
                top_n=self.config.top_n_anomalies,
            )

            # Get latest feature row per secid using pre-built index
            feat_idx = feat_ts_to_idx.get(ts, [])
            latest_features = features_df.iloc[feat_idx] if len(feat_idx) > 0 else pd.DataFrame()

            # Generate signals for anomalies
            for anomaly in anomalies:
                ticker = anomaly.secid

                # Skip if already have position or in cooldown
                if ticker in self.positions:
                    continue

                cooldown_end = self.cooldown_map.get(ticker, datetime.min.replace(tzinfo=timezone.utc))
                if ts < cooldown_end:
                    continue

                # Get feature row
                row = latest_features[latest_features["secid"] == ticker]
                if row.empty:
                    continue

                # Get features
                try:
                    X = row[FEATURE_COLS].to_numpy(dtype=float)
                except KeyError:
                    continue

                # Find best horizon
                best_h, best_p = self.models.best_horizon(X)
                if best_h is None:
                    continue

                # Risk gatekeeper
                if not pass_gatekeeper(
                    p=best_p,
                    p_threshold=self.config.p_threshold,
                    turnover_5m=anomaly.turnover_5m,
                    spread=anomaly.spread_bps,
                    risk=self.risk_params,
                ):
                    continue

                # Calculate price targets
                if ticker not in quotes:
                    continue
                last_price = quotes[ticker]["last"]
                atr = float(row["atr_14"].iloc[0]) if "atr_14" in row.columns else None

                take = None
                stop = None
                if atr and atr > 0:
                    take_atr = self.config.signals.price_exit.take_atr
                    stop_atr = self.config.signals.price_exit.stop_atr

                    if anomaly.direction == Direction.LONG:
                        take = last_price + take_atr * atr
                        stop = last_price - stop_atr * atr
                    else:
                        take = last_price - take_atr * atr
                        stop = last_price + stop_atr * atr

                # Get TTL
                ttl = next(
                    (h.minutes for h in self.config.horizons if h.name == best_h),
                    60,
                )

                # Apply Qwen rule-based filtering (same as live trading)
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
                        "is_opening": False,  # Simplified for backtest
                        "ticker_liquidity": "MEDIUM",
                    },
                }

                analysis = _rule_based_analysis(payload)
                if analysis.skip:
                    continue  # Skip signals that don't pass rules

                # Only trade STRONG_BUY/BUY and STRONG_SELL/SELL
                if analysis.recommendation not in ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]:
                    continue

                # Open position
                self._open_position(
                    ticker=ticker,
                    direction=anomaly.direction,
                    price=last_price,
                    timestamp=ts,
                    take=take,
                    stop=stop,
                    ttl_minutes=ttl,
                    horizon=best_h,
                    probability=best_p,
                )

                signal_count += 1
                self.cooldown_map[ticker] = ts + timedelta(minutes=self.config.cooldown_minutes)

            # Record equity
            self.equity_curve.append((ts, self.capital))

            processed += 1
            if verbose and processed % 1000 == 0:
                logger.info(f"Processed {processed:,} timestamps, {len(self.trades)} trades, capital: {self.capital:,.0f}")

        # Close any remaining positions at end
        final_row = all_candles[all_candles["ts"] == timestamps[-1]]
        for ticker in list(self.positions.keys()):
            ticker_row = final_row[final_row["secid"] == ticker]
            if not ticker_row.empty:
                self._close_position(ticker, ticker_row.iloc[0]["close"], timestamps[-1], "end")

        conn.close()

        # Calculate metrics
        return self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        max_dd_pct = 0
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        # Sharpe ratio (simplified, daily returns)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i][1] - self.equity_curve[i-1][1]) / self.equity_curve[i-1][1]
                returns.append(ret)
            returns = np.array(returns)
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # By direction
        long_trades = [t for t in self.trades if t.direction == "LONG"]
        short_trades = [t for t in self.trades if t.direction == "SHORT"]
        long_wins = len([t for t in long_trades if t.pnl > 0])
        short_wins = len([t for t in short_trades if t.pnl > 0])
        long_win_rate = long_wins / len(long_trades) if long_trades else 0
        short_win_rate = short_wins / len(short_trades) if short_trades else 0

        # By horizon
        horizon_stats = {}
        for h in ["5m", "10m", "30m", "1h", "1d", "1w"]:
            h_trades = [t for t in self.trades if t.horizon == h]
            if h_trades:
                h_wins = len([t for t in h_trades if t.pnl > 0])
                horizon_stats[h] = {
                    "trades": len(h_trades),
                    "win_rate": h_wins / len(h_trades),
                    "avg_pnl": np.mean([t.pnl for t in h_trades]),
                    "total_pnl": sum([t.pnl for t in h_trades]),
                }

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            trades=self.trades,
            equity_curve=self.equity_curve,
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            horizon_stats=horizon_stats,
        )


def print_report(result: BacktestResult) -> None:
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)

    print(f"\n{'PORTFOLIO PERFORMANCE':=^60}")
    print(f"  Initial Capital:    {result.initial_capital:>15,.0f} RUB")
    print(f"  Final Capital:      {result.final_capital:>15,.0f} RUB")
    print(f"  Total Return:       {result.total_return:>15,.0f} RUB ({result.total_return_pct:+.2f}%)")
    print(f"  Max Drawdown:       {result.max_drawdown:>15,.0f} RUB ({result.max_drawdown_pct:.2f}%)")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:>15.2f}")

    print(f"\n{'TRADE STATISTICS':=^60}")
    print(f"  Total Trades:       {result.total_trades:>15,d}")
    print(f"  Winning Trades:     {result.winning_trades:>15,d} ({result.win_rate*100:.1f}%)")
    print(f"  Losing Trades:      {result.losing_trades:>15,d}")
    print(f"  Avg Win:            {result.avg_win:>15,.0f} RUB")
    print(f"  Avg Loss:           {result.avg_loss:>15,.0f} RUB")
    print(f"  Profit Factor:      {result.profit_factor:>15.2f}")

    print(f"\n{'BY DIRECTION':=^60}")
    print(f"  LONG Trades:        {result.long_trades:>15,d} (Win rate: {result.long_win_rate*100:.1f}%)")
    print(f"  SHORT Trades:       {result.short_trades:>15,d} (Win rate: {result.short_win_rate*100:.1f}%)")

    print(f"\n{'BY HORIZON':=^60}")
    for h, stats in result.horizon_stats.items():
        print(f"  {h:>5}: {stats['trades']:>4} trades | "
              f"WR: {stats['win_rate']*100:>5.1f}% | "
              f"Avg: {stats['avg_pnl']:>+8,.0f} | "
              f"Total: {stats['total_pnl']:>+10,.0f}")

    # Exit reasons
    take_exits = len([t for t in result.trades if t.exit_reason == "take"])
    stop_exits = len([t for t in result.trades if t.exit_reason == "stop"])
    timeout_exits = len([t for t in result.trades if t.exit_reason == "timeout"])

    print(f"\n{'EXIT REASONS':=^60}")
    print(f"  Take Profit:        {take_exits:>15,d}")
    print(f"  Stop Loss:          {stop_exits:>15,d}")
    print(f"  Timeout:            {timeout_exits:>15,d}")

    print("\n" + "=" * 60)


def export_trades(result: BacktestResult, path: str) -> None:
    """Export trades to CSV."""
    df = pd.DataFrame([
        {
            "ticker": t.ticker,
            "direction": t.direction,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": t.size,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "horizon": t.horizon,
            "probability": t.probability,
            "exit_reason": t.exit_reason,
        }
        for t in result.trades
    ])
    df.to_csv(path, index=False)
    print(f"Trades exported to: {path}")


def main():
    parser = argparse.ArgumentParser(description="MOEX Agent Backtester")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital (RUB)")
    parser.add_argument("--position-size", type=float, default=0.05, help="Position size (fraction of capital)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--export", help="Export trades to CSV file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    backtester = Backtester(
        config_path=args.config,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
    )

    result = backtester.run(
        start_date=args.start,
        end_date=args.end,
        verbose=args.verbose,
    )

    print_report(result)

    if args.export:
        export_trades(result, args.export)


if __name__ == "__main__":
    main()

"""
MOEX Agent Margin Paper Trading

Paper trading with full margin risk controls.
Uses MarginRiskEngine for position sizing, leverage, and kill-switches.

Features:
- Dynamic leverage based on regime, volatility, confidence
- Kill-switch after consecutive losses
- Position sizing by risk (max 0.5% loss per trade)
- NO 1d/1w horizons (gap risk)
- Real-time risk monitoring

Usage:
    python -m moex_agent.margin_paper_trading --capital 200000 --max-leverage 3

Author: MOEX Agent Risk Team
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anomaly import compute_anomalies, Direction
from .config_schema import load_config
from .db_state import StateStorage, load_trading_state, save_trading_state
from .engine import PipelineEngine
from .margin_risk_engine import (
    MarginRiskEngine,
    KillSwitchConfig,
    LeverageConfig,
    RiskDecision,
    MarketRegime,
    stress_test_strategy,
)
from .moex_iss import close_session
from .qwen import _rule_based_analysis, _get_ticker_liquidity
from .storage import connect, get_window
from .telegram import send_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moex_agent.margin_paper")

STATE_FILE = Path("data/margin_paper_state.json")


@dataclass
class MarginPosition:
    """Leveraged position."""
    ticker: str
    direction: str
    entry_time: str
    entry_price: float
    size: int
    leverage: float
    take: Optional[float]
    stop: Optional[float]
    ttl_minutes: int
    horizon: str
    probability: float
    regime: str
    notional_value: float  # position_value * leverage


@dataclass
class MarginTrade:
    """Completed margin trade."""
    ticker: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: int
    leverage: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    horizon: str
    regime: str


@dataclass
class MarginAccount:
    """Margin trading account with risk tracking."""
    initial_capital: float
    cash: float
    margin_used: float = 0.0
    positions: Dict[str, MarginPosition] = field(default_factory=dict)
    closed_trades: List[MarginTrade] = field(default_factory=list)
    start_time: str = ""
    last_report_time: str = ""
    cooldown_map: Dict[str, str] = field(default_factory=dict)

    # Risk tracking
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    kill_switch_active: bool = False
    kill_switch_reason: str = ""

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now(timezone.utc).isoformat()
        if not self.last_report_time:
            self.last_report_time = self.start_time
        if self.peak_equity == 0:
            self.peak_equity = self.initial_capital

    @property
    def equity(self) -> float:
        """Current equity (cash + unrealized PnL)."""
        return self.cash + sum(
            pos.entry_price * pos.size for pos in self.positions.values()
        )

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "margin_used": self.margin_used,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades],
            "start_time": self.start_time,
            "last_report_time": self.last_report_time,
            "cooldown_map": self.cooldown_map,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "consecutive_losses": self.consecutive_losses,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarginAccount":
        """Deserialize from dict."""
        positions = {k: MarginPosition(**v) for k, v in data.get("positions", {}).items()}
        closed_trades = [MarginTrade(**t) for t in data.get("closed_trades", [])]
        return cls(
            initial_capital=data["initial_capital"],
            cash=data["cash"],
            margin_used=data.get("margin_used", 0.0),
            positions=positions,
            closed_trades=closed_trades,
            start_time=data.get("start_time", ""),
            last_report_time=data.get("last_report_time", ""),
            cooldown_map=data.get("cooldown_map", {}),
            peak_equity=data.get("peak_equity", data["initial_capital"]),
            daily_pnl=data.get("daily_pnl", 0.0),
            weekly_pnl=data.get("weekly_pnl", 0.0),
            consecutive_losses=data.get("consecutive_losses", 0),
            kill_switch_active=data.get("kill_switch_active", False),
            kill_switch_reason=data.get("kill_switch_reason", ""),
        )

    def save(self, path: Path = STATE_FILE) -> None:
        """Save state to database (PostgreSQL/SQLite) or file."""
        # Сначала пробуем PostgreSQL/SQLite
        try:
            success = save_trading_state(self.to_dict(), key="margin_paper")
            if success:
                logger.debug("State saved to database")
                return
        except Exception as e:
            logger.debug(f"Database save failed, falling back to file: {e}")

        # Fallback на файл
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> Optional["MarginAccount"]:
        """Load state from database (PostgreSQL/SQLite) or file."""
        # Сначала пробуем PostgreSQL/SQLite
        try:
            data = load_trading_state(key="margin_paper")
            if data and data.get("initial_capital"):
                logger.info(f"Loaded state from database: {len(data.get('closed_trades', []))} trades")
                return cls.from_dict(data)
        except Exception as e:
            logger.debug(f"Database load failed, trying file: {e}")

        # Fallback на файл
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return None


class MarginPaperTrader:
    """
    Margin paper trading with full risk controls.
    """

    # Disabled horizons for margin trading
    DISABLED_HORIZONS = {"1d", "1w"}

    def __init__(
        self,
        config_path: str = "config.yaml",
        initial_capital: float = 200_000,
        max_leverage: float = 3.0,
        max_positions: int = 3,
        resume: bool = True,
    ):
        self.config = load_config(config_path)
        self.max_positions = max_positions

        # Load or create account
        if resume:
            self.account = MarginAccount.load()
        else:
            self.account = None

        if self.account is None:
            self.account = MarginAccount(
                initial_capital=initial_capital,
                cash=initial_capital,
            )
            logger.info(f"Created new margin account: {initial_capital:,.0f} RUB")
        else:
            logger.info(f"Resumed margin account: equity={self.account.equity:,.0f}")

        # Initialize risk engine
        self.risk_engine = MarginRiskEngine(
            initial_equity=self.account.equity,
            kill_switch_config=KillSwitchConfig(
                max_loss_per_trade_pct=0.5,
                max_daily_loss_pct=2.0,
                max_weekly_loss_pct=5.0,
                max_consecutive_losses=2,  # HALT_DAY after 2 losses
                max_drawdown_pct=10.0,
            ),
            leverage_config=LeverageConfig(
                base_leverage=1.0,
                max_leverage=max_leverage,
                min_confidence_to_trade=0.54,
            ),
        )

        # Sync risk engine state from account
        self.risk_engine.state.equity = self.account.equity
        self.risk_engine.state.peak_equity = self.account.peak_equity
        self.risk_engine.state.consecutive_losses = self.account.consecutive_losses
        self.risk_engine.state.daily_pnl = self.account.daily_pnl
        self.risk_engine.state.weekly_pnl = self.account.weekly_pnl

        # Initialize pipeline engine
        self.engine = PipelineEngine(self.config)
        self.engine.load_models()

        self.conn = connect(self.config.sqlite_path)

        # Run initial stress test
        self._run_stress_test()

    def _run_stress_test(self) -> None:
        """Run stress test on startup."""
        logger.info("Running stress tests...")
        results = stress_test_strategy(self.risk_engine)

        for scenario in results["scenarios"]:
            logger.info(
                f"  {scenario['name']}: "
                f"MaxDD={scenario['total_drawdown_pct']:.1f}%, "
                f"Kill={scenario['kill_switch_triggered']}"
            )

        # Reset after stress test
        self.risk_engine.state.equity = self.account.equity
        self.risk_engine.state.consecutive_losses = self.account.consecutive_losses
        self.risk_engine.state.kill_switch_active = self.account.kill_switch_active

        if results["summary"]["all_kill_switches_worked"]:
            logger.info("Stress tests PASSED")
        else:
            logger.warning("Stress tests FAILED - review risk parameters")

    def _open_position(
        self,
        ticker: str,
        direction: Direction,
        price: float,
        size: int,
        leverage: float,
        take: Optional[float],
        stop: Optional[float],
        ttl_minutes: int,
        horizon: str,
        probability: float,
        regime: MarketRegime,
    ) -> bool:
        """Open a leveraged position."""
        if ticker in self.account.positions:
            return False

        if len(self.account.positions) >= self.max_positions:
            logger.debug(f"Max positions reached ({self.max_positions})")
            return False

        # Calculate costs
        position_value = price * size
        margin_required = position_value / leverage  # Simplified margin calculation
        notional_value = position_value * leverage

        if margin_required > self.account.cash:
            logger.debug(f"Insufficient margin: need {margin_required:,.0f}, have {self.account.cash:,.0f}")
            return False

        # Reserve margin
        self.account.cash -= margin_required
        self.account.margin_used += margin_required

        now = datetime.now(timezone.utc)
        self.account.positions[ticker] = MarginPosition(
            ticker=ticker,
            direction=direction.value,
            entry_time=now.isoformat(),
            entry_price=price,
            size=size,
            leverage=leverage,
            take=take,
            stop=stop,
            ttl_minutes=ttl_minutes,
            horizon=horizon,
            probability=probability,
            regime=regime.value,
            notional_value=notional_value,
        )

        # Cooldown
        cooldown_end = now + timedelta(minutes=self.config.cooldown_minutes)
        self.account.cooldown_map[ticker] = cooldown_end.isoformat()

        logger.info(
            f"OPEN: {ticker} {direction.value} @ {price:.2f} x{size} "
            f"[{leverage:.1f}x leverage, {regime.value}]"
        )

        return True

    def _close_position(
        self,
        ticker: str,
        price: float,
        reason: str,
    ) -> Optional[MarginTrade]:
        """Close a leveraged position."""
        if ticker not in self.account.positions:
            return None

        pos = self.account.positions[ticker]
        now = datetime.now(timezone.utc)

        # Calculate P&L (leveraged)
        if pos.direction == "LONG":
            pnl_per_share = price - pos.entry_price
        else:
            pnl_per_share = pos.entry_price - price

        pnl = pnl_per_share * pos.size * pos.leverage
        pnl_pct = (pnl / (pos.entry_price * pos.size)) * 100

        # Return margin + P&L
        margin_released = pos.entry_price * pos.size / pos.leverage
        self.account.cash += margin_released + pnl
        self.account.margin_used -= margin_released

        # Update peak equity
        self.account.peak_equity = max(self.account.peak_equity, self.account.equity)

        # Record trade
        trade = MarginTrade(
            ticker=ticker,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=now.isoformat(),
            entry_price=pos.entry_price,
            exit_price=price,
            size=pos.size,
            leverage=pos.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            horizon=pos.horizon,
            regime=pos.regime,
        )
        self.account.closed_trades.append(trade)

        # Update risk engine
        is_win = pnl > 0
        self.risk_engine.record_trade_result(pnl, is_win)

        # Sync account state
        if is_win:
            self.account.consecutive_losses = 0
        else:
            self.account.consecutive_losses += 1

        self.account.daily_pnl += pnl
        self.account.weekly_pnl += pnl

        del self.account.positions[ticker]

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} CLOSE: {ticker} @ {price:.2f} | "
            f"PnL: {pnl:+,.0f} ({pnl_pct:+.2f}%) [{reason}]"
        )

        return trade

    def _check_exits(self, quotes: Dict[str, Dict]) -> List[MarginTrade]:
        """Check exit conditions for all positions."""
        closed = []
        now = datetime.now(timezone.utc)

        for ticker in list(self.account.positions.keys()):
            pos = self.account.positions[ticker]
            quote = quotes.get(ticker, {})
            last_price = quote.get("last")

            if last_price is None:
                continue

            exit_reason = None
            exit_price = None

            # Check take profit
            if pos.take is not None:
                if pos.direction == "LONG" and last_price >= pos.take:
                    exit_reason = "take"
                    exit_price = pos.take
                elif pos.direction == "SHORT" and last_price <= pos.take:
                    exit_reason = "take"
                    exit_price = pos.take

            # Check stop loss
            if exit_reason is None and pos.stop is not None:
                if pos.direction == "LONG" and last_price <= pos.stop:
                    exit_reason = "stop"
                    exit_price = pos.stop
                elif pos.direction == "SHORT" and last_price >= pos.stop:
                    exit_reason = "stop"
                    exit_price = pos.stop

            # Check timeout
            if exit_reason is None:
                entry_time = datetime.fromisoformat(pos.entry_time)
                elapsed = (now - entry_time).total_seconds() / 60
                if elapsed >= pos.ttl_minutes:
                    exit_reason = "timeout"
                    exit_price = last_price

            if exit_reason:
                trade = self._close_position(ticker, exit_price, exit_reason)
                if trade:
                    closed.append(trade)

        return closed

    def _send_telegram_alert(self, message: str) -> None:
        """Send alert to Telegram."""
        if not self.config.telegram.enabled:
            return

        send_telegram(
            bot_token=self.config.telegram.bot_token or "",
            chat_id=self.config.telegram.chat_id or "",
            text=message,
        )

    def _send_trade_alert(self, trade: MarginTrade) -> None:
        """Send trade alert."""
        emoji = "✅" if trade.pnl > 0 else "❌"
        result = "ПРИБЫЛЬ" if trade.pnl > 0 else "УБЫТОК"

        risk_status = self.risk_engine.get_status()

        message = (
            f"🎰 MARGIN PAPER TRADING\n\n"
            f"{emoji} СДЕЛКА ЗАКРЫТА\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 {trade.ticker} ({trade.direction})\n"
            f"📊 Плечо: {trade.leverage:.1f}x\n"
            f"💰 {result}: {abs(trade.pnl):,.0f} ₽ ({trade.pnl_pct:+.2f}%)\n"
            f"📊 Причина: {trade.exit_reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 EQUITY: {self.account.equity:,.0f} ₽\n"
            f"📉 Drawdown: {risk_status['drawdown_pct']:.1f}%\n"
            f"🔢 Loss Streak: {risk_status['consecutive_losses']}"
        )

        if risk_status["kill_switch_active"]:
            message += f"\n\n⛔ KILL-SWITCH: {risk_status['kill_switch_reason']}"

        self._send_telegram_alert(message)

    def _send_signal_alert(
        self,
        ticker: str,
        direction: str,
        price: float,
        leverage: float,
        horizon: str,
        probability: float,
        regime: str,
        take: Optional[float],
        stop: Optional[float],
    ) -> None:
        """Send new signal alert."""
        pos = self.account.positions.get(ticker)
        if not pos:
            return

        dir_emoji = "📈" if direction == "LONG" else "📉"

        message = (
            f"🎰 MARGIN PAPER TRADING\n\n"
            f"{dir_emoji} НОВАЯ ПОЗИЦИЯ\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 {ticker} | {direction}\n"
            f"📊 Плечо: {leverage:.1f}x | Режим: {regime}\n"
            f"💵 Цена: {price:.2f} ₽\n"
            f"📦 Размер: {pos.size} шт. ({pos.notional_value:,.0f} ₽)\n"
            f"🎯 Take: {take:.2f} | Stop: {stop:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 EQUITY: {self.account.equity:,.0f} ₽"
        )

        self._send_telegram_alert(message)

    def run_cycle(self) -> None:
        """Run one trading cycle."""
        try:
            # Check kill-switch first
            kill_active, kill_reason = self.risk_engine.check_kill_switch()
            if kill_active:
                if not self.account.kill_switch_active:
                    # Just activated
                    self.account.kill_switch_active = True
                    self.account.kill_switch_reason = kill_reason or ""
                    self._send_telegram_alert(
                        f"⛔ KILL-SWITCH АКТИВИРОВАН\n\n{kill_reason}\n\n"
                        f"Торговля приостановлена."
                    )
                return

            self.account.kill_switch_active = False

            # Fetch data
            result = self.engine.run_cycle(self.conn)

            if result.errors:
                for err in result.errors:
                    logger.warning(f"Cycle error: {err}")

            # Get quotes
            quotes = self.engine.fetch_quotes_parallel()

            # Check exits
            closed = self._check_exits(quotes)
            for trade in closed:
                self._send_trade_alert(trade)

            # Get candles for regime detection
            candles_df = get_window(self.conn, minutes=3 * 24 * 60, interval=1)

            # Process signals
            now = datetime.now(timezone.utc)

            for sig in result.signals:
                ticker = sig.secid

                # Skip disabled horizons
                if sig.horizon in self.DISABLED_HORIZONS:
                    continue

                # Check cooldown
                if ticker in self.account.cooldown_map:
                    cooldown_end = datetime.fromisoformat(self.account.cooldown_map[ticker])
                    if now < cooldown_end:
                        continue

                # Risk assessment
                ticker_candles = candles_df[candles_df["secid"] == ticker]
                last_price = quotes.get(ticker, {}).get("last")
                atr = sig.stop and sig.entry and abs(sig.entry - sig.stop) / 0.4 if sig.stop and sig.entry else 0

                if not last_price or atr <= 0:
                    continue

                assessment = self.risk_engine.assess_trade(
                    ticker=ticker,
                    direction=sig.direction.value if hasattr(sig.direction, 'value') else sig.direction,
                    horizon=sig.horizon,
                    model_confidence=sig.probability,
                    candles_df=ticker_candles,
                    price=last_price,
                    atr=atr,
                )

                # Log assessment
                logger.debug(
                    f"{ticker}: {assessment.decision.value} "
                    f"(lev={assessment.leverage:.1f}, regime={assessment.regime.value})"
                )

                if assessment.decision == RiskDecision.DISABLE:
                    continue

                # Calculate position size
                shares, stop_price = self.risk_engine.calculate_position_size(
                    price=last_price,
                    atr=atr,
                    leverage=assessment.leverage,
                    direction=sig.direction.value if hasattr(sig.direction, 'value') else sig.direction,
                )

                if shares == 0:
                    continue

                # Calculate take profit (Risk:Reward = 1:1.75)
                stop_distance = abs(last_price - stop_price)
                if sig.direction == Direction.LONG or sig.direction == "LONG":
                    take_price = last_price + stop_distance * 1.75
                else:
                    take_price = last_price - stop_distance * 1.75

                # Open position
                direction = sig.direction if isinstance(sig.direction, Direction) else Direction(sig.direction)

                opened = self._open_position(
                    ticker=ticker,
                    direction=direction,
                    price=last_price,
                    size=shares,
                    leverage=assessment.leverage,
                    take=take_price,
                    stop=stop_price,
                    ttl_minutes=sig.ttl_minutes or 60,
                    horizon=sig.horizon,
                    probability=sig.probability,
                    regime=assessment.regime,
                )

                if opened:
                    self._send_signal_alert(
                        ticker=ticker,
                        direction=direction.value,
                        price=last_price,
                        leverage=assessment.leverage,
                        horizon=sig.horizon,
                        probability=sig.probability,
                        regime=assessment.regime.value,
                        take=take_price,
                        stop=stop_price,
                    )

            # Save state
            self.account.save()

        except Exception as e:
            logger.error(f"Cycle error: {e}")

    def run(self, duration_hours: float = 168) -> None:
        """Run margin paper trading."""
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            logger.info("Shutdown signal received...")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=duration_hours)

        logger.info("Margin paper trading started")
        logger.info(f"Capital: {self.account.equity:,.0f} RUB")
        logger.info(f"Max leverage: {self.risk_engine.lev_config.max_leverage}x")

        # Send startup message
        risk_status = self.risk_engine.get_status()
        self._send_telegram_alert(
            f"🎰 MARGIN PAPER TRADING\n\n"
            f"🚀 ТОРГОВЛЯ ЗАПУЩЕНА\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Капитал: {self.account.equity:,.0f} ₽\n"
            f"📊 Max плечо: {self.risk_engine.lev_config.max_leverage}x\n"
            f"⛔ Kill-switch: 5 losses / 2% daily / 10% DD\n"
            f"🚫 Запрещено: 1d, 1w горизонты"
        )

        cycle_count = 0
        while not shutdown_requested and datetime.now(timezone.utc) < end_time:
            cycle_count += 1
            self.run_cycle()

            if cycle_count % 12 == 0:
                status = self.risk_engine.get_status()
                logger.info(
                    f"Cycle {cycle_count}: equity={status['equity']:,.0f}, "
                    f"DD={status['drawdown_pct']:.1f}%, "
                    f"pos={len(self.account.positions)}, "
                    f"trades={len(self.account.closed_trades)}"
                )

            time.sleep(self.config.poll_seconds)

        logger.info("Margin paper trading stopped")

        # Final report
        self._send_hourly_report()

        self.conn.close()
        close_session()

    def _send_hourly_report(self) -> None:
        """Send hourly status report."""
        status = self.risk_engine.get_status()
        total_trades = len(self.account.closed_trades)
        wins = len([t for t in self.account.closed_trades if t.pnl > 0])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in self.account.closed_trades)
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        message = (
            f"🎰 MARGIN PAPER TRADING\n\n"
            f"📊 ОТЧЁТ\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Equity: {status['equity']:,.0f} ₽\n"
            f"📉 Drawdown: {status['drawdown_pct']:.1f}%\n"
            f"{pnl_emoji} P&L: {total_pnl:+,.0f} ₽\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Сделок: {total_trades}\n"
            f"✅ Win Rate: {win_rate:.1f}%\n"
            f"🔢 Loss Streak: {status['consecutive_losses']}\n"
            f"📍 Позиций: {len(self.account.positions)}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Status: {status['status']}"
        )

        if status["kill_switch_active"]:
            message += f"\n⛔ {status['kill_switch_reason']}"

        self._send_telegram_alert(message)


def main():
    parser = argparse.ArgumentParser(description="MOEX Agent Margin Paper Trading")
    parser.add_argument("--capital", type=float, default=200_000, help="Initial capital")
    parser.add_argument("--max-leverage", type=float, default=3.0, help="Maximum leverage")
    parser.add_argument("--max-positions", type=int, default=3, help="Max simultaneous positions")
    parser.add_argument("--duration-days", type=float, default=7, help="Duration in days")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")

    args = parser.parse_args()

    trader = MarginPaperTrader(
        config_path=args.config,
        initial_capital=args.capital,
        max_leverage=args.max_leverage,
        max_positions=args.max_positions,
        resume=not args.no_resume,
    )

    trader.run(duration_hours=args.duration_days * 24)


if __name__ == "__main__":
    main()

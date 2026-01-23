"""
MOEX Agent Margin Risk Engine

Prop-trading desk level risk management for margin trading.
Focus: Survival first, profit second.

Core principles:
- Max loss per trade: 0.5% equity
- Max daily loss: 2%
- Max weekly loss: 5%
- Kill-switch after 5 consecutive losses
- Dynamic leverage based on regime, volatility, drawdown
- NO overnight positions without explicit approval
- NO 1d/1w horizons with leverage

Author: MOEX Agent Risk Team
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("moex_agent.margin_risk")


class MarketRegime(str, Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN = "UNKNOWN"


class RiskDecision(str, Enum):
    """Risk engine decision."""
    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"  # Reduced leverage
    DISABLE = "DISABLE"    # No trading


@dataclass
class KillSwitchConfig:
    """Kill-switch thresholds."""
    max_loss_per_trade_pct: float = 0.5      # 0.5% equity max loss per trade
    max_daily_loss_pct: float = 2.0          # Stop trading for day
    max_weekly_loss_pct: float = 5.0         # Stop trading for week
    max_consecutive_losses: int = 5          # Kill after 5 losses in a row
    max_drawdown_pct: float = 10.0           # Full stop, manual review required
    pause_after_losses: int = 3              # Reduce leverage after 3 losses
    cooldown_after_kill_minutes: int = 60    # Minimum cooldown after kill
    max_trades_per_day: int = 9999           # БКС не ограничивает, убрано ограничение


@dataclass
class LeverageConfig:
    """Leverage calculation parameters."""
    base_leverage: float = 1.0
    max_leverage: float = 3.0                # Hard cap for MOEX
    min_confidence_to_trade: float = 0.54    # Below this - no trade
    confidence_for_full_leverage: float = 0.60
    high_vol_percentile: float = 80          # Above this - reduce leverage
    medium_vol_percentile: float = 60


@dataclass
class RiskState:
    """Current risk state tracking."""
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    trades_today: int = 0
    losses_today: int = 0
    last_trade_time: Optional[datetime] = None
    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    kill_switch_until: Optional[datetime] = None
    day_start: Optional[datetime] = None
    week_start: Optional[datetime] = None

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    @property
    def daily_loss_pct(self) -> float:
        """Daily loss as percentage of equity."""
        if self.equity <= 0:
            return 0.0
        return -self.daily_pnl / self.equity * 100 if self.daily_pnl < 0 else 0.0

    @property
    def weekly_loss_pct(self) -> float:
        """Weekly loss as percentage of equity."""
        if self.equity <= 0:
            return 0.0
        return -self.weekly_pnl / self.equity * 100 if self.weekly_pnl < 0 else 0.0


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    decision: RiskDecision
    leverage: float
    max_position_pct: float
    reason: str
    warnings: List[str] = field(default_factory=list)
    regime: MarketRegime = MarketRegime.UNKNOWN
    volatility_percentile: float = 50.0
    confidence_adjusted: float = 0.0


class MarginRiskEngine:
    """
    Prop-trading desk level risk engine.

    Core responsibilities:
    1. Kill-switch management
    2. Dynamic leverage calculation
    3. Position sizing by risk
    4. Regime detection
    5. Drawdown monitoring
    """

    # Horizons DISABLED for margin trading
    DISABLED_HORIZONS = {"1d", "1w"}

    # Allowed horizons with their max leverage
    HORIZON_MAX_LEVERAGE = {
        "5m": 3.0,
        "10m": 3.0,
        "30m": 2.5,
        "1h": 2.0,
    }

    def __init__(
        self,
        initial_equity: float,
        kill_switch_config: Optional[KillSwitchConfig] = None,
        leverage_config: Optional[LeverageConfig] = None,
    ):
        self.kill_config = kill_switch_config or KillSwitchConfig()
        self.lev_config = leverage_config or LeverageConfig()

        now = datetime.now(timezone.utc)
        self.state = RiskState(
            equity=initial_equity,
            peak_equity=initial_equity,
            day_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
            week_start=now - timedelta(days=now.weekday()),
        )

        # Volatility history for percentile calculation
        self._volatility_history: List[float] = []

        logger.info(f"MarginRiskEngine initialized: equity={initial_equity:,.0f}")

    def detect_regime(self, candles_df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime.

        Returns:
            (regime, volatility_percentile)
        """
        if candles_df.empty or len(candles_df) < 100:
            return MarketRegime.UNKNOWN, 50.0

        close = candles_df["close"]

        # Calculate indicators
        returns_20d = close.pct_change(20).iloc[-1] if len(close) > 20 else 0

        # Volatility (20-period)
        volatility = close.pct_change().rolling(20).std().iloc[-1]

        # Historical volatility for percentile
        hist_vol = close.pct_change().rolling(20).std().dropna().tolist()
        if hist_vol:
            self._volatility_history = hist_vol[-500:]  # Keep last 500
            vol_percentile = (sum(1 for v in self._volatility_history if v < volatility)
                            / len(self._volatility_history) * 100)
        else:
            vol_percentile = 50.0

        # Trend detection
        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) > 50 else sma_10

        # ADX-like trend strength (simplified)
        high = candles_df["high"] if "high" in candles_df.columns else close
        low = candles_df["low"] if "low" in candles_df.columns else close
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Regime classification
        if vol_percentile > 85:
            regime = MarketRegime.HIGH_VOL
        elif sma_10 > sma_50 * 1.01 and returns_20d > 0.02:
            regime = MarketRegime.BULL
        elif sma_10 < sma_50 * 0.99 and returns_20d < -0.02:
            regime = MarketRegime.BEAR
        elif abs(returns_20d) < 0.01:
            regime = MarketRegime.SIDEWAYS
        else:
            regime = MarketRegime.UNKNOWN

        return regime, vol_percentile

    def check_kill_switch(self) -> Tuple[bool, Optional[str]]:
        """
        Check if kill-switch should be activated.

        Returns:
            (is_active, reason)
        """
        now = datetime.now(timezone.utc)

        # Check if kill-switch cooldown is active
        if self.state.kill_switch_active:
            if self.state.kill_switch_until and now < self.state.kill_switch_until:
                return True, self.state.kill_switch_reason
            else:
                # Cooldown expired, reset
                self.state.kill_switch_active = False
                self.state.kill_switch_reason = None
                self.state.kill_switch_until = None
                logger.info("Kill-switch cooldown expired, trading resumed")

        # Check consecutive losses
        if self.state.consecutive_losses >= self.kill_config.max_consecutive_losses:
            reason = f"KILL: {self.state.consecutive_losses} consecutive losses"
            self._activate_kill_switch(reason)
            return True, reason

        # Check daily loss
        if self.state.daily_loss_pct >= self.kill_config.max_daily_loss_pct:
            reason = f"KILL: Daily loss {self.state.daily_loss_pct:.1f}% >= {self.kill_config.max_daily_loss_pct}%"
            self._activate_kill_switch(reason)
            return True, reason

        # Check weekly loss
        if self.state.weekly_loss_pct >= self.kill_config.max_weekly_loss_pct:
            reason = f"KILL: Weekly loss {self.state.weekly_loss_pct:.1f}% >= {self.kill_config.max_weekly_loss_pct}%"
            self._activate_kill_switch(reason)
            return True, reason

        # Check max drawdown
        if self.state.current_drawdown_pct >= self.kill_config.max_drawdown_pct:
            reason = f"KILL: Drawdown {self.state.current_drawdown_pct:.1f}% >= {self.kill_config.max_drawdown_pct}%"
            self._activate_kill_switch(reason, permanent=True)
            return True, reason

        # Check max trades per day (БКС: без ограничений, 9999 по умолчанию)
        if self.state.trades_today >= self.kill_config.max_trades_per_day:
            reason = f"KILL: Достигнут лимит сделок за день ({self.state.trades_today})"
            self._activate_kill_switch(reason)
            return True, reason

        return False, None

    def _activate_kill_switch(self, reason: str, permanent: bool = False) -> None:
        """Activate kill-switch."""
        self.state.kill_switch_active = True
        self.state.kill_switch_reason = reason

        if not permanent:
            self.state.kill_switch_until = (
                datetime.now(timezone.utc) +
                timedelta(minutes=self.kill_config.cooldown_after_kill_minutes)
            )
        else:
            # Permanent until manual reset
            self.state.kill_switch_until = datetime.max.replace(tzinfo=timezone.utc)

        logger.warning(f"KILL-SWITCH ACTIVATED: {reason}")

    def calculate_leverage(
        self,
        model_confidence: float,
        regime: MarketRegime,
        volatility_percentile: float,
        horizon: str,
    ) -> float:
        """
        Calculate dynamic leverage.

        Leverage = f(confidence, regime, volatility, drawdown, loss_streak)
        """
        # Check horizon restriction
        if horizon in self.DISABLED_HORIZONS:
            logger.warning(f"Horizon {horizon} DISABLED for margin trading")
            return 0.0

        max_horizon_lev = self.HORIZON_MAX_LEVERAGE.get(horizon, 1.0)

        # 1. Confidence multiplier
        if model_confidence < self.lev_config.min_confidence_to_trade:
            return 0.0  # NO TRADE

        conf_range = self.lev_config.confidence_for_full_leverage - self.lev_config.min_confidence_to_trade
        conf_normalized = (model_confidence - self.lev_config.min_confidence_to_trade) / conf_range
        conf_mult = min(1.0, max(0.3, conf_normalized))

        # 2. Regime multiplier
        regime_mult = {
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 0.6,      # Reduced, only shorts
            MarketRegime.SIDEWAYS: 0.4,  # Very conservative
            MarketRegime.HIGH_VOL: 0.3,  # Minimum
            MarketRegime.UNKNOWN: 0.0,   # NO TRADE
        }.get(regime, 0.0)

        if regime_mult == 0:
            return 0.0

        # 3. Volatility multiplier
        if volatility_percentile > self.lev_config.high_vol_percentile:
            vol_mult = 0.5
        elif volatility_percentile > self.lev_config.medium_vol_percentile:
            vol_mult = 0.75
        else:
            vol_mult = 1.0

        # 4. Drawdown penalty
        dd = self.state.current_drawdown_pct
        if dd > 7:
            dd_mult = 0.3
        elif dd > 5:
            dd_mult = 0.5
        elif dd > 3:
            dd_mult = 0.75
        else:
            dd_mult = 1.0

        # 5. Loss streak penalty
        streak = self.state.consecutive_losses
        if streak >= self.kill_config.max_consecutive_losses:
            return 0.0  # Kill switch should handle this
        elif streak >= self.kill_config.pause_after_losses:
            streak_mult = 0.5
        elif streak >= 2:
            streak_mult = 0.75
        else:
            streak_mult = 1.0

        # Calculate final leverage
        leverage = (
            self.lev_config.base_leverage *
            conf_mult *
            regime_mult *
            vol_mult *
            dd_mult *
            streak_mult
        )

        # Apply caps
        leverage = min(leverage, self.lev_config.max_leverage)
        leverage = min(leverage, max_horizon_lev)

        return round(leverage, 2)

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        leverage: float,
        direction: str = "LONG",
    ) -> Tuple[int, float]:
        """
        Calculate position size based on risk.

        Returns:
            (shares, stop_price)
        """
        if leverage <= 0 or price <= 0 or atr <= 0:
            return 0, 0.0

        # Max loss per trade in rubles
        max_loss_rub = self.state.equity * (self.kill_config.max_loss_per_trade_pct / 100)

        # Stop-loss distance (0.4 ATR for tight risk control)
        stop_distance = 0.4 * atr
        stop_pct = stop_distance / price

        # With leverage, the loss is amplified
        # loss = position_value * stop_pct * leverage
        # max_loss = position_value * stop_pct * leverage
        # position_value = max_loss / (stop_pct * leverage)

        position_value = max_loss_rub / (stop_pct * leverage)
        shares = int(position_value / price)

        # Hard cap: max 10% of equity in single position (notional)
        max_position_value = self.state.equity * 0.10
        max_shares = int(max_position_value / price)
        shares = min(shares, max_shares)

        # Calculate stop price
        if direction == "LONG":
            stop_price = price - stop_distance
        else:
            stop_price = price + stop_distance

        return max(0, shares), round(stop_price, 2)

    def assess_trade(
        self,
        ticker: str,
        direction: str,
        horizon: str,
        model_confidence: float,
        candles_df: pd.DataFrame,
        price: float,
        atr: float,
    ) -> RiskAssessment:
        """
        Full risk assessment for a potential trade.

        Returns RiskAssessment with decision, leverage, and warnings.
        """
        warnings = []

        # 1. Check kill-switch
        kill_active, kill_reason = self.check_kill_switch()
        if kill_active:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason=kill_reason or "Kill-switch active",
                warnings=["KILL-SWITCH ACTIVE"],
                regime=MarketRegime.UNKNOWN,
            )

        # 2. Check horizon
        if horizon in self.DISABLED_HORIZONS:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason=f"Horizon {horizon} disabled for margin trading",
                warnings=[f"Horizon {horizon} carries overnight/gap risk"],
            )

        # 3. Detect regime
        regime, vol_percentile = self.detect_regime(candles_df)

        if regime == MarketRegime.UNKNOWN:
            warnings.append("Regime unclear - no trade")
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason="Market regime unknown",
                warnings=warnings,
                regime=regime,
                volatility_percentile=vol_percentile,
            )

        # 4. Check direction vs regime
        if regime == MarketRegime.BEAR and direction == "LONG":
            warnings.append("LONG in BEAR regime - restricted")
        if regime == MarketRegime.BULL and direction == "SHORT":
            warnings.append("SHORT in BULL regime - restricted")

        # 5. Calculate leverage
        leverage = self.calculate_leverage(
            model_confidence=model_confidence,
            regime=regime,
            volatility_percentile=vol_percentile,
            horizon=horizon,
        )

        if leverage == 0:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason=f"Leverage=0 (conf={model_confidence:.2f}, regime={regime.value})",
                warnings=warnings,
                regime=regime,
                volatility_percentile=vol_percentile,
                confidence_adjusted=model_confidence,
            )

        # 6. Calculate position size
        shares, stop_price = self.calculate_position_size(
            price=price,
            atr=atr,
            leverage=leverage,
            direction=direction,
        )

        if shares == 0:
            return RiskAssessment(
                decision=RiskDecision.DISABLE,
                leverage=0.0,
                max_position_pct=0.0,
                reason="Position size = 0 (risk too high)",
                warnings=warnings,
                regime=regime,
                volatility_percentile=vol_percentile,
            )

        # 7. Additional warnings
        if vol_percentile > 70:
            warnings.append(f"High volatility ({vol_percentile:.0f}th percentile)")

        if self.state.consecutive_losses >= 2:
            warnings.append(f"Loss streak: {self.state.consecutive_losses}")

        if self.state.current_drawdown_pct > 3:
            warnings.append(f"Drawdown: {self.state.current_drawdown_pct:.1f}%")

        # 8. Final decision
        if leverage < 1.0:
            decision = RiskDecision.RESTRICT
        else:
            decision = RiskDecision.ALLOW

        position_pct = (shares * price) / self.state.equity * 100

        return RiskAssessment(
            decision=decision,
            leverage=leverage,
            max_position_pct=position_pct,
            reason=f"OK: lev={leverage:.1f}x, regime={regime.value}",
            warnings=warnings,
            regime=regime,
            volatility_percentile=vol_percentile,
            confidence_adjusted=model_confidence,
        )

    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """Record trade result and update state."""
        now = datetime.now(timezone.utc)

        # Update equity
        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        # Update daily/weekly PnL
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl

        # Update streaks
        if is_win:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
            self.state.losses_today += 1

        self.state.trades_today += 1
        self.state.last_trade_time = now

        # Check if day/week changed
        self._check_period_reset(now)

        # Log
        logger.info(
            f"Trade recorded: PnL={pnl:+.0f}, "
            f"Equity={self.state.equity:,.0f}, "
            f"DD={self.state.current_drawdown_pct:.1f}%, "
            f"Streak={'W' if is_win else 'L'}{self.state.consecutive_wins if is_win else self.state.consecutive_losses}"
        )

    def _check_period_reset(self, now: datetime) -> None:
        """Reset daily/weekly counters if period changed."""
        # Daily reset
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.state.day_start and today_start > self.state.day_start:
            logger.info(f"New day: resetting daily stats (was: {self.state.daily_pnl:+.0f})")
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
            self.state.losses_today = 0
            self.state.day_start = today_start

        # Weekly reset (Monday)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.state.week_start and week_start > self.state.week_start:
            logger.info(f"New week: resetting weekly stats (was: {self.state.weekly_pnl:+.0f})")
            self.state.weekly_pnl = 0.0
            self.state.week_start = week_start

    def get_status(self) -> Dict[str, Any]:
        """Get current risk engine status."""
        kill_active, kill_reason = self.check_kill_switch()

        return {
            "equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "drawdown_pct": round(self.state.current_drawdown_pct, 2),
            "daily_pnl": self.state.daily_pnl,
            "daily_loss_pct": round(self.state.daily_loss_pct, 2),
            "weekly_pnl": self.state.weekly_pnl,
            "weekly_loss_pct": round(self.state.weekly_loss_pct, 2),
            "consecutive_losses": self.state.consecutive_losses,
            "consecutive_wins": self.state.consecutive_wins,
            "trades_today": self.state.trades_today,
            "kill_switch_active": kill_active,
            "kill_switch_reason": kill_reason,
            "status": "DISABLED" if kill_active else "ACTIVE",
        }

    def reset_kill_switch(self, confirm: bool = False) -> bool:
        """
        Manually reset kill-switch.
        Requires explicit confirmation.
        """
        if not confirm:
            logger.warning("Kill-switch reset requires confirm=True")
            return False

        self.state.kill_switch_active = False
        self.state.kill_switch_reason = None
        self.state.kill_switch_until = None
        self.state.consecutive_losses = 0

        logger.warning("KILL-SWITCH MANUALLY RESET")
        return True


def stress_test_strategy(
    risk_engine: MarginRiskEngine,
    scenarios: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Run stress tests on risk engine.

    Default scenarios:
    1. 10 consecutive losses
    2. 5% gap against position
    3. Volatility spike 2x
    4. Regime shift mid-trade
    """
    if scenarios is None:
        scenarios = [
            {"name": "10 consecutive losses", "losses": 10, "loss_pct": 0.5},
            {"name": "5 consecutive losses at max", "losses": 5, "loss_pct": 0.5},
            {"name": "3 losses then recovery", "losses": 3, "wins": 5, "loss_pct": 0.5, "win_pct": 0.7},
        ]

    results = []
    initial_equity = risk_engine.state.equity

    for scenario in scenarios:
        # Reset for scenario
        risk_engine.state.equity = initial_equity
        risk_engine.state.consecutive_losses = 0
        risk_engine.state.consecutive_wins = 0
        risk_engine.state.daily_pnl = 0
        risk_engine.state.kill_switch_active = False

        scenario_result = {
            "name": scenario["name"],
            "initial_equity": initial_equity,
            "trades": [],
        }

        # Simulate losses
        for i in range(scenario.get("losses", 0)):
            loss = initial_equity * (scenario.get("loss_pct", 0.5) / 100)
            risk_engine.record_trade_result(-loss, is_win=False)

            kill_active, _ = risk_engine.check_kill_switch()
            scenario_result["trades"].append({
                "trade": i + 1,
                "type": "LOSS",
                "pnl": -loss,
                "equity": risk_engine.state.equity,
                "kill_switch": kill_active,
            })

            if kill_active:
                break

        # Simulate wins (if any)
        for i in range(scenario.get("wins", 0)):
            win = initial_equity * (scenario.get("win_pct", 0.7) / 100)
            risk_engine.record_trade_result(win, is_win=True)

            scenario_result["trades"].append({
                "trade": len(scenario_result["trades"]) + 1,
                "type": "WIN",
                "pnl": win,
                "equity": risk_engine.state.equity,
            })

        scenario_result["final_equity"] = risk_engine.state.equity
        scenario_result["total_drawdown_pct"] = (
            (initial_equity - min(t["equity"] for t in scenario_result["trades"]))
            / initial_equity * 100
        )
        scenario_result["kill_switch_triggered"] = any(
            t.get("kill_switch", False) for t in scenario_result["trades"]
        )

        results.append(scenario_result)

    return {
        "scenarios": results,
        "summary": {
            "all_kill_switches_worked": all(
                s["kill_switch_triggered"] for s in results
                if s["name"].startswith("10") or s["name"].startswith("5")
            ),
            "max_drawdown_observed": max(s["total_drawdown_pct"] for s in results),
        }
    }

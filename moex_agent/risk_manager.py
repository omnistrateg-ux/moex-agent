"""
MOEX Agent - Расширенный риск-менеджмент

Контролирует риски на уровне портфеля:
- Дневной стоп-лосс
- Максимальный drawdown
- Ограничение корреляции позиций
- Динамический размер позиции
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("moex_agent.risk_manager")


@dataclass
class RiskLimits:
    """Лимиты риска."""
    max_daily_loss_pct: float = 2.0  # Максимальный дневной убыток (%)
    max_drawdown_pct: float = 10.0  # Максимальный drawdown от пика (%)
    max_position_pct: float = 10.0  # Максимальный размер позиции (%)
    max_correlated_positions: int = 3  # Макс. число позиций в одном секторе
    min_cash_pct: float = 20.0  # Минимальный остаток кэша (%)
    pause_after_losses: int = 3  # Пауза после N убыточных сделок подряд
    pause_duration_minutes: int = 60  # Длительность паузы


@dataclass
class RiskState:
    """Текущее состояние риска."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    trading_paused: bool = False
    pause_until: Optional[str] = None
    daily_reset_time: str = ""

    def __post_init__(self):
        if not self.daily_reset_time:
            self.daily_reset_time = datetime.now(timezone.utc).date().isoformat()


class RiskManager:
    """
    Менеджер рисков для контроля торговли.

    Автоматически останавливает торговлю при достижении лимитов.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.state = RiskState()

    def update_equity(self, equity: float, initial_capital: float) -> None:
        """Обновить пиковый капитал и drawdown."""
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (
                (self.state.peak_equity - equity) / self.state.peak_equity * 100
            )

    def check_daily_reset(self) -> None:
        """Сбросить дневную статистику если новый день."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.daily_reset_time != today:
            logger.info(f"Daily risk reset: {self.state.daily_reset_time} -> {today}")
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_reset_time = today

    def record_trade(self, pnl: float) -> None:
        """Записать результат сделки."""
        self.check_daily_reset()

        self.state.daily_pnl += pnl
        self.state.daily_trades += 1

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Проверить паузу после серии убытков
        if self.state.consecutive_losses >= self.limits.pause_after_losses:
            self._pause_trading(f"После {self.state.consecutive_losses} убыточных сделок подряд")

    def _pause_trading(self, reason: str) -> None:
        """Приостановить торговлю."""
        pause_until = datetime.now(timezone.utc) + timedelta(minutes=self.limits.pause_duration_minutes)
        self.state.trading_paused = True
        self.state.pause_until = pause_until.isoformat()
        logger.warning(f"Trading paused until {pause_until}: {reason}")

    def _check_pause_expired(self) -> None:
        """Проверить не истекла ли пауза."""
        if self.state.trading_paused and self.state.pause_until:
            pause_until = datetime.fromisoformat(self.state.pause_until)
            if datetime.now(timezone.utc) >= pause_until:
                self.state.trading_paused = False
                self.state.pause_until = None
                self.state.consecutive_losses = 0
                logger.info("Trading resumed after pause")

    def can_open_position(
        self,
        equity: float,
        cash: float,
        initial_capital: float,
        position_value: float,
        current_positions: int,
    ) -> tuple[bool, str]:
        """
        Проверить можно ли открыть новую позицию.

        Returns:
            (allowed, reason)
        """
        self.check_daily_reset()
        self._check_pause_expired()
        self.update_equity(equity, initial_capital)

        # 1. Торговля на паузе
        if self.state.trading_paused:
            return False, f"Trading paused until {self.state.pause_until}"

        # 2. Дневной лимит убытка
        daily_loss_pct = abs(self.state.daily_pnl) / initial_capital * 100
        if self.state.daily_pnl < 0 and daily_loss_pct >= self.limits.max_daily_loss_pct:
            return False, f"Daily loss limit reached: {daily_loss_pct:.2f}% >= {self.limits.max_daily_loss_pct}%"

        # 3. Максимальный drawdown
        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            return False, f"Max drawdown reached: {self.state.current_drawdown_pct:.2f}% >= {self.limits.max_drawdown_pct}%"

        # 4. Минимальный остаток кэша
        cash_pct = cash / initial_capital * 100
        if cash_pct < self.limits.min_cash_pct:
            return False, f"Minimum cash reserve: {cash_pct:.1f}% < {self.limits.min_cash_pct}%"

        # 5. Максимальный размер позиции
        position_pct = position_value / initial_capital * 100
        if position_pct > self.limits.max_position_pct:
            return False, f"Position too large: {position_pct:.1f}% > {self.limits.max_position_pct}%"

        return True, "OK"

    def get_position_size_multiplier(self, equity: float, initial_capital: float) -> float:
        """
        Получить множитель размера позиции на основе текущего риска.

        При увеличении drawdown уменьшаем размер позиции.
        """
        # Базовый размер = 1.0
        multiplier = 1.0

        # Уменьшаем при drawdown
        if self.state.current_drawdown_pct > 3:
            multiplier *= 0.75
        if self.state.current_drawdown_pct > 5:
            multiplier *= 0.5
        if self.state.current_drawdown_pct > 7:
            multiplier *= 0.25

        # Уменьшаем после серии убытков
        if self.state.consecutive_losses >= 2:
            multiplier *= 0.75

        return max(0.1, multiplier)

    def get_status(self) -> Dict:
        """Получить текущий статус риск-менеджера."""
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "consecutive_losses": self.state.consecutive_losses,
            "current_drawdown_pct": self.state.current_drawdown_pct,
            "trading_paused": self.state.trading_paused,
            "pause_until": self.state.pause_until,
            "position_size_multiplier": self.get_position_size_multiplier(0, 1),
        }

    def to_dict(self) -> Dict:
        """Сериализация состояния."""
        return {
            "limits": {
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_position_pct": self.limits.max_position_pct,
                "pause_after_losses": self.limits.pause_after_losses,
            },
            "state": {
                "daily_pnl": self.state.daily_pnl,
                "daily_trades": self.state.daily_trades,
                "consecutive_losses": self.state.consecutive_losses,
                "peak_equity": self.state.peak_equity,
                "current_drawdown_pct": self.state.current_drawdown_pct,
                "trading_paused": self.state.trading_paused,
                "pause_until": self.state.pause_until,
                "daily_reset_time": self.state.daily_reset_time,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RiskManager":
        """Десериализация состояния."""
        limits_data = data.get("limits", {})
        state_data = data.get("state", {})

        limits = RiskLimits(
            max_daily_loss_pct=limits_data.get("max_daily_loss_pct", 2.0),
            max_drawdown_pct=limits_data.get("max_drawdown_pct", 10.0),
            max_position_pct=limits_data.get("max_position_pct", 10.0),
            pause_after_losses=limits_data.get("pause_after_losses", 3),
        )

        manager = cls(limits)
        manager.state = RiskState(
            daily_pnl=state_data.get("daily_pnl", 0.0),
            daily_trades=state_data.get("daily_trades", 0),
            consecutive_losses=state_data.get("consecutive_losses", 0),
            peak_equity=state_data.get("peak_equity", 0.0),
            current_drawdown_pct=state_data.get("current_drawdown_pct", 0.0),
            trading_paused=state_data.get("trading_paused", False),
            pause_until=state_data.get("pause_until"),
            daily_reset_time=state_data.get("daily_reset_time", ""),
        )

        return manager

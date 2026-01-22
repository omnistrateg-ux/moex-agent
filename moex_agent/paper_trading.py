"""
MOEX Agent Paper Trading (Virtual Account)

Симуляция торговли в реальном времени с виртуальным счётом.
Использует реальные данные с биржи, но сделки виртуальные.

Usage:
    python -m moex_agent.paper_trading --capital 200000 --duration-days 7

Features:
    - Виртуальный счёт с начальным капиталом
    - Торговля по сигналам в реальном времени
    - Hourly отчёты в Telegram
    - Сохранение состояния в JSON
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
from .engine import PipelineEngine
from .moex_iss import close_session
from .qwen import _rule_based_analysis, _get_ticker_liquidity, format_telegram_message, QwenAnalysis
from .storage import connect, get_window
from .telegram import send_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moex_agent.paper_trading")

STATE_FILE = Path("data/paper_trading_state.json")


@dataclass
class Position:
    """Открытая позиция."""
    ticker: str
    direction: str  # LONG or SHORT
    entry_time: str
    entry_price: float
    size: int  # количество акций
    take: Optional[float]
    stop: Optional[float]
    ttl_minutes: int
    horizon: str
    probability: float
    recommendation: str


@dataclass
class ClosedTrade:
    """Закрытая сделка."""
    ticker: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: int
    pnl: float  # в рублях
    pnl_pct: float
    exit_reason: str
    horizon: str


@dataclass
class PaperAccount:
    """Виртуальный торговый счёт."""
    initial_capital: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    start_time: str = ""
    last_report_time: str = ""
    cooldown_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now(timezone.utc).isoformat()
        if not self.last_report_time:
            self.last_report_time = self.start_time

    @property
    def positions_value(self) -> float:
        """Стоимость открытых позиций по цене входа."""
        return sum(pos.entry_price * pos.size for pos in self.positions.values())

    @property
    def equity(self) -> float:
        """Текущий капитал (cash + позиции)."""
        return self.cash + self.positions_value

    @property
    def total_pnl(self) -> float:
        """Общий P&L."""
        return sum(t.pnl for t in self.closed_trades)

    @property
    def total_pnl_pct(self) -> float:
        """P&L в процентах от начального капитала."""
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def win_rate(self) -> float:
        """Процент выигрышных сделок."""
        if not self.closed_trades:
            return 0.0
        wins = len([t for t in self.closed_trades if t.pnl > 0])
        return wins / len(self.closed_trades) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades],
            "start_time": self.start_time,
            "last_report_time": self.last_report_time,
            "cooldown_map": self.cooldown_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperAccount":
        """Десериализация из словаря."""
        positions = {k: Position(**v) for k, v in data.get("positions", {}).items()}
        closed_trades = [ClosedTrade(**t) for t in data.get("closed_trades", [])]
        return cls(
            initial_capital=data["initial_capital"],
            cash=data["cash"],
            positions=positions,
            closed_trades=closed_trades,
            start_time=data.get("start_time", ""),
            last_report_time=data.get("last_report_time", ""),
            cooldown_map=data.get("cooldown_map", {}),
        )

    def save(self, path: Path = STATE_FILE) -> None:
        """Сохранить состояние в файл."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug(f"State saved to {path}")

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> Optional["PaperAccount"]:
        """Загрузить состояние из файла."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return None


class PaperTrader:
    """
    Виртуальный трейдер.

    Использует реальные данные с MOEX, но сделки виртуальные.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        initial_capital: float = 200_000,
        position_size_pct: float = 0.05,  # 5% на сделку
        max_positions: int = 5,
        resume: bool = True,
    ):
        self.config = load_config(config_path)
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

        # Загрузить или создать счёт
        if resume:
            self.account = PaperAccount.load()
        else:
            self.account = None

        if self.account is None:
            self.account = PaperAccount(
                initial_capital=initial_capital,
                cash=initial_capital,
            )
            logger.info(f"Created new paper account with {initial_capital:,.0f} RUB")
        else:
            logger.info(f"Resumed paper account: cash={self.account.cash:,.0f}, "
                       f"positions={len(self.account.positions)}, "
                       f"trades={len(self.account.closed_trades)}")

        # Инициализация движка
        self.engine = PipelineEngine(self.config)
        self.engine.load_models()

        self.conn = connect(self.config.sqlite_path)

    def _calculate_position_size(self, price: float) -> int:
        """Рассчитать размер позиции в акциях."""
        position_value = self.account.cash * self.position_size_pct
        shares = int(position_value / price)
        return max(1, shares)

    def _open_position(
        self,
        ticker: str,
        direction: Direction,
        price: float,
        take: Optional[float],
        stop: Optional[float],
        ttl_minutes: int,
        horizon: str,
        probability: float,
        recommendation: str,
    ) -> bool:
        """Открыть позицию."""
        if ticker in self.account.positions:
            return False

        if len(self.account.positions) >= self.max_positions:
            logger.debug(f"Max positions reached ({self.max_positions})")
            return False

        size = self._calculate_position_size(price)
        cost = price * size

        if cost > self.account.cash:
            logger.debug(f"Insufficient cash for {ticker}: need {cost:,.0f}, have {self.account.cash:,.0f}")
            return False

        # Резервируем деньги
        self.account.cash -= cost

        now = datetime.now(timezone.utc)
        self.account.positions[ticker] = Position(
            ticker=ticker,
            direction=direction.value,
            entry_time=now.isoformat(),
            entry_price=price,
            size=size,
            take=take,
            stop=stop,
            ttl_minutes=ttl_minutes,
            horizon=horizon,
            probability=probability,
            recommendation=recommendation,
        )

        # Cooldown
        cooldown_end = now + timedelta(minutes=self.config.cooldown_minutes)
        self.account.cooldown_map[ticker] = cooldown_end.isoformat()

        take_str = f"{take:.2f}" if take else "N/A"
        stop_str = f"{stop:.2f}" if stop else "N/A"
        logger.info(f"📈 OPEN: {ticker} {direction.value} @ {price:.2f} x{size} "
                   f"(take={take_str}, stop={stop_str})")

        return True

    def _close_position(
        self,
        ticker: str,
        price: float,
        reason: str,
    ) -> Optional[ClosedTrade]:
        """Закрыть позицию."""
        if ticker not in self.account.positions:
            return None

        pos = self.account.positions[ticker]
        now = datetime.now(timezone.utc)

        # Рассчитать P&L
        if pos.direction == "LONG":
            pnl = (price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - price) * pos.size

        pnl_pct = (pnl / (pos.entry_price * pos.size)) * 100

        # Вернуть деньги + P&L
        self.account.cash += pos.entry_price * pos.size + pnl

        trade = ClosedTrade(
            ticker=ticker,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=now.isoformat(),
            entry_price=pos.entry_price,
            exit_price=price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            horizon=pos.horizon,
        )
        self.account.closed_trades.append(trade)

        del self.account.positions[ticker]

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{emoji} CLOSE: {ticker} @ {price:.2f} | PnL: {pnl:+,.0f} ({pnl_pct:+.2f}%) [{reason}]")

        return trade

    def _check_exits(self, quotes: Dict[str, Dict]) -> List[ClosedTrade]:
        """Проверить условия закрытия позиций."""
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

    def _send_hourly_report(self) -> None:
        """Отправить часовой отчёт в Telegram."""
        if not self.config.telegram.enabled:
            logger.warning("Telegram disabled in config")
            return

        now = datetime.now(timezone.utc)
        uptime = now - datetime.fromisoformat(self.account.start_time)
        hours = uptime.total_seconds() / 3600

        # Статистика
        total_trades = len(self.account.closed_trades)
        wins = len([t for t in self.account.closed_trades if t.pnl > 0])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        # Рассчитываем прибыль/убыток
        pnl = self.account.total_pnl
        pnl_emoji = "📈 ПРИБЫЛЬ" if pnl >= 0 else "📉 УБЫТОК"

        # Формируем сообщение
        lines = [
            "🎮 СИМУЛЯЦИЯ - ВИРТУАЛЬНЫЙ СЧЁТ",
            "",
            "📊 ОТЧЁТ ПО ТОРГОВЛЕ",
            f"Время работы: {hours:.1f} ч.",
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "💰 СОСТОЯНИЕ СЧЁТА",
            "━━━━━━━━━━━━━━━━━━━━",
            f"Начальный капитал: {self.account.initial_capital:,.0f} ₽",
            f"Свободные деньги: {self.account.cash:,.0f} ₽",
            f"В открытых сделках: {self.account.positions_value:,.0f} ₽",
            "",
            f"💵 БАЛАНС СЧЁТА: {self.account.equity:,.0f} ₽",
            f"{pnl_emoji}: {abs(pnl):,.0f} ₽ ({self.account.total_pnl_pct:+.2f}%)",
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "📊 СТАТИСТИКА",
            "━━━━━━━━━━━━━━━━━━━━",
            f"Всего сделок: {total_trades}",
            f"Выигрышных: {wins} ({win_rate:.1f}%)",
            f"Открытых позиций: {len(self.account.positions)}",
        ]

        # Открытые позиции
        if self.account.positions:
            lines.append("")
            lines.append("📍 ОТКРЫТЫЕ ПОЗИЦИИ:")
            for ticker, pos in self.account.positions.items():
                cost = pos.entry_price * pos.size
                dir_ru = "ПОКУПКА" if pos.direction == "LONG" else "ПРОДАЖА"
                lines.append(f"  • {ticker} {dir_ru} @ {pos.entry_price:.2f} x{pos.size} = {cost:,.0f}₽")

        # Закрытые сделки
        if self.account.closed_trades:
            lines.append("")
            lines.append("📋 ИСТОРИЯ СДЕЛОК:")

            for i, t in enumerate(self.account.closed_trades[-10:], 1):  # Последние 10
                emoji = "✅" if t.pnl > 0 else "❌"
                result = "прибыль" if t.pnl > 0 else "убыток"
                dir_ru = "покупка" if t.direction == "LONG" else "продажа"
                lines.append(
                    f"{i}. {emoji} {t.ticker} ({dir_ru}): {result} {abs(t.pnl):,.0f}₽"
                )

            lines.append("")
            if pnl >= 0:
                lines.append(f"✅ ОБЩАЯ ПРИБЫЛЬ: +{pnl:,.0f} ₽")
            else:
                lines.append(f"❌ ОБЩИЙ УБЫТОК: {pnl:,.0f} ₽")

        message = "\n".join(lines)

        success = send_telegram(
            bot_token=self.config.telegram.bot_token or "",
            chat_id=self.config.telegram.chat_id or "",
            text=message,
        )

        if success:
            logger.info("Hourly report sent to Telegram")
            self.account.last_report_time = now.isoformat()
        else:
            logger.warning("Failed to send Telegram report")

    def _send_trade_alert(self, trade: ClosedTrade) -> None:
        """Отправить алерт о сделке в Telegram."""
        if not self.config.telegram.enabled:
            return

        emoji = "✅" if trade.pnl > 0 else "❌"
        result_text = "ПРИБЫЛЬ" if trade.pnl > 0 else "УБЫТОК"
        dir_ru = "покупка" if trade.direction == "LONG" else "продажа"

        exit_ru = {"take": "тейк-профит", "stop": "стоп-лосс", "timeout": "по времени"}.get(trade.exit_reason, trade.exit_reason)

        total_pnl = self.account.total_pnl
        total_result = "ПРИБЫЛЬ" if total_pnl >= 0 else "УБЫТОК"

        message = (
            f"🎮 СИМУЛЯЦИЯ - ВИРТУАЛЬНЫЙ СЧЁТ\n\n"
            f"{emoji} СДЕЛКА ЗАКРЫТА\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 {trade.ticker} ({dir_ru})\n"
            f"💰 {result_text}: {abs(trade.pnl):,.0f} ₽ ({trade.pnl_pct:+.2f}%)\n"
            f"📊 Причина: {exit_ru}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 БАЛАНС СЧЁТА: {self.account.equity:,.0f} ₽\n"
            f"📈 ОБЩИЙ {total_result}: {abs(total_pnl):,.0f} ₽"
        )

        send_telegram(
            bot_token=self.config.telegram.bot_token or "",
            chat_id=self.config.telegram.chat_id or "",
            text=message,
        )

    def _send_signal_alert(self, ticker: str, direction: str, price: float,
                          horizon: str, probability: float, recommendation: str,
                          take: Optional[float], stop: Optional[float]) -> None:
        """Отправить алерт о новом сигнале."""
        if not self.config.telegram.enabled:
            return

        emoji_map = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "STRONG_SELL": "🔴🔴", "SELL": "🔴"}
        emoji = emoji_map.get(recommendation, "⚪")
        dir_emoji = "📈" if direction == "LONG" else "📉"
        dir_ru = "ПОКУПКА" if direction == "LONG" else "ПРОДАЖА"

        # Стоимость позиции
        pos = self.account.positions.get(ticker)
        cost = pos.entry_price * pos.size if pos else 0

        lines = [
            "🎮 СИМУЛЯЦИЯ - ВИРТУАЛЬНЫЙ СЧЁТ",
            "",
            f"{emoji} НОВАЯ ПОЗИЦИЯ",
            "━━━━━━━━━━━━━━━━━━━━",
            f"{dir_emoji} {ticker} | {dir_ru}",
            f"📊 Вероятность: {probability:.0%}",
            f"⏱ Горизонт: {horizon}",
            f"💵 Цена входа: {price:.2f} ₽",
        ]

        if pos:
            lines.append(f"📦 Количество: {pos.size} шт.")
            lines.append(f"💰 Сумма: {cost:,.0f} ₽")

        if take and stop:
            lines.append(f"🎯 Тейк: {take:.2f} | Стоп: {stop:.2f}")

        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"💵 БАЛАНС СЧЁТА: {self.account.equity:,.0f} ₽")

        message = "\n".join(lines)

        send_telegram(
            bot_token=self.config.telegram.bot_token or "",
            chat_id=self.config.telegram.chat_id or "",
            text=message,
        )

    def run_cycle(self) -> None:
        """Выполнить один торговый цикл."""
        try:
            # Получить данные
            result = self.engine.run_cycle(self.conn)

            if result.errors:
                for err in result.errors:
                    logger.warning(f"Cycle error: {err}")

            # Получить текущие котировки
            quotes = self.engine.fetch_quotes_parallel()

            # Проверить выходы
            closed = self._check_exits(quotes)
            for trade in closed:
                self._send_trade_alert(trade)

            # Обработать сигналы
            now = datetime.now(timezone.utc)

            for sig in result.signals:
                ticker = sig.secid

                # Проверить cooldown
                if ticker in self.account.cooldown_map:
                    cooldown_end = datetime.fromisoformat(self.account.cooldown_map[ticker])
                    if now < cooldown_end:
                        continue

                # Применить правила Qwen
                payload = {
                    "ticker": ticker,
                    "direction": sig.direction.value if hasattr(sig.direction, 'value') else sig.direction,
                    "horizon": sig.horizon,
                    "p": sig.probability,
                    "anomaly": {
                        "z_ret_5m": sig.z_ret_5m,
                        "z_vol_5m": sig.z_vol_5m,
                        "spread_bps": sig.spread_bps or 0,
                        "volume_spike": sig.volume_spike,
                    },
                    "market_context": {
                        "is_opening": False,
                        "ticker_liquidity": _get_ticker_liquidity(ticker),
                    },
                }

                analysis = _rule_based_analysis(payload)
                if analysis.skip:
                    continue

                # Только сильные сигналы
                if analysis.recommendation not in ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]:
                    continue

                # Фильтр Telegram (как в live)
                if analysis.recommendation not in self.config.telegram.send_recommendations:
                    continue

                # Открыть позицию
                direction = sig.direction if isinstance(sig.direction, Direction) else Direction(sig.direction)
                last_price = quotes.get(ticker, {}).get("last")

                if last_price is None:
                    continue

                opened = self._open_position(
                    ticker=ticker,
                    direction=direction,
                    price=last_price,
                    take=sig.take,
                    stop=sig.stop,
                    ttl_minutes=sig.ttl_minutes or 60,
                    horizon=sig.horizon,
                    probability=sig.probability,
                    recommendation=analysis.recommendation,
                )

                if opened:
                    self._send_signal_alert(
                        ticker=ticker,
                        direction=direction.value,
                        price=last_price,
                        horizon=sig.horizon,
                        probability=sig.probability,
                        recommendation=analysis.recommendation,
                        take=sig.take,
                        stop=sig.stop,
                    )

            # Проверить время отчёта (каждый час)
            last_report = datetime.fromisoformat(self.account.last_report_time)
            if (now - last_report).total_seconds() >= 3600:
                self._send_hourly_report()

            # Сохранить состояние
            self.account.save()

        except Exception as e:
            logger.error(f"Cycle error: {e}")

    def run(self, duration_hours: float = 168) -> None:  # 168 hours = 1 week
        """
        Запустить paper trading.

        Args:
            duration_hours: Длительность в часах (по умолчанию 168 = 1 неделя)
        """
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            logger.info("Shutdown signal received...")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=duration_hours)

        logger.info(f"Paper trading started")
        logger.info(f"Capital: {self.account.cash:,.0f} RUB")
        logger.info(f"Duration: {duration_hours} hours")
        logger.info(f"End time: {end_time.isoformat()}")

        # Отправить стартовый отчёт
        if self.config.telegram.enabled:
            send_telegram(
                bot_token=self.config.telegram.bot_token or "",
                chat_id=self.config.telegram.chat_id or "",
                text=f"🎮 СИМУЛЯЦИЯ - ВИРТУАЛЬНЫЙ СЧЁТ\n\n🚀 ТОРГОВЛЯ ЗАПУЩЕНА\n━━━━━━━━━━━━━━━━━━━━\n💰 Капитал: {self.account.cash:,.0f} ₽\n⏱ Длительность: {duration_hours} ч.",
            )

        cycle_count = 0
        while not shutdown_requested and datetime.now(timezone.utc) < end_time:
            cycle_count += 1
            self.run_cycle()

            # Лог каждые 12 циклов (~1 минута)
            if cycle_count % 12 == 0:
                logger.info(f"Cycle {cycle_count}: cash={self.account.cash:,.0f}, "
                           f"positions={len(self.account.positions)}, "
                           f"trades={len(self.account.closed_trades)}")

            time.sleep(self.config.poll_seconds)

        # Финальный отчёт
        logger.info("Paper trading stopped")
        self._send_hourly_report()

        # Закрыть ресурсы
        self.conn.close()
        close_session()

    def print_summary(self) -> None:
        """Вывести итоговую статистику."""
        print("\n" + "=" * 60)
        print("PAPER TRADING SUMMARY")
        print("=" * 60)

        print(f"\n💰 ACCOUNT")
        print(f"  Initial Capital: {self.account.initial_capital:,.0f} RUB")
        print(f"  Current Cash:    {self.account.cash:,.0f} RUB")
        print(f"  Total P&L:       {self.account.total_pnl:+,.0f} RUB ({self.account.total_pnl_pct:+.2f}%)")

        print(f"\n📊 STATISTICS")
        print(f"  Total Trades:    {len(self.account.closed_trades)}")
        print(f"  Win Rate:        {self.account.win_rate:.1f}%")
        print(f"  Open Positions:  {len(self.account.positions)}")

        if self.account.closed_trades:
            wins = [t for t in self.account.closed_trades if t.pnl > 0]
            losses = [t for t in self.account.closed_trades if t.pnl <= 0]

            if wins:
                print(f"  Avg Win:         {sum(t.pnl for t in wins)/len(wins):+,.0f} RUB")
            if losses:
                print(f"  Avg Loss:        {sum(t.pnl for t in losses)/len(losses):+,.0f} RUB")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MOEX Agent Paper Trading")
    parser.add_argument("--capital", type=float, default=200_000, help="Initial capital (RUB)")
    parser.add_argument("--duration-days", type=float, default=7, help="Duration in days")
    parser.add_argument("--position-size", type=float, default=0.05, help="Position size (fraction)")
    parser.add_argument("--max-positions", type=int, default=5, help="Max simultaneous positions")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from saved state")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")

    args = parser.parse_args()

    trader = PaperTrader(
        config_path=args.config,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        max_positions=args.max_positions,
        resume=not args.no_resume,
    )

    if args.status:
        trader.print_summary()
        return

    duration_hours = args.duration_days * 24
    trader.run(duration_hours=duration_hours)
    trader.print_summary()


if __name__ == "__main__":
    main()

"""
MOEX Agent - Конфигурация логирования с ротацией

Настройка логирования для всех модулей системы.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "data/logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
) -> None:
    """
    Настройка логирования с ротацией файлов.

    Args:
        log_dir: Директория для логов
        log_level: Уровень логирования
        max_bytes: Максимальный размер файла
        backup_count: Количество бэкапов
        console: Выводить в консоль
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Формат логов
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Очистить существующие handlers
    root_logger.handlers = []

    # Файл для всех логов (ротация по размеру)
    all_handler = RotatingFileHandler(
        log_path / "moex_agent.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    all_handler.setLevel(log_level)
    all_handler.setFormatter(formatter)
    root_logger.addHandler(all_handler)

    # Файл для ошибок (ротация по времени - ежедневно)
    error_handler = TimedRotatingFileHandler(
        log_path / "errors.log",
        when="midnight",
        interval=1,
        backupCount=30,  # Хранить 30 дней
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Файл для сделок
    trades_handler = RotatingFileHandler(
        log_path / "trades.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    trades_handler.setLevel(log_level)
    trades_handler.setFormatter(formatter)

    trades_logger = logging.getLogger("moex_agent.trades")
    trades_logger.addHandler(trades_handler)

    # Консольный вывод
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.info("Logging initialized")


class TradeLogger:
    """Специальный логгер для сделок."""

    def __init__(self):
        self.logger = logging.getLogger("moex_agent.trades")

    def log_open(
        self,
        ticker: str,
        direction: str,
        price: float,
        size: int,
        horizon: str,
        probability: float,
    ) -> None:
        """Логирование открытия позиции."""
        self.logger.info(
            f"OPEN | {ticker} | {direction} | price={price:.2f} | size={size} | "
            f"horizon={horizon} | prob={probability:.2f}"
        )

    def log_close(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: int,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Логирование закрытия позиции."""
        emoji = "✅" if pnl > 0 else "❌"
        self.logger.info(
            f"{emoji} CLOSE | {ticker} | {direction} | entry={entry_price:.2f} | "
            f"exit={exit_price:.2f} | size={size} | pnl={pnl:+.2f} ({pnl_pct:+.2f}%) | "
            f"reason={reason}"
        )

    def log_daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        pnl: float,
        equity: float,
    ) -> None:
        """Логирование дневного итога."""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        self.logger.info(
            f"DAILY | {date} | trades={trades} | wins={wins} ({win_rate:.1f}%) | "
            f"pnl={pnl:+.2f} | equity={equity:.2f}"
        )


# Глобальный экземпляр
trade_logger = TradeLogger()

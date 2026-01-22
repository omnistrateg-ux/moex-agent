"""
BCS Broker (БКС) Adapter

Адаптация риск-системы под лимиты БКС брокера.

БКС лимиты:
- КСУР (стандартный риск): плечо до 1:5
- КПУР (повышенный риск): плечо до 1:6
- Начальная маржа: 15-50% в зависимости от бумаги
- Минимальная маржа: 10-25%
- Принудительное закрытие при margin < min_margin
- Комиссия: 0.05% для активных / 0.3% для остальных
- Процент за маржинальное кредитование: ~16-20% годовых

Ставки риска (D-коэффициенты) из списка ЦБ РФ:
- SBER, GAZP, LKOH: D1=0.15, D2=0.10 (плечо ~6x)
- VTBR, MTSS: D1=0.20, D2=0.15 (плечо ~5x)
- Менее ликвидные: D1=0.30-0.50, D2=0.20-0.35

Author: MOEX Agent Risk Team
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger("moex_agent.bcs")


class BCSRiskCategory(str, Enum):
    """Категория риска клиента БКС."""
    KSUR = "КСУР"      # Стандартный уровень риска (до 1:5)
    KPUR = "КПУР"      # Повышенный уровень риска (до 1:6)
    SPECIAL = "SPEC"   # Специальные условия


class BCSSecurityTier(str, Enum):
    """Категория ликвидности бумаги."""
    TIER1 = "TIER1"    # Blue chips: SBER, GAZP, LKOH, ROSN, NVTK
    TIER2 = "TIER2"    # Ликвидные: VTBR, TATN, MTSS, CHMF, NLMK
    TIER3 = "TIER3"    # Средняя ликвидность: OZON, POSI, AFLT
    TIER4 = "TIER4"    # Низкая ликвидность: остальные


@dataclass
class BCSMarginParams:
    """Маржинальные параметры БКС для бумаги."""
    d1_long: float   # Начальная маржа Long (D-long)
    d2_long: float   # Минимальная маржа Long
    d1_short: float  # Начальная маржа Short (D-short)
    d2_short: float  # Минимальная маржа Short
    max_leverage_long: float
    max_leverage_short: float
    shortable: bool = True


# Ставки риска по тикерам (на основе списка ЦБ РФ)
BCS_MARGIN_RATES: Dict[str, BCSMarginParams] = {
    # TIER 1 - Blue Chips
    "SBER": BCSMarginParams(d1_long=0.15, d2_long=0.10, d1_short=0.15, d2_short=0.10, max_leverage_long=6.0, max_leverage_short=6.0),
    "SBERP": BCSMarginParams(d1_long=0.15, d2_long=0.10, d1_short=0.18, d2_short=0.12, max_leverage_long=6.0, max_leverage_short=5.0),
    "GAZP": BCSMarginParams(d1_long=0.15, d2_long=0.10, d1_short=0.15, d2_short=0.10, max_leverage_long=6.0, max_leverage_short=6.0),
    "LKOH": BCSMarginParams(d1_long=0.15, d2_long=0.10, d1_short=0.15, d2_short=0.10, max_leverage_long=6.0, max_leverage_short=6.0),
    "ROSN": BCSMarginParams(d1_long=0.18, d2_long=0.12, d1_short=0.18, d2_short=0.12, max_leverage_long=5.5, max_leverage_short=5.5),
    "NVTK": BCSMarginParams(d1_long=0.18, d2_long=0.12, d1_short=0.18, d2_short=0.12, max_leverage_long=5.5, max_leverage_short=5.5),
    "GMKN": BCSMarginParams(d1_long=0.18, d2_long=0.12, d1_short=0.18, d2_short=0.12, max_leverage_long=5.5, max_leverage_short=5.5),
    "PLZL": BCSMarginParams(d1_long=0.20, d2_long=0.15, d1_short=0.20, d2_short=0.15, max_leverage_long=5.0, max_leverage_short=5.0),

    # TIER 2 - Liquid
    "VTBR": BCSMarginParams(d1_long=0.20, d2_long=0.15, d1_short=0.20, d2_short=0.15, max_leverage_long=5.0, max_leverage_short=5.0),
    "TATN": BCSMarginParams(d1_long=0.20, d2_long=0.15, d1_short=0.20, d2_short=0.15, max_leverage_long=5.0, max_leverage_short=5.0),
    "TATNP": BCSMarginParams(d1_long=0.22, d2_long=0.16, d1_short=0.22, d2_short=0.16, max_leverage_long=4.5, max_leverage_short=4.5),
    "SNGS": BCSMarginParams(d1_long=0.22, d2_long=0.16, d1_short=0.22, d2_short=0.16, max_leverage_long=4.5, max_leverage_short=4.5),
    "SNGSP": BCSMarginParams(d1_long=0.22, d2_long=0.16, d1_short=0.22, d2_short=0.16, max_leverage_long=4.5, max_leverage_short=4.5),
    "MTSS": BCSMarginParams(d1_long=0.20, d2_long=0.15, d1_short=0.20, d2_short=0.15, max_leverage_long=5.0, max_leverage_short=5.0),
    "CHMF": BCSMarginParams(d1_long=0.22, d2_long=0.16, d1_short=0.22, d2_short=0.16, max_leverage_long=4.5, max_leverage_short=4.5),
    "NLMK": BCSMarginParams(d1_long=0.22, d2_long=0.16, d1_short=0.22, d2_short=0.16, max_leverage_long=4.5, max_leverage_short=4.5),
    "MAGN": BCSMarginParams(d1_long=0.25, d2_long=0.18, d1_short=0.25, d2_short=0.18, max_leverage_long=4.0, max_leverage_short=4.0),
    "ALRS": BCSMarginParams(d1_long=0.25, d2_long=0.18, d1_short=0.25, d2_short=0.18, max_leverage_long=4.0, max_leverage_short=4.0),
    "MOEX": BCSMarginParams(d1_long=0.22, d2_long=0.16, d1_short=0.22, d2_short=0.16, max_leverage_long=4.5, max_leverage_short=4.5),
    "T": BCSMarginParams(d1_long=0.25, d2_long=0.18, d1_short=0.30, d2_short=0.22, max_leverage_long=4.0, max_leverage_short=3.3),
    "YDEX": BCSMarginParams(d1_long=0.25, d2_long=0.18, d1_short=0.30, d2_short=0.22, max_leverage_long=4.0, max_leverage_short=3.3),

    # TIER 3 - Medium Liquidity
    "OZON": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "POSI": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "AFLT": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "IRAO": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "HYDR": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "FEES": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "RUAL": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "MGNT": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "X5": BCSMarginParams(d1_long=0.30, d2_long=0.22, d1_short=0.35, d2_short=0.25, max_leverage_long=3.3, max_leverage_short=2.9),
    "SIBN": BCSMarginParams(d1_long=0.25, d2_long=0.18, d1_short=0.28, d2_short=0.20, max_leverage_long=4.0, max_leverage_short=3.6),

    # TIER 4 - Low Liquidity (ограниченно/не шортится)
    "VKCO": BCSMarginParams(d1_long=0.40, d2_long=0.30, d1_short=0.50, d2_short=0.40, max_leverage_long=2.5, max_leverage_short=2.0, shortable=True),
    "SMLT": BCSMarginParams(d1_long=0.40, d2_long=0.30, d1_short=0.50, d2_short=0.40, max_leverage_long=2.5, max_leverage_short=2.0, shortable=True),
    "PIKK": BCSMarginParams(d1_long=0.40, d2_long=0.30, d1_short=0.50, d2_short=0.40, max_leverage_long=2.5, max_leverage_short=2.0, shortable=True),
    "CBOM": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "LENT": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "OKEY": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "SPBE": BCSMarginParams(d1_long=0.40, d2_long=0.30, d1_short=0.50, d2_short=0.40, max_leverage_long=2.5, max_leverage_short=2.0, shortable=True),
    "SFIN": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "SOFL": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "PHOR": BCSMarginParams(d1_long=0.35, d2_long=0.25, d1_short=0.40, d2_short=0.30, max_leverage_long=2.9, max_leverage_short=2.5),
    "TRNFP": BCSMarginParams(d1_long=0.35, d2_long=0.25, d1_short=0.40, d2_short=0.30, max_leverage_long=2.9, max_leverage_short=2.5),
    "FLOT": BCSMarginParams(d1_long=0.40, d2_long=0.30, d1_short=0.50, d2_short=0.40, max_leverage_long=2.5, max_leverage_short=2.0),
    "HEAD": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "ENPG": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
    "RSTI": BCSMarginParams(d1_long=0.50, d2_long=0.40, d1_short=1.00, d2_short=0.80, max_leverage_long=2.0, max_leverage_short=1.0, shortable=False),
}

# Дефолтные значения для неизвестных бумаг (консервативные)
DEFAULT_MARGIN_PARAMS = BCSMarginParams(
    d1_long=0.50,
    d2_long=0.40,
    d1_short=1.00,
    d2_short=0.80,
    max_leverage_long=2.0,
    max_leverage_short=1.0,
    shortable=False,
)


@dataclass
class BCSConfig:
    """Конфигурация БКС брокера."""
    risk_category: BCSRiskCategory = BCSRiskCategory.KSUR

    # Комиссии
    commission_active: float = 0.0005     # 0.05% для активных клиентов
    commission_standard: float = 0.003    # 0.3% для остальных
    min_commission: float = 0.01          # Минимум 1 копейка

    # Процент за маржинальное кредитование (годовые)
    margin_interest_long: float = 0.16    # 16% годовых за лонг
    margin_interest_short: float = 0.18   # 18% годовых за шорт

    # Лимиты
    max_leverage_ksur: float = 5.0        # Макс плечо для КСУР
    max_leverage_kpur: float = 6.0        # Макс плечо для КПУР

    # Время закрытия маржин-колла (минуты)
    margin_call_timeout: int = 30

    # Буфер безопасности (выше min_margin на этот %)
    safety_buffer_pct: float = 5.0

    # Торговые часы MOEX
    trading_start_hour: int = 10  # 10:00 MSK
    trading_end_hour: int = 19    # 18:50 MSK (округлено)

    # T+ режим
    settlement_days: int = 2      # T+2


def get_margin_params(ticker: str) -> BCSMarginParams:
    """Получить маржинальные параметры для тикера."""
    return BCS_MARGIN_RATES.get(ticker, DEFAULT_MARGIN_PARAMS)


def get_security_tier(ticker: str) -> BCSSecurityTier:
    """Определить категорию ликвидности бумаги."""
    params = get_margin_params(ticker)

    if params.d1_long <= 0.18:
        return BCSSecurityTier.TIER1
    elif params.d1_long <= 0.25:
        return BCSSecurityTier.TIER2
    elif params.d1_long <= 0.35:
        return BCSSecurityTier.TIER3
    else:
        return BCSSecurityTier.TIER4


def calculate_bcs_leverage(
    ticker: str,
    direction: str,
    risk_category: BCSRiskCategory = BCSRiskCategory.KSUR,
) -> float:
    """
    Рассчитать максимально допустимое плечо БКС для бумаги.

    Returns:
        Максимальное плечо с учётом ограничений БКС
    """
    params = get_margin_params(ticker)

    if direction == "SHORT" and not params.shortable:
        return 0.0  # Шорт запрещён

    # Базовое плечо из ставок риска
    if direction == "LONG":
        base_leverage = params.max_leverage_long
    else:
        base_leverage = params.max_leverage_short

    # Ограничение по категории клиента
    if risk_category == BCSRiskCategory.KSUR:
        category_limit = 5.0
    elif risk_category == BCSRiskCategory.KPUR:
        category_limit = 6.0
    else:
        category_limit = 10.0

    return min(base_leverage, category_limit)


def calculate_margin_required(
    ticker: str,
    direction: str,
    position_value: float,
    leverage: float,
) -> float:
    """
    Рассчитать требуемую маржу для позиции.

    Returns:
        Сумма маржи в рублях
    """
    params = get_margin_params(ticker)

    if direction == "LONG":
        d1 = params.d1_long
    else:
        d1 = params.d1_short

    # Маржа = notional * D1
    # При плече: position_value = equity * leverage
    # equity = position_value / leverage
    # margin = position_value * D1 / leverage = equity * D1

    notional = position_value * leverage
    margin = notional * d1

    return margin


def check_margin_call(
    ticker: str,
    direction: str,
    entry_price: float,
    current_price: float,
    position_value: float,
    leverage: float,
    equity: float,
) -> tuple[bool, float, float]:
    """
    Проверить условия маржин-колла.

    Returns:
        (is_margin_call, current_margin_pct, min_margin_pct)
    """
    params = get_margin_params(ticker)

    if direction == "LONG":
        d2 = params.d2_long
        pnl = (current_price - entry_price) / entry_price
    else:
        d2 = params.d2_short
        pnl = (entry_price - current_price) / entry_price

    # Текущий equity с учётом PnL
    notional = position_value * leverage
    current_equity = equity + (pnl * notional)

    # Текущая маржа в процентах
    if notional > 0:
        current_margin_pct = current_equity / notional
    else:
        current_margin_pct = 1.0

    # Маржин-колл если текущая маржа < минимальной
    is_margin_call = current_margin_pct < d2

    return is_margin_call, current_margin_pct, d2


def calculate_forced_liquidation_price(
    ticker: str,
    direction: str,
    entry_price: float,
    leverage: float,
) -> float:
    """
    Рассчитать цену принудительной ликвидации.

    При этой цене маржа падает до минимальной (D2).
    """
    params = get_margin_params(ticker)

    if direction == "LONG":
        d1 = params.d1_long
        d2 = params.d2_long
        # PnL% при котором маржа = d2
        # equity = initial_equity + pnl
        # initial_margin = notional * d1
        # equity / notional = d2
        # (initial_equity + pnl) / notional = d2
        # initial_equity / notional + pnl / notional = d2
        # d1 + pnl% = d2
        # pnl% = d2 - d1 (отрицательное значение)
        pnl_pct = d2 - d1
        liquidation_price = entry_price * (1 + pnl_pct / leverage)
    else:
        d1 = params.d1_short
        d2 = params.d2_short
        pnl_pct = d2 - d1
        liquidation_price = entry_price * (1 - pnl_pct / leverage)

    return liquidation_price


class BCSRiskAdapter:
    """
    Адаптер для интеграции лимитов БКС с MarginRiskEngine.
    """

    def __init__(self, config: Optional[BCSConfig] = None):
        self.config = config or BCSConfig()

    def validate_trade(
        self,
        ticker: str,
        direction: str,
        requested_leverage: float,
    ) -> tuple[bool, float, str]:
        """
        Валидация сделки по правилам БКС.

        Returns:
            (is_allowed, adjusted_leverage, reason)
        """
        params = get_margin_params(ticker)

        # Проверка шорта
        if direction == "SHORT" and not params.shortable:
            return False, 0.0, f"{ticker} не доступен для шорта"

        # Максимальное плечо БКС
        max_bcs_leverage = calculate_bcs_leverage(
            ticker, direction, self.config.risk_category
        )

        if requested_leverage > max_bcs_leverage:
            return True, max_bcs_leverage, f"Плечо снижено до {max_bcs_leverage:.1f}x (лимит БКС)"

        return True, requested_leverage, "OK"

    def calculate_safe_position_size(
        self,
        ticker: str,
        direction: str,
        price: float,
        equity: float,
        leverage: float,
        max_loss_pct: float = 0.5,
    ) -> tuple[int, float, float]:
        """
        Рассчитать безопасный размер позиции с учётом БКС лимитов.

        Returns:
            (shares, margin_required, liquidation_price)
        """
        params = get_margin_params(ticker)

        # Скорректировать плечо
        _, adjusted_leverage, _ = self.validate_trade(ticker, direction, leverage)

        if adjusted_leverage == 0:
            return 0, 0.0, 0.0

        # Маржинальные требования
        if direction == "LONG":
            d1 = params.d1_long
        else:
            d1 = params.d1_short

        # Безопасный буфер над минимальной маржой
        safety_margin = d1 * (1 + self.config.safety_buffer_pct / 100)

        # Максимальный notional исходя из equity и маржи
        max_notional = equity / safety_margin

        # Максимальная позиция
        max_position_value = max_notional / adjusted_leverage
        shares = int(max_position_value / price)

        # Но также ограничиваем по max_loss
        # max_loss = equity * max_loss_pct / 100
        # Если стоп = liquidation_price, то loss = ...

        # Рассчитать цену ликвидации
        liquidation_price = calculate_forced_liquidation_price(
            ticker, direction, price, adjusted_leverage
        )

        # Margin required
        position_value = shares * price
        margin_required = position_value * d1

        return shares, margin_required, liquidation_price

    def get_ticker_info(self, ticker: str) -> dict:
        """Получить информацию о тикере для БКС."""
        params = get_margin_params(ticker)
        tier = get_security_tier(ticker)

        return {
            "ticker": ticker,
            "tier": tier.value,
            "d1_long": params.d1_long,
            "d2_long": params.d2_long,
            "d1_short": params.d1_short,
            "d2_short": params.d2_short,
            "max_leverage_long": params.max_leverage_long,
            "max_leverage_short": params.max_leverage_short,
            "shortable": params.shortable,
        }


def print_bcs_limits():
    """Вывести лимиты БКС для всех бумаг."""
    print("=" * 80)
    print("БКС БРОКЕР - МАРЖИНАЛЬНЫЕ ЛИМИТЫ")
    print("=" * 80)
    print(f"{'Тикер':<8} {'Tier':<6} {'D1 Long':<8} {'D2 Long':<8} {'Max Lev':<8} {'Short':<6}")
    print("-" * 80)

    for ticker, params in sorted(BCS_MARGIN_RATES.items()):
        tier = get_security_tier(ticker)
        short_str = "✅" if params.shortable else "❌"
        print(
            f"{ticker:<8} {tier.value:<6} "
            f"{params.d1_long:>6.0%}   {params.d2_long:>6.0%}   "
            f"{params.max_leverage_long:>6.1f}x  {short_str}"
        )


if __name__ == "__main__":
    print_bcs_limits()

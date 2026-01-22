from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskParams:
    max_spread_bps: float
    min_turnover_rub_5m: float


def spread_bps(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 10000


def pass_gatekeeper(
    p: float,
    p_threshold: float,
    turnover_5m: float,
    spread: Optional[float],
    risk: RiskParams,
) -> bool:
    if p < p_threshold:
        return False
    if turnover_5m < risk.min_turnover_rub_5m:
        return False
    if spread is not None and spread > risk.max_spread_bps:
        return False
    return True

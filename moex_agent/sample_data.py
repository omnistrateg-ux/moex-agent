"""Deterministic sample market scenario for MOEX simulation."""
from __future__ import annotations


def build_sample_market_data() -> dict:
    """Return deterministic MOEX-like market data payload."""
    return {
        "ticker": "SBER",
        "timeframe": "5m",
        "candles": [
            {"ts": "2026-01-01T10:00:00+03:00", "open": 280.0, "high": 281.2, "low": 279.8, "close": 281.0},
            {"ts": "2026-01-01T10:05:00+03:00", "open": 281.0, "high": 281.6, "low": 280.7, "close": 281.4},
            {"ts": "2026-01-01T10:10:00+03:00", "open": 281.4, "high": 282.0, "low": 281.1, "close": 281.8},
        ],
        "volume": 1250000,
        "spread_bps": 8.0,
        "session": "main",
    }


def build_sample_context() -> dict:
    """Return deterministic context metadata."""
    return {
        "scenario": "sample_moex_v1",
        "fee_bps": 8.0,
        "market": "MOEX",
    }

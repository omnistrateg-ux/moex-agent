"""Real-bound feature adapter using project feature engineering module."""

from __future__ import annotations

import pandas as pd

from .common import AdapterCompatibilityError, require_symbol


def get_real_adapter_diagnostics(config: dict | None = None) -> dict:
    missing: list[str] = []
    for module_name, symbol_name in [
        ("moex_agent.features", "build_feature_frame"),
    ]:
        try:
            require_symbol(module_name, symbol_name)
        except AdapterCompatibilityError as exc:
            missing.append(str(exc))
    return {"available": not missing, "details": missing}


def run_agent(config: dict, inputs: dict) -> dict:
    slug = config.get("slug", "unknown")
    diagnostics = get_real_adapter_diagnostics(config)
    if not diagnostics["available"]:
        raise AdapterCompatibilityError("; ".join(diagnostics["details"]))

    build_feature_frame = require_symbol("moex_agent.features", "build_feature_frame")

    artifacts = (inputs or {}).get("artifacts", {})
    market_data = artifacts.get("market_data")
    if not isinstance(market_data, dict):
        return {"status": "error", "agent": slug, "message": "market_data is required"}

    candles = market_data.get("candles") or []
    ticker = str(market_data.get("ticker", "SBER"))
    if not candles:
        return {"status": "error", "agent": slug, "message": "market_data.candles is required"}

    rows = []
    for idx, c in enumerate(candles):
        rows.append(
            {
                "secid": ticker,
                "ts": c.get("ts", f"2025-01-01T10:{idx:02d}:00Z"),
                "open": float(c.get("open", 0.0)),
                "high": float(c.get("high", 0.0)),
                "low": float(c.get("low", 0.0)),
                "close": float(c.get("close", 0.0)),
                "value": float(c.get("value", c.get("volume", 0.0))),
                "volume": float(c.get("volume", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    feat_df = build_feature_frame(df)
    if feat_df.empty:
        return {"status": "error", "agent": slug, "message": "feature frame is empty"}

    latest = feat_df.tail(1).to_dict(orient="records")[0]
    features = {
        "version": "1.0",
        "secid": latest.get("secid", ticker),
        "r_1m": float(latest.get("r_1m", 0.0)),
        "rsi_14": float(latest.get("rsi_14", 0.0)),
        "volatility_10": float(latest.get("volatility_10", 0.0)),
    }
    return {"status": "ok", "agent": slug, "provides": {"features": features}}

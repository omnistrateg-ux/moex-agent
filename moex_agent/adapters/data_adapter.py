"""Real-bound data adapter with deterministic fallback input shaping."""

from __future__ import annotations

from .common import AdapterCompatibilityError, require_symbol


def get_real_adapter_diagnostics(config: dict | None = None) -> dict:
    checks = [
        ("moex_agent.risk", "spread_bps"),
        ("moex_agent.sample_data", "build_sample_market_data"),
    ]
    missing: list[str] = []
    for module_name, symbol_name in checks:
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

    spread_bps_fn = require_symbol("moex_agent.risk", "spread_bps")
    build_sample_market_data = require_symbol("moex_agent.sample_data", "build_sample_market_data")

    artifacts = (inputs or {}).get("artifacts", {})
    market_data = artifacts.get("market_data")
    if not isinstance(market_data, dict):
        # deterministic baseline from project sample-data module
        market_data = build_sample_market_data()

    normalized = dict(market_data)
    quote = normalized.get("quote") if isinstance(normalized.get("quote"), dict) else {}
    bid = quote.get("bid") if isinstance(quote, dict) else None
    ask = quote.get("ask") if isinstance(quote, dict) else None
    calc_spread = spread_bps_fn(bid, ask)
    if calc_spread is not None:
        normalized["spread_bps"] = float(calc_spread)
    elif "spread_bps" not in normalized:
        normalized["spread_bps"] = 0.0

    return {"status": "ok", "agent": slug, "provides": {"market_data": normalized}}

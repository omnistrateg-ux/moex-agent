"""Real-bound risk adapter using project risk module."""

from __future__ import annotations

from .common import AdapterCompatibilityError, require_symbol


def get_real_adapter_diagnostics(config: dict | None = None) -> dict:
    checks = [
        ("moex_agent.risk", "RiskParams"),
        ("moex_agent.risk", "pass_gatekeeper"),
        ("moex_agent.risk", "spread_bps"),
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

    RiskParams = require_symbol("moex_agent.risk", "RiskParams")
    pass_gatekeeper = require_symbol("moex_agent.risk", "pass_gatekeeper")
    spread_bps_fn = require_symbol("moex_agent.risk", "spread_bps")

    artifacts = (inputs or {}).get("artifacts", {})
    market_data = artifacts.get("market_data")
    signals = artifacts.get("signals")
    if not isinstance(market_data, dict) or not isinstance(signals, dict):
        return {"status": "error", "agent": slug, "message": "market_data and signals are required"}

    quote = market_data.get("quote") if isinstance(market_data.get("quote"), dict) else {}
    spread = spread_bps_fn(quote.get("bid"), quote.get("ask"))
    if spread is None:
        spread = float(market_data.get("spread_bps", 0.0))

    p_long = float(signals.get("probability_long", 0.0))
    turnover = float(market_data.get("volume", 0.0))

    risk_cfg = config.get("risk", {}) if isinstance(config.get("risk"), dict) else {}
    params = RiskParams(
        max_spread_bps=float(risk_cfg.get("max_spread_bps", 30.0)),
        min_turnover_rub_5m=float(risk_cfg.get("min_turnover_rub_5m", 100000.0)),
    )
    allowed = pass_gatekeeper(
        p=p_long,
        p_threshold=float(config.get("p_threshold", 0.3)),
        turnover_5m=turnover,
        spread=spread,
        risk=params,
    )

    risk_state = {
        "kill_switch": not bool(allowed),
        "max_drawdown": 0.0,
        "position_limit": 1.0 if allowed else 0.0,
    }
    return {"status": "ok", "agent": slug, "provides": {"risk_state": risk_state}}

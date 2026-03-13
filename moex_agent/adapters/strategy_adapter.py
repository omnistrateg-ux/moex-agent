"""Real-bound strategy adapter using project risk gate logic."""

from __future__ import annotations

from .common import AdapterCompatibilityError, require_symbol


_DIAG_OK = "OK"
_DIAG_ERROR = "ERROR"


def get_real_adapter_diagnostics(config: dict | None = None) -> list[tuple[str, str]]:
    checks = [
        ("moex_agent.risk", "RiskParams"),
        ("moex_agent.risk", "pass_gatekeeper"),
    ]
    out: list[tuple[str, str]] = []
    for module_name, symbol_name in checks:
        try:
            require_symbol(module_name, symbol_name)
            out.append((_DIAG_OK, f"{module_name}.{symbol_name} available"))
        except AdapterCompatibilityError as exc:
            out.append((_DIAG_ERROR, str(exc)))
    return out


def run_agent(config: dict, inputs: dict) -> dict:
    slug = config.get("slug", "unknown")
    diag = get_real_adapter_diagnostics(config)
    errors = [m for lvl, m in diag if lvl == _DIAG_ERROR]
    if errors:
        raise AdapterCompatibilityError("; ".join(errors))

    RiskParams = require_symbol("moex_agent.risk", "RiskParams")
    pass_gatekeeper = require_symbol("moex_agent.risk", "pass_gatekeeper")

    artifacts = (inputs or {}).get("artifacts", {})
    signals = artifacts.get("signals")
    risk_state = artifacts.get("risk_state")
    market_data = artifacts.get("market_data")
    if not isinstance(signals, dict) or not isinstance(risk_state, dict):
        return {"status": "error", "agent": slug, "message": "signals and risk_state are required"}

    if bool(risk_state.get("kill_switch", False)):
        plan = {"action": "HOLD", "reason": "risk kill-switch"}
        return {"status": "ok", "agent": slug, "provides": {"trade_plan": plan}}

    turnover = 0.0
    spread = None
    if isinstance(market_data, dict):
        turnover = float(market_data.get("volume", 0.0))
        spread = market_data.get("spread_bps")
        spread = float(spread) if isinstance(spread, (int, float)) else None

    params = RiskParams(
        max_spread_bps=float(config.get("max_spread_bps", 50.0)),
        min_turnover_rub_5m=float(config.get("min_turnover_rub_5m", 0.0)),
    )
    p_long = float(signals.get("probability_long", 0.5))
    p_short = float(signals.get("probability_short", 0.5))
    p_use = max(p_long, p_short)
    allowed = pass_gatekeeper(
        p=p_use,
        p_threshold=float(config.get("p_threshold", 0.3)),
        turnover_5m=turnover,
        spread=spread,
        risk=params,
    )

    action = "HOLD"
    if allowed:
        action = "BUY" if p_long >= p_short else "SELL"

    plan = {"action": action, "reason": "real-strategy", "confidence": float(signals.get("confidence", 0.0))}
    return {"status": "ok", "agent": slug, "provides": {"trade_plan": plan}}

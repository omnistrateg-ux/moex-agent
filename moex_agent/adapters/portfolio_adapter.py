"""Real-bound portfolio adapter using project risk manager logic."""

from __future__ import annotations

from .common import AdapterCompatibilityError, require_symbol


_DIAG_OK = "OK"
_DIAG_ERROR = "ERROR"


def get_real_adapter_diagnostics(config: dict | None = None) -> list[tuple[str, str]]:
    checks = [
        ("moex_agent.risk_manager", "RiskManager"),
        ("moex_agent.risk_manager", "RiskLimits"),
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

    RiskManager = require_symbol("moex_agent.risk_manager", "RiskManager")
    RiskLimits = require_symbol("moex_agent.risk_manager", "RiskLimits")

    artifacts = (inputs or {}).get("artifacts", {})
    plan = artifacts.get("trade_plan")
    risk_state = artifacts.get("risk_state")
    if not isinstance(plan, dict) or not isinstance(risk_state, dict):
        return {"status": "error", "agent": slug, "message": "trade_plan and risk_state are required"}

    limits = RiskLimits()
    manager = RiskManager(limits=limits)
    multiplier = manager.get_position_size_multiplier(equity=1_000_000.0, initial_capital=1_000_000.0)

    action = str(plan.get("action", "HOLD"))
    risk_budget = 0.0
    if action in {"BUY", "SELL"} and not bool(risk_state.get("kill_switch", False)):
        risk_budget = min(1.0, max(0.0, float(multiplier)))

    allocation = {
        "cash": round(1.0 - risk_budget, 6),
        "risk_budget": round(risk_budget, 6),
        "action": action,
    }
    return {"status": "ok", "agent": slug, "provides": {"allocation": allocation}}

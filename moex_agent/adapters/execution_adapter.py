"""Real-bound execution adapter (paper-safe only, no real trade placement)."""

from __future__ import annotations

from .common import AdapterCompatibilityError, require_symbol


_DIAG_OK = "OK"
_DIAG_ERROR = "ERROR"


def get_real_adapter_diagnostics(config: dict | None = None) -> list[tuple[str, str]]:
    checks = [
        ("moex_agent.risk_manager", "RiskManager"),
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

    require_symbol("moex_agent.risk_manager", "RiskManager")

    artifacts = (inputs or {}).get("artifacts", {})
    allocation = artifacts.get("allocation")
    if not isinstance(allocation, dict):
        return {"status": "error", "agent": slug, "message": "allocation is required"}

    risk_budget = float(allocation.get("risk_budget", 0.0))
    qty = 1 if risk_budget > 0 else 0
    report = {
        "status": "PAPER_SIMULATED",
        "paper": True,
        "fills": [{"qty": qty, "action": allocation.get("action", "HOLD")}],
    }
    return {"status": "ok", "agent": slug, "provides": {"execution_report": report}}

"""Real-bound monitoring adapter using project semantic contract validator."""

from __future__ import annotations

from .common import AdapterCompatibilityError, require_symbol


def get_real_adapter_diagnostics(config: dict | None = None) -> dict:
    missing: list[str] = []
    try:
        require_symbol("moex_agent.semantic_contracts", "validate_semantic_output")
    except AdapterCompatibilityError as exc:
        missing.append(str(exc))
    return {"available": not missing, "details": missing}


def run_agent(config: dict, inputs: dict) -> dict:
    slug = config.get("slug", "unknown")
    diagnostics = get_real_adapter_diagnostics(config)
    if not diagnostics["available"]:
        raise AdapterCompatibilityError("; ".join(diagnostics["details"]))

    validate_semantic_output = require_symbol("moex_agent.semantic_contracts", "validate_semantic_output")

    artifacts = (inputs or {}).get("artifacts", {})
    required = ["market_data", "signals", "risk_state", "execution_report"]
    missing = [k for k in required if k not in artifacts]
    if missing:
        return {"status": "error", "agent": slug, "message": f"missing artifacts: {', '.join(missing)}"}

    # verify upstream semantic payloads that monitoring depends on
    validate_semantic_output("ml", {"signals": artifacts["signals"]})
    validate_semantic_output("risk", {"risk_state": artifacts["risk_state"]})
    validate_semantic_output("execution", {"execution_report": artifacts["execution_report"]})

    alerts = {
        "level": "INFO",
        "messages": ["runtime compatibility checks passed"],
    }
    return {"status": "ok", "agent": slug, "provides": {"alerts": alerts}}

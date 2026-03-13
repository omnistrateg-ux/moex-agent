"""Semantic runtime contracts for MOEX agent outputs."""
from __future__ import annotations


class SemanticContractError(ValueError):
    """Raised when semantic output contract is violated."""


def _require_key(container: dict, key: str, agent_type: str) -> None:
    if key not in container:
        raise SemanticContractError(f"{agent_type}: missing semantic key '{key}'")


def _require_nested_keys(container: dict, key: str, required: list[str], agent_type: str) -> None:
    _require_key(container, key, agent_type)
    value = container[key]
    if not isinstance(value, dict):
        raise SemanticContractError(f"{agent_type}: '{key}' must be dict")
    for nested in required:
        if nested not in value:
            raise SemanticContractError(f"{agent_type}: '{key}' missing '{nested}'")


def validate_semantic_output(agent_type: str, provides: dict) -> None:
    """Validate semantic output structure by agent type."""
    if not isinstance(provides, dict):
        raise SemanticContractError(f"{agent_type}: provides must be dict")

    agent_type = (agent_type or "").strip().lower()

    if agent_type == "data":
        _require_key(provides, "market_data", agent_type)
    elif agent_type == "feature":
        _require_key(provides, "features", agent_type)
    elif agent_type == "ml":
        _require_nested_keys(
            provides,
            "signals",
            ["probability_long", "probability_short", "confidence"],
            agent_type,
        )
    elif agent_type == "risk":
        _require_nested_keys(
            provides,
            "risk_state",
            ["kill_switch", "max_drawdown", "position_limit"],
            agent_type,
        )
    elif agent_type == "strategy":
        _require_key(provides, "trade_plan", agent_type)
    elif agent_type == "portfolio":
        _require_key(provides, "allocation", agent_type)
    elif agent_type == "execution":
        _require_key(provides, "execution_report", agent_type)
    elif agent_type == "monitoring":
        _require_key(provides, "alerts", agent_type)
    else:
        raise SemanticContractError(f"unknown agent_type: {agent_type}")

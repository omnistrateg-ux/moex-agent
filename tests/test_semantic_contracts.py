import subprocess
import sys
from pathlib import Path

import pytest

from moex_agent.semantic_contracts import SemanticContractError, validate_semantic_output


def _create_system(tmp_path: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(tmp_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return tmp_path / "registry.json"


def test_each_core_agent_type_passes_semantic_validation():
    validate_semantic_output("data", {"market_data": {}})
    validate_semantic_output("feature", {"features": {}})
    validate_semantic_output("ml", {"signals": {"probability_long": 0.5, "probability_short": 0.5, "confidence": 0.0}})
    validate_semantic_output("risk", {"risk_state": {"kill_switch": False, "max_drawdown": 0.0, "position_limit": 0.0}})
    validate_semantic_output("strategy", {"trade_plan": {}})
    validate_semantic_output("portfolio", {"allocation": {}})
    validate_semantic_output("execution", {"execution_report": {}})
    validate_semantic_output("monitoring", {"alerts": {}})


def test_invalid_ml_output_fails():
    with pytest.raises(SemanticContractError):
        validate_semantic_output("ml", {"signals": {"probability_long": 0.5}})


def test_invalid_risk_output_fails():
    with pytest.raises(SemanticContractError):
        validate_semantic_output("risk", {"risk_state": {"kill_switch": False}})


def test_orchestrate_simulate_ready_for_generated_core_system(tmp_path: Path):
    registry = _create_system(tmp_path)
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "orchestrate",
        "--simulate",
        "--registry",
        str(registry),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "STATUS: READY" in res.stdout

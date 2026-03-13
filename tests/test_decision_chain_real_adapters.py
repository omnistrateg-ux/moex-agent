from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from moex_agent.adapters.common import AdapterCompatibilityError
from moex_agent.adapters.execution_adapter import run_agent as run_execution_adapter
from moex_agent.adapters.ml_adapter import run_agent as run_ml_adapter
from moex_agent.adapters.portfolio_adapter import run_agent as run_portfolio_adapter
from moex_agent.adapters.strategy_adapter import run_agent as run_strategy_adapter
from moex_agent.orchestrator_loader import load_system_registry
from moex_agent.runtime_contracts import run_execution_plan
from moex_agent.semantic_contracts import validate_semantic_output


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


def _set_adapter(cfg_path: Path, module: str):
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["runtime_mode"] = "real"
    payload["runtime_adapter"] = {"module": module, "callable": "run_agent"}
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_ml_real_adapter_works_when_dependency_available() -> None:
    features = {"r_1m": 0.01, "momentum": 0.2, "rsi_14": 55.0}
    result = run_ml_adapter({"slug": "signal-model"}, {"artifacts": {"features": features}})
    assert result["status"] == "ok"
    validate_semantic_output("ml", result["provides"])


def test_strategy_portfolio_execution_real_adapters_semantic_valid() -> None:
    signals = {"probability_long": 0.7, "probability_short": 0.2, "confidence": 0.5}
    risk_state = {"kill_switch": False, "max_drawdown": 0.0, "position_limit": 1.0}

    strategy = run_strategy_adapter({"slug": "strategy-planner"}, {"artifacts": {"signals": signals, "risk_state": risk_state}})
    assert strategy["status"] == "ok"
    validate_semantic_output("strategy", strategy["provides"])

    portfolio = run_portfolio_adapter(
        {"slug": "portfolio-allocator"},
        {"artifacts": {"trade_plan": strategy["provides"]["trade_plan"], "risk_state": risk_state}},
    )
    assert portfolio["status"] == "ok"
    validate_semantic_output("portfolio", portfolio["provides"])

    execution = run_execution_adapter(
        {"slug": "execution-router"},
        {"artifacts": {"allocation": portfolio["provides"]["allocation"]}},
    )
    assert execution["status"] == "ok"
    validate_semantic_output("execution", execution["provides"])


def test_missing_project_dependency_fails_clearly() -> None:
    with patch("moex_agent.adapters.common.importlib.import_module", side_effect=ImportError("missing")):
        with pytest.raises(AdapterCompatibilityError, match="missing adapter dependency module"):
            run_strategy_adapter({"slug": "strategy-planner"}, {"artifacts": {"signals": {}, "risk_state": {}}})


def test_stub_mode_remains_unchanged(tmp_path: Path) -> None:
    registry = _create_system(tmp_path)
    system = load_system_registry(registry)
    results = run_execution_plan(system)
    assert len(results) == 8
    assert all(r.get("status") == "ok" for r in results)


def test_full_real_mode_simulate_reaches_execution_report(tmp_path: Path) -> None:
    registry = _create_system(tmp_path)
    adapter_map = {
        "data-collector": "moex_agent.adapters.data_adapter",
        "feature-builder": "moex_agent.adapters.feature_adapter",
        "signal-model": "moex_agent.adapters.ml_adapter",
        "risk-guard": "moex_agent.adapters.risk_adapter",
        "strategy-planner": "moex_agent.adapters.strategy_adapter",
        "portfolio-allocator": "moex_agent.adapters.portfolio_adapter",
        "execution-router": "moex_agent.adapters.execution_adapter",
        "system-monitor": "moex_agent.adapters.monitoring_adapter",
    }
    for slug, module in adapter_map.items():
        _set_adapter(tmp_path / slug / "agent_config.json", module)

    system = load_system_registry(registry)
    results = run_execution_plan(system, runtime_mode_override="real", with_sample_data=True)
    assert any("execution_report" in (r.get("provides") or {}) for r in results)

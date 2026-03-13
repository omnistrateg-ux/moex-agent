import json
import subprocess
import sys
from pathlib import Path

import pytest

from moex_agent.orchestrator_loader import load_system_registry
from moex_agent.runtime_contracts import AgentRuntimeError, run_execution_plan


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
    payload["runtime_adapter"] = {"module": module, "callable": "run_agent"}
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_default_generated_system_runs_stub_mode(tmp_path: Path):
    registry = _create_system(tmp_path)
    system = load_system_registry(registry)
    results = run_execution_plan(system)
    assert len(results) == 8


def test_overriding_real_mode_works_for_agents_with_adapters(tmp_path: Path):
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
    for slug, mod in adapter_map.items():
        _set_adapter(tmp_path / slug / "agent_config.json", mod)

    system = load_system_registry(registry)
    results = run_execution_plan(system, runtime_mode_override="real")
    assert len(results) == 8


def test_invalid_adapter_config_fails_clearly(tmp_path: Path):
    registry = _create_system(tmp_path)
    cfg = tmp_path / "data-collector" / "agent_config.json"
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["runtime_adapter"] = {"module": "moex_agent.adapters.data_adapter", "callable": "missing_fn"}
    cfg.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    system = load_system_registry(registry)
    with pytest.raises(AgentRuntimeError):
        run_execution_plan(system, runtime_mode_override="real")


def test_adapter_output_passes_semantic_validation(tmp_path: Path):
    registry = _create_system(tmp_path)
    _set_adapter(tmp_path / "data-collector" / "agent_config.json", "moex_agent.adapters.data_adapter")
    _set_adapter(tmp_path / "feature-builder" / "agent_config.json", "moex_agent.adapters.feature_adapter")
    _set_adapter(tmp_path / "risk-guard" / "agent_config.json", "moex_agent.adapters.risk_adapter")
    _set_adapter(tmp_path / "system-monitor" / "agent_config.json", "moex_agent.adapters.monitoring_adapter")

    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "orchestrate",
        "--simulate",
        "--runtime-mode",
        "real",
        "--registry",
        str(registry),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0  # not all agents have proper adapters yet

    # now assign all adapters and ensure READY
    _set_adapter(tmp_path / "signal-model" / "agent_config.json", "moex_agent.adapters.ml_adapter")
    _set_adapter(tmp_path / "strategy-planner" / "agent_config.json", "moex_agent.adapters.strategy_adapter")
    _set_adapter(tmp_path / "portfolio-allocator" / "agent_config.json", "moex_agent.adapters.portfolio_adapter")
    _set_adapter(tmp_path / "execution-router" / "agent_config.json", "moex_agent.adapters.execution_adapter")
    res2 = subprocess.run(cmd, capture_output=True, text=True)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert "STATUS: READY" in res2.stdout

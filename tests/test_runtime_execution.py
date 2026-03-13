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


def test_full_system_runtime_executes_deterministically(tmp_path: Path):
    registry = _create_system(tmp_path)
    system = load_system_registry(registry)
    results = run_execution_plan(system)
    slugs = [r["agent"] for r in results]
    assert slugs == [
        "data-collector",
        "feature-builder",
        "signal-model",
        "risk-guard",
        "strategy-planner",
        "portfolio-allocator",
        "execution-router",
        "system-monitor",
    ]


def test_artifact_store_accumulates_expected_provides(tmp_path: Path):
    registry = _create_system(tmp_path)
    system = load_system_registry(registry)
    results = run_execution_plan(system)

    provided = set()
    for item in results:
        provided.update((item.get("provides") or {}).keys())

    assert provided == {
        "market_data",
        "features",
        "signals",
        "risk_state",
        "trade_plan",
        "allocation",
        "execution_report",
        "alerts",
    }


def test_invalid_runtime_result_raises(tmp_path: Path):
    registry = _create_system(tmp_path)
    # break one runtime: remove 'provides' from ok result
    src = tmp_path / "data-collector" / "src" / "data-collector.py"
    text = src.read_text(encoding="utf-8")
    text = text.replace('return {"status": "ok", "agent": "data-collector", "provides": provides}',
                        'return {"status": "ok", "agent": "data-collector"}')
    src.write_text(text, encoding="utf-8")

    system = load_system_registry(registry)
    with pytest.raises(AgentRuntimeError):
        run_execution_plan(system)


def test_orchestrate_simulate_cli_ready(tmp_path: Path):
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

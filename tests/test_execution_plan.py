import json
import subprocess
import sys
from pathlib import Path

import pytest

from moex_agent.orchestrator_loader import (
    LoadedAgent,
    SystemRegistry,
    SystemRegistryError,
    build_execution_plan,
    load_system_registry,
)


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


def test_execution_plan_for_full_system_is_deterministic(tmp_path: Path):
    registry = _create_system(tmp_path)
    system = load_system_registry(registry)
    plan = build_execution_plan(system)
    slugs = [a.slug for a in plan]
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


def test_execution_plan_fails_for_incomplete_system(tmp_path: Path):
    registry = _create_system(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["agents"] = [a for a in payload["agents"] if a["agent_type"] != "monitoring"]
    registry.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    system = load_system_registry(registry)
    with pytest.raises(SystemRegistryError):
        build_execution_plan(system)


def test_execution_plan_detects_cycle():
    a = LoadedAgent(
        name="A",
        slug="a",
        agent_type="data",
        role="a",
        timeframe="5m",
        config_path=Path("a.json"),
        files=["a"],
        dependencies_requires=["yb"],
        dependencies_provides=["xa"],
    )
    b = LoadedAgent(
        name="B",
        slug="b",
        agent_type="feature",
        role="b",
        timeframe="5m",
        config_path=Path("b.json"),
        files=["b"],
        dependencies_requires=["xa"],
        dependencies_provides=["yb"],
    )
    system = SystemRegistry(
        schema_version="1.0",
        preset="manual",
        generated_at="static",
        agents=[a, b],
        agents_by_type={
            "data": [a],
            "feature": [b],
            "ml": [LoadedAgent("m","m","ml","r","5m",Path("m"),["m"],["none"],["none"])],
            "risk": [LoadedAgent("r","r","risk","r","5m",Path("r"),["r"],["none"],["none"])],
            "strategy": [LoadedAgent("s","s","strategy","r","5m",Path("s"),["s"],["none"],["none"])],
            "portfolio": [LoadedAgent("p","p","portfolio","r","5m",Path("p"),["p"],["none"],["none"])],
            "execution": [LoadedAgent("e","e","execution","r","5m",Path("e"),["e"],["none"],["none"])],
            "monitoring": [LoadedAgent("o","o","monitoring","r","5m",Path("o"),["o"],["none"],["none"])],
        },
        status="READY",
        missing_types=[],
    )
    with pytest.raises(SystemRegistryError):
        build_execution_plan(system)


def test_equal_priority_agents_sorted_alphabetically():
    a = LoadedAgent("A", "alpha", "data", "r", "5m", Path("a"), ["a"], [], ["a_out"])
    b = LoadedAgent("B", "beta", "feature", "r", "5m", Path("b"), ["b"], [], ["b_out"])
    c = LoadedAgent("C", "gamma", "ml", "r", "5m", Path("c"), ["c"], ["a_out", "b_out"], ["c_out"])

    system = SystemRegistry(
        schema_version="1.0",
        preset="manual",
        generated_at="static",
        agents=[b, c, a],
        agents_by_type={
            "data": [a],
            "feature": [b],
            "ml": [c],
            "risk": [LoadedAgent("r","risk","risk","r","5m",Path("r"),["r"],[],["r_out"])],
            "strategy": [LoadedAgent("s","strategy","strategy","r","5m",Path("s"),["s"],[],["s_out"])],
            "portfolio": [LoadedAgent("p","portfolio","portfolio","r","5m",Path("p"),["p"],[],["p_out"])],
            "execution": [LoadedAgent("e","execution","execution","r","5m",Path("e"),["e"],[],["e_out"])],
            "monitoring": [LoadedAgent("m","monitoring","monitoring","r","5m",Path("m"),["m"],[],["m_out"])],
        },
        status="READY",
        missing_types=[],
    )

    plan = build_execution_plan(system)
    slugs = [x.slug for x in plan]
    assert slugs.index("alpha") < slugs.index("beta") < slugs.index("gamma")


def test_orchestrate_cli_dry_run_ready(tmp_path: Path):
    registry = _create_system(tmp_path)
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "orchestrate",
        "--dry-run",
        "--registry",
        str(registry),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "STATUS: READY" in res.stdout

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _create_system(tmp_path: Path) -> tuple[Path, Path]:
    system_dir = tmp_path / "system"
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(system_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return system_dir, system_dir / "registry.json"


def _set_real_adapters(system_dir: Path) -> None:
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
        cfg_path = system_dir / slug / "agent_config.json"
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        payload["runtime_mode"] = "real"
        payload["runtime_adapter"] = {"module": module, "callable": "run_agent"}
        cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_paper(registry: Path, runs_dir: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "orchestrate",
        "--paper",
        "--registry",
        str(registry),
        "--runs-dir",
        str(runs_dir),
        "--with-sample-data",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_paper_run_creates_run_directory_and_manifest(tmp_path: Path) -> None:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"

    res = _run_paper(registry, runs_dir)
    assert res.returncode == 0, res.stdout + res.stderr

    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    assert [p.name for p in run_dirs] == ["run_000001"]
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run_000001"
    assert manifest["runtime_mode"] == "real"
    assert manifest["registry_path"] == str(registry)


def test_paper_run_persists_expected_artifacts_and_execution_report(tmp_path: Path) -> None:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"

    res = _run_paper(registry, runs_dir)
    assert res.returncode == 0, res.stdout + res.stderr

    run_dir = runs_dir / "run_000001"
    expected = {
        "manifest.json",
        "market_data.json",
        "features.json",
        "signals.json",
        "risk_state.json",
        "trade_plan.json",
        "allocation.json",
        "execution_report.json",
        "alerts.json",
    }
    files = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert expected.issubset(files)

    execution_report = json.loads((run_dir / "execution_report.json").read_text(encoding="utf-8"))
    assert execution_report.get("paper") is True
    assert execution_report.get("status") == "PAPER_SIMULATED"


def test_paper_run_does_not_require_live_trading(tmp_path: Path) -> None:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"

    res = _run_paper(registry, runs_dir)
    assert res.returncode == 0, res.stdout + res.stderr

    report = json.loads((runs_dir / "run_000001" / "execution_report.json").read_text(encoding="utf-8"))
    assert report.get("paper") is True
    assert report.get("status") != "LIVE"


def test_repeated_paper_run_creates_separate_run_directories(tmp_path: Path) -> None:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"

    first = _run_paper(registry, runs_dir)
    second = _run_paper(registry, runs_dir)
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr

    run_dirs = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    assert run_dirs == ["run_000001", "run_000002"]

    m1 = json.loads((runs_dir / "run_000001" / "manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((runs_dir / "run_000002" / "manifest.json").read_text(encoding="utf-8"))
    assert m1["artifact_files"] == m2["artifact_files"]

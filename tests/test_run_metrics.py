from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_paper_run import _create_system, _run_paper, _set_real_adapters
from moex_agent.__main__ import _build_run_metrics


def _make_run(tmp_path: Path) -> Path:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"
    res = _run_paper(registry, runs_dir)
    assert res.returncode == 0, res.stdout + res.stderr
    return runs_dir / "run_000001"


def test_paper_run_writes_metrics_json(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    assert (run_dir / "metrics.json").exists()


def test_metrics_values_are_deterministic(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "1.0"
    assert metrics["run_id"] == "run_000001"
    assert isinstance(metrics["signals_count"], int)
    assert isinstance(metrics["long_probability_avg"], float)
    assert isinstance(metrics["short_probability_avg"], float)
    assert isinstance(metrics["confidence_avg"], float)
    assert isinstance(metrics["risk_kill_switch"], bool)
    assert isinstance(metrics["allocation_total"], float)
    assert isinstance(metrics["alerts_count"], int)
    assert isinstance(metrics["execution_status"], str)


def test_show_run_prints_run_and_metrics_summary(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    cmd = [sys.executable, "-m", "moex_agent", "show-run", "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Run" in res.stdout
    assert "Artifacts" in res.stdout
    assert "Metrics" in res.stdout
    assert "OK: show-run completed" in res.stdout


def test_metrics_fallback_when_source_artifact_missing(tmp_path: Path) -> None:
    _ = _make_run(tmp_path)
    metrics = _build_run_metrics("run_x", {})
    assert metrics["signals_count"] == 0
    assert metrics["long_probability_avg"] == 0.0
    assert metrics["short_probability_avg"] == 0.0
    assert metrics["confidence_avg"] == 0.0
    assert metrics["risk_kill_switch"] is False
    assert metrics["allocation_total"] == 0.0
    assert metrics["alerts_count"] == 0
    assert metrics["execution_status"] == "UNKNOWN"

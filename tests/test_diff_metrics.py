from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_paper_run import _create_system, _run_paper, _set_real_adapters


def _make_two_runs(tmp_path: Path) -> tuple[Path, Path]:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"
    first = _run_paper(registry, runs_dir)
    second = _run_paper(registry, runs_dir)
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    return runs_dir / "run_000001", runs_dir / "run_000002"


def _diff(left: Path, right: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "moex_agent", "diff-metrics", "--left", str(left), "--right", str(right)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_diff_metrics_identical_no_changes(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Metric Diff" in res.stdout
    assert "OK: no metric changes" in res.stdout


def test_diff_metrics_confidence_change_detected(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    m = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
    m["confidence_avg"] = round(float(m.get("confidence_avg", 0.0)) + 0.01, 6)
    (right / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CHANGED: confidence_avg:" in res.stdout


def test_diff_metrics_alerts_count_change_detected(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    m = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
    m["alerts_count"] = int(m.get("alerts_count", 0)) + 1
    (right / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CHANGED: alerts_count:" in res.stdout


def test_diff_metrics_allocation_total_change_detected(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    m = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
    m["allocation_total"] = 0.0
    (right / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CHANGED: allocation_total:" in res.stdout


def test_diff_metrics_missing_file_fails_clearly(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    (right / "metrics.json").unlink()

    res = _diff(left, right)
    assert res.returncode != 0
    assert "ERROR: right metrics.json missing:" in res.stdout

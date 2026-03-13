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


def _regress(baseline: Path, candidate: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "run-regression",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_metric_confidence_drop_warns(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    m = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    m["confidence_avg"] = max(0.0, float(m.get("confidence_avg", 0.0)) - 0.1)
    (candidate / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _regress(baseline, candidate)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARNING: metric confidence_avg decreased" in res.stdout


def test_metric_alerts_increase_warns(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    m = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    m["alerts_count"] = int(m.get("alerts_count", 0)) + 2
    (candidate / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _regress(baseline, candidate)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARNING: metric alerts_count increased" in res.stdout


def test_metric_allocation_to_zero_fails(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    m = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    m["allocation_total"] = 1.0
    (baseline / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    m2 = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    m2["allocation_total"] = 0.0
    (candidate / "metrics.json").write_text(json.dumps(m2, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _regress(baseline, candidate)
    assert res.returncode != 0
    assert "ERROR: metric allocation_total changed from >0 to 0" in res.stdout


def test_metric_kill_switch_flips_true_fails(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    mb = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    mb["risk_kill_switch"] = False
    (baseline / "metrics.json").write_text(json.dumps(mb, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mc = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    mc["risk_kill_switch"] = True
    (candidate / "metrics.json").write_text(json.dumps(mc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _regress(baseline, candidate)
    assert res.returncode != 0
    assert "ERROR: metric risk_kill_switch changed from false to true" in res.stdout


def test_metric_missing_candidate_metrics_fails(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    (candidate / "metrics.json").unlink()

    res = _regress(baseline, candidate)
    assert res.returncode != 0
    assert "ERROR: candidate metrics.json missing while baseline has metrics.json" in res.stdout

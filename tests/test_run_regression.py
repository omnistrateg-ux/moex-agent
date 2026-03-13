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


def test_run_regression_identical_runs_pass(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    res = _regress(baseline, candidate)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Overall" in res.stdout
    assert "OK: regression passed" in res.stdout


def test_run_regression_changed_signals_only_warns(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    path = candidate / "signals.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["confidence"] = round(float(payload.get("confidence", 0.0)) + 0.01, 6)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _regress(baseline, candidate)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARNING: changed artifact: signals.json" in res.stdout
    assert "OK: regression passed with warnings" in res.stdout


def test_run_regression_missing_execution_report_fails(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_files"] = [x for x in manifest["artifact_files"] if x != "execution_report.json"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if (candidate / "execution_report.json").exists():
        (candidate / "execution_report.json").unlink()

    res = _regress(baseline, candidate)
    assert res.returncode != 0
    assert "ERROR: execution_report.json missing in candidate" in res.stdout


def test_run_regression_changed_executed_agents_fails(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["executed_agents"] = list(reversed(manifest["executed_agents"]))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _regress(baseline, candidate)
    assert res.returncode != 0
    assert "ERROR: executed_agents changed" in res.stdout


def test_run_regression_invalid_candidate_fails(tmp_path: Path) -> None:
    baseline, candidate = _make_two_runs(tmp_path)
    (candidate / "manifest.json").unlink()

    res = _regress(baseline, candidate)
    assert res.returncode != 0
    assert "ERROR: candidate run failed replay validation" in res.stdout

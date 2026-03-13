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
    cmd = [sys.executable, "-m", "moex_agent", "diff-run", "--left", str(left), "--right", str(right)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_diff_run_identical_runs_no_changes(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Added artifacts" in res.stdout
    assert "Removed artifacts" in res.stdout
    assert "Changed artifacts" in res.stdout
    assert "Manifest changes" in res.stdout
    assert "OK: none" in res.stdout


def test_diff_run_changed_signals_detected(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    signals_path = right / "signals.json"
    signals = json.loads(signals_path.read_text(encoding="utf-8"))
    signals["confidence"] = round(float(signals.get("confidence", 0.0)) + 0.01, 6)
    signals_path.write_text(json.dumps(signals, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CHANGED: signals.json" in res.stdout


def test_diff_run_added_removed_artifact_detected(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    manifest_path = right / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_files"] = [x for x in manifest["artifact_files"] if x != "alerts.json"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if (right / "alerts.json").exists():
        (right / "alerts.json").unlink()

    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "REMOVED: alerts.json" in res.stdout


def test_diff_run_changed_executed_agents_detected(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    manifest_path = right / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["executed_agents"] = list(reversed(manifest["executed_agents"]))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = _diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CHANGED: executed_agents" in res.stdout


def test_diff_run_invalid_input_fails_clearly(tmp_path: Path) -> None:
    left, right = _make_two_runs(tmp_path)
    (right / "manifest.json").unlink()

    res = _diff(left, right)
    assert res.returncode != 0
    assert "ERROR: right run failed replay validation" in res.stdout

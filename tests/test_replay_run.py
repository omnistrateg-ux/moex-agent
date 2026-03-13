from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_paper_run import _create_system, _run_paper, _set_real_adapters


def _make_paper_run(tmp_path: Path) -> Path:
    system_dir, registry = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs"
    res = _run_paper(registry, runs_dir)
    assert res.returncode == 0, res.stdout + res.stderr
    return runs_dir / "run_000001"


def test_replay_run_valid_paper_run_passes(tmp_path: Path) -> None:
    run_dir = _make_paper_run(tmp_path)
    cmd = [sys.executable, "-m", "moex_agent", "replay-run", "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Overall" in res.stdout
    assert "OK: replay validation passed" in res.stdout


def test_replay_run_missing_listed_artifact_fails(tmp_path: Path) -> None:
    run_dir = _make_paper_run(tmp_path)
    (run_dir / "signals.json").unlink()

    cmd = [sys.executable, "-m", "moex_agent", "replay-run", "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: listed artifact is missing: signals.json" in res.stdout


def test_replay_run_broken_json_artifact_fails(tmp_path: Path) -> None:
    run_dir = _make_paper_run(tmp_path)
    (run_dir / "signals.json").write_text("{broken", encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "replay-run", "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: invalid JSON in signals.json" in res.stdout


def test_replay_run_missing_optional_unlisted_artifact_warns_only(tmp_path: Path) -> None:
    run_dir = _make_paper_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_files"] = [x for x in manifest["artifact_files"] if x != "alerts.json"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if (run_dir / "alerts.json").exists():
        (run_dir / "alerts.json").unlink()

    cmd = [sys.executable, "-m", "moex_agent", "replay-run", "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARNING: optional artifact absent and unlisted: alerts.json" in res.stdout

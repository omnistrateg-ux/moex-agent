from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _export_snapshot(tmp_path: Path, name: str) -> Path:
    system_dir = tmp_path / f"system_{name}"
    create_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(system_dir),
    ]
    created = subprocess.run(create_cmd, capture_output=True, text=True)
    assert created.returncode == 0, created.stdout + created.stderr

    snapshot = tmp_path / f"{name}.json"
    export_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "export-system",
        "--registry",
        str(system_dir / "registry.json"),
        "--out",
        str(snapshot),
    ]
    exported = subprocess.run(export_cmd, capture_output=True, text=True)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    return snapshot


def _run_diff(left: Path, right: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "diff-snapshot",
        "--left",
        str(left),
        "--right",
        str(right),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_diff_snapshot_identical_has_no_changes(tmp_path: Path) -> None:
    a = _export_snapshot(tmp_path, "a")
    b = tmp_path / "b.json"
    b.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")

    res = _run_diff(a, b)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: snapshots are identical" in res.stdout


def test_diff_snapshot_added_agent_detected(tmp_path: Path) -> None:
    a = _export_snapshot(tmp_path, "a")
    b = tmp_path / "b.json"
    payload = json.loads(a.read_text(encoding="utf-8"))
    extra = json.loads(json.dumps(payload["agents"][0]))
    extra["slug"] = "z-added-agent"
    extra["name"] = "Z Added Agent"
    extra["config"]["slug"] = "z-added-agent"
    extra["config"]["name"] = "Z Added Agent"
    payload["agents"].append(extra)
    b.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_diff(a, b)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ADDED: z-added-agent" in res.stdout


def test_diff_snapshot_removed_agent_detected(tmp_path: Path) -> None:
    a = _export_snapshot(tmp_path, "a")
    b = tmp_path / "b.json"
    payload = json.loads(a.read_text(encoding="utf-8"))
    removed_slug = payload["agents"][0]["slug"]
    payload["agents"] = payload["agents"][1:]
    b.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_diff(a, b)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"REMOVED: {removed_slug}" in res.stdout


def test_diff_snapshot_changed_timeframe_and_runtime_mode_detected(tmp_path: Path) -> None:
    a = _export_snapshot(tmp_path, "a")
    b = tmp_path / "b.json"
    payload = json.loads(a.read_text(encoding="utf-8"))
    payload["agents"][0]["timeframe"] = "1h"
    payload["agents"][0]["config"]["runtime_mode"] = "real"
    b.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_diff(a, b)
    assert res.returncode == 0, res.stdout + res.stderr
    slug = payload["agents"][0]["slug"]
    assert f"CHANGED: {slug}.timeframe" in res.stdout
    assert f"CHANGED: {slug}.runtime_mode" in res.stdout


def test_diff_snapshot_output_is_deterministic(tmp_path: Path) -> None:
    a = _export_snapshot(tmp_path, "a")
    b = tmp_path / "b.json"
    payload = json.loads(a.read_text(encoding="utf-8"))
    payload["agents"] = list(reversed(payload["agents"]))
    b.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    r1 = _run_diff(a, b)
    r2 = _run_diff(a, b)
    assert r1.returncode == 0 and r2.returncode == 0
    assert r1.stdout == r2.stdout

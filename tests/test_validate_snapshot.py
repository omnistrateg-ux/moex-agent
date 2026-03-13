from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _create_snapshot(tmp_path: Path) -> Path:
    system_dir = tmp_path / "system"
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

    snapshot = tmp_path / "snapshot.json"
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


def test_validate_snapshot_valid_export_passes(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)

    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "validate-snapshot",
        "--snapshot",
        str(snapshot),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: snapshot valid" in res.stdout


def test_validate_snapshot_duplicate_slug_fails(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["agents"][1]["slug"] = payload["agents"][0]["slug"]
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "validate-snapshot", "--snapshot", str(snapshot)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: duplicate slug" in res.stdout


def test_validate_snapshot_missing_top_level_field_fails(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload.pop("preset", None)
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "validate-snapshot", "--snapshot", str(snapshot)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: missing top-level fields: preset" in res.stdout


def test_validate_snapshot_config_mismatch_fails(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["agents"][0]["config"]["name"] = "Mismatch"
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "validate-snapshot", "--snapshot", str(snapshot)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: agents[0] config.name mismatch" in res.stdout

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _create_system(tmp_path: Path, name: str = "system") -> Path:
    out = tmp_path / name
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return out


def _export_snapshot(system_dir: Path, snapshot_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "export-system",
        "--registry",
        str(system_dir / "registry.json"),
        "--out",
        str(snapshot_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_check_compatibility_valid_generated_system(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)

    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "check-compatibility",
        "--registry",
        str(system_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Overall" in res.stdout
    assert "OK: compatible" in res.stdout


def test_check_compatibility_missing_adapter_in_real_mode_fails(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)
    cfg_path = system_dir / "data-collector" / "agent_config.json"
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["runtime_mode"] = "real"
    payload["runtime_adapter"] = None
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "check-compatibility", "--registry", str(system_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "runtime_adapter required for runtime_mode=real" in res.stdout


def test_check_compatibility_unsupported_snapshot_version_fails(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)
    snap = tmp_path / "snapshot.json"
    _export_snapshot(system_dir, snap)

    payload = json.loads(snap.read_text(encoding="utf-8"))
    payload["snapshot_version"] = "2.0"
    snap.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "check-compatibility", "--snapshot", str(snap)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: failed to load snapshot" in res.stdout


def test_check_compatibility_inconsistent_dependencies_fails(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)
    cfg_path = system_dir / "strategy-planner" / "agent_config.json"
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dependencies"]["provides"] = ["wrong_token"]
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cmd = [sys.executable, "-m", "moex_agent", "check-compatibility", "--registry", str(system_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "semantic contract mismatch" in res.stdout

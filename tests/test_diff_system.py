from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _create_system(tmp_path: Path, name: str) -> Path:
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


def _run_diff(left: Path, right: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "diff-system",
        "--left",
        str(left),
        "--right",
        str(right),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_diff_system_identical_generated_systems_no_changes(tmp_path: Path) -> None:
    left = _create_system(tmp_path, "left")
    right = _create_system(tmp_path, "right")

    res = _run_diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: snapshots are identical" in res.stdout


def test_diff_system_changed_runtime_mode_detected(tmp_path: Path) -> None:
    left = _create_system(tmp_path, "left")
    right = _create_system(tmp_path, "right")

    cfg = right / "data-collector" / "agent_config.json"
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["runtime_mode"] = "real"
    cfg.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    res = _run_diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CHANGED: data-collector.runtime_mode" in res.stdout


def test_diff_system_removed_bundle_agent_detected(tmp_path: Path) -> None:
    left = _create_system(tmp_path, "left")
    right = _create_system(tmp_path, "right")

    registry = right / "registry.json"
    payload = json.loads(registry.read_text(encoding="utf-8"))
    removed_slug = payload["agents"][0]["slug"]
    payload["agents"] = [a for a in payload["agents"] if a["slug"] != removed_slug]
    registry.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    res = _run_diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"REMOVED: {removed_slug}" in res.stdout


def test_diff_system_directory_input_works(tmp_path: Path) -> None:
    left = _create_system(tmp_path, "left")
    right = _create_system(tmp_path, "right")

    res = _run_diff(left, right)
    assert res.returncode == 0, res.stdout + res.stderr


def test_diff_system_registry_path_input_works(tmp_path: Path) -> None:
    left = _create_system(tmp_path, "left")
    right = _create_system(tmp_path, "right")

    res = _run_diff(left / "registry.json", right / "registry.json")
    assert res.returncode == 0, res.stdout + res.stderr

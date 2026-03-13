from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_snapshot(tmp_path: Path) -> Path:
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


def test_migrate_snapshot_valid_1_0_succeeds(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    out = tmp_path / "migrated.json"

    cmd = [sys.executable, "-m", "moex_agent", "migrate-snapshot", "--snapshot", str(snapshot), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["snapshot_version"] == "1.0"
    assert payload["schema_version"] == "1.0"


def test_migrate_snapshot_output_is_deterministic(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    out1 = tmp_path / "m1.json"
    out2 = tmp_path / "m2.json"

    for out in [out1, out2]:
        cmd = [sys.executable, "-m", "moex_agent", "migrate-snapshot", "--snapshot", str(snapshot), "--out", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr

    p1 = json.loads(out1.read_text(encoding="utf-8"))
    p2 = json.loads(out2.read_text(encoding="utf-8"))
    assert p1 == p2


def test_migrate_snapshot_sorts_agents_by_slug(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["agents"] = list(reversed(payload["agents"]))
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out = tmp_path / "sorted.json"
    cmd = [sys.executable, "-m", "moex_agent", "migrate-snapshot", "--snapshot", str(snapshot), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    migrated = json.loads(out.read_text(encoding="utf-8"))
    slugs = [a["slug"] for a in migrated["agents"]]
    assert slugs == sorted(slugs)


def test_migrate_snapshot_normalizes_optional_config_defaults(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    cfg = payload["agents"][0]["config"]
    cfg.pop("runtime_mode", None)
    cfg.pop("dependencies", None)
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out = tmp_path / "normalized.json"
    cmd = [sys.executable, "-m", "moex_agent", "migrate-snapshot", "--snapshot", str(snapshot), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    migrated = json.loads(out.read_text(encoding="utf-8"))
    normalized_cfg = migrated["agents"][0]["config"]
    assert normalized_cfg["runtime_mode"] == "stub"
    assert normalized_cfg["dependencies"] == {"requires": [], "provides": []}


def test_migrate_snapshot_unsupported_version_fails(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["snapshot_version"] = "2.0"
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out = tmp_path / "bad.json"
    cmd = [sys.executable, "-m", "moex_agent", "migrate-snapshot", "--snapshot", str(snapshot), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR: unsupported snapshot_version: 2.0" in res.stdout

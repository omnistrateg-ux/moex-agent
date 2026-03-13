from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _create_system(tmp_path: Path) -> Path:
    out = tmp_path / "system"
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


def test_export_system_writes_valid_deterministic_snapshot(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)
    snapshot = tmp_path / "snapshot.json"

    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "export-system",
        "--registry",
        str(system_dir / "registry.json"),
        "--out",
        str(snapshot),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert snapshot.exists()

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["snapshot_version"] == "1.0"
    assert payload["preset"] == "core_moex_v1"
    assert payload["generated_at"] == "static"
    assert len(payload["agents"]) == 8
    slugs = [a["slug"] for a in payload["agents"]]
    assert slugs == sorted(slugs)


def test_import_system_recreates_valid_system_and_dry_run(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)
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
    export = subprocess.run(export_cmd, capture_output=True, text=True)
    assert export.returncode == 0, export.stdout + export.stderr

    imported_dir = tmp_path / "imported"
    import_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "import-system",
        "--snapshot",
        str(snapshot),
        "--output-dir",
        str(imported_dir),
    ]
    imported = subprocess.run(import_cmd, capture_output=True, text=True)
    assert imported.returncode == 0, imported.stdout + imported.stderr

    registry = imported_dir / "registry.json"
    assert registry.exists()

    sample_slug = "data-collector"
    assert (imported_dir / sample_slug / "AGENT_PROMPT.md").exists()
    assert (imported_dir / sample_slug / "README.md").exists()
    assert (imported_dir / sample_slug / "src" / f"{sample_slug}.py").exists()
    assert (imported_dir / sample_slug / "tests" / f"test_{sample_slug}.py").exists()

    show_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "show-system",
        "--registry",
        str(registry),
    ]
    shown = subprocess.run(show_cmd, capture_output=True, text=True)
    assert shown.returncode == 0, shown.stdout + shown.stderr

    dry_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "orchestrate",
        "--dry-run",
        "--registry",
        str(registry),
    ]
    dry = subprocess.run(dry_cmd, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "STATUS: READY" in dry.stdout


def test_exporting_same_system_twice_is_equivalent(tmp_path: Path) -> None:
    system_dir = _create_system(tmp_path)
    s1 = tmp_path / "snap1.json"
    s2 = tmp_path / "snap2.json"

    for out in [s1, s2]:
        cmd = [
            sys.executable,
            "-m",
            "moex_agent",
            "export-system",
            "--registry",
            str(system_dir / "registry.json"),
            "--out",
            str(out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr

    p1 = json.loads(s1.read_text(encoding="utf-8"))
    p2 = json.loads(s2.read_text(encoding="utf-8"))
    assert p1 == p2

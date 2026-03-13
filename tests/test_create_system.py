import json
import subprocess
import sys
from pathlib import Path


EXPECTED_SLUGS = {
    "data-collector",
    "feature-builder",
    "signal-model",
    "risk-guard",
    "strategy-planner",
    "portfolio-allocator",
    "execution-router",
    "system-monitor",
}


def test_create_system_generates_core_moex_v1(tmp_path: Path):
    create_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(tmp_path),
    ]
    create = subprocess.run(create_cmd, capture_output=True, text=True)
    assert create.returncode == 0, create.stdout + create.stderr

    generated_dirs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert EXPECTED_SLUGS.issubset(generated_dirs)

    registry_path = tmp_path / "registry.json"
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["schema_version"] == "1.0"
    assert registry["preset"] == "core_moex_v1"
    assert registry["generated_at"] == "static"
    assert len(registry["agents"]) == 8

    for item in registry["agents"]:
        cfg = tmp_path / item["config_path"]
        assert cfg.exists(), f"missing config path: {cfg}"

    check_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "agents-check",
        "--agents-dir",
        str(tmp_path),
    ]
    checked = subprocess.run(check_cmd, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr

    dry_run_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "agents-dry-run",
        "--agents-dir",
        str(tmp_path),
    ]
    dry = subprocess.run(dry_run_cmd, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "RESULT: READY" in dry.stdout

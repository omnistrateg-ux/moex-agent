import json
import subprocess
import sys
from pathlib import Path

import pytest

from moex_agent.orchestrator_loader import SystemRegistryError, load_system_registry


def _create_system(tmp_path: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(tmp_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return tmp_path / "registry.json"


def test_load_valid_generated_system(tmp_path: Path):
    registry_path = _create_system(tmp_path)
    system = load_system_registry(registry_path)

    assert system.preset == "core_moex_v1"
    assert len(system.agents) == 8
    assert system.status == "READY"
    assert system.missing_types == []


def test_missing_type_leads_to_incomplete(tmp_path: Path):
    registry_path = _create_system(tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    payload["agents"] = [a for a in payload["agents"] if a["agent_type"] != "monitoring"]
    registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    system = load_system_registry(registry_path)
    assert system.status == "INCOMPLETE"
    assert "monitoring" in system.missing_types


def test_missing_bundle_files_raises(tmp_path: Path):
    registry_path = _create_system(tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    first = payload["agents"][0]
    config_path = tmp_path / first["config_path"]
    readme_path = config_path.parent / "README.md"
    readme_path.unlink()

    with pytest.raises(SystemRegistryError):
        load_system_registry(registry_path)

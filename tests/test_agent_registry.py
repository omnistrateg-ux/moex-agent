import json
from pathlib import Path

import pytest

from moex_agent.agent_factory import AgentBlueprint, create_agent_files
from moex_agent.agent_registry import RegistryValidationError, build_dry_run_plan, load_registry


REQUIRED_TYPES = ["data", "feature", "ml", "risk", "strategy", "portfolio", "execution", "monitoring"]


def test_load_registry_ok(tmp_path: Path):
    for agent_type in REQUIRED_TYPES:
        blueprint = AgentBlueprint(
            name=f"{agent_type.title()} Agent",
            role=f"{agent_type} role",
            agent_type=agent_type,
            timeframe="5m",
        )
        create_agent_files(blueprint, output_dir=tmp_path)

    specs = load_registry(tmp_path)
    assert len(specs) == len(REQUIRED_TYPES)

    plan = build_dry_run_plan(specs)
    assert "RESULT: READY" in plan


def test_load_registry_rejects_invalid_agent_type(tmp_path: Path):
    payload = {
        "name": "Bad Agent",
        "slug": "bad-agent",
        "agent_type": "invalid",
        "role": "broken",
        "timeframe": "5m",
        "risk_profile": "balanced",
        "input_data": ["x"],
        "output_data": ["x"],
        "algorithms": ["x"],
        "risk_control": ["x"],
        "files_to_create": ["x.py"],
    }
    (tmp_path / "bad-agent.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RegistryValidationError):
        load_registry(tmp_path)

import json
from pathlib import Path

from moex_agent.agent_factory import AgentBlueprint, create_agent_files


def test_create_agent_files(tmp_path: Path):
    blueprint = AgentBlueprint(
        name="Momentum Scout",
        role="Ищет импульсные точки входа",
        agent_type="strategy",
        timeframe="5m",
        risk_profile="aggressive",
        goals=["Находить пробои", "Фильтровать ложные импульсы"],
    )

    prompt_path, config_path = create_agent_files(blueprint, output_dir=tmp_path)

    assert prompt_path.exists()
    assert config_path.exists()
    assert prompt_path.name == "momentum-scout_PROMPT.md"

    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "## INPUT DATA" in prompt_text
    assert "## OUTPUT DATA" in prompt_text
    assert '"Недостаточно данных"' in prompt_text

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["slug"] == "momentum-scout"
    assert config["agent_type"] == "strategy"
    assert "input_data" in config
    assert "files" in config and config["files"]
    assert config["schema_version"] == "1.0"
    assert config["runtime_mode"] == "stub"
    assert config["runtime_adapter"] is None
    assert "dependencies" in config
    assert "requires" in config["dependencies"] and "provides" in config["dependencies"]
    assert config["constraints"]["if_missing_data"] == "Недостаточно данных"

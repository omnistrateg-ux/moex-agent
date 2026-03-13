import json
import subprocess
import sys
from pathlib import Path


def test_create_agent_bundle_cli(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-agent-bundle",
        "--name",
        "Momentum ML Agent",
        "--role",
        "Прогнозирует long/short вероятности",
        "--agent-type",
        "ml",
        "--timeframe",
        "5m",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    slug = "momentum-ml-agent"
    bundle_dir = tmp_path / slug
    config_path = bundle_dir / "agent_config.json"
    prompt_path = bundle_dir / "AGENT_PROMPT.md"
    readme_path = bundle_dir / "README.md"
    src_path = bundle_dir / "src" / f"{slug}.py"
    test_path = bundle_dir / "tests" / f"test_{slug}.py"

    assert bundle_dir.exists()
    assert config_path.exists()
    assert prompt_path.exists()
    assert readme_path.exists()
    assert src_path.exists()
    assert test_path.exists()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["agent_type"] == "ml"
    assert config["schema_version"] == "1.0"
    assert config["runtime_mode"] == "stub"
    assert config["runtime_adapter"] is None
    assert "dependencies" in config
    assert "requires" in config["dependencies"] and "provides" in config["dependencies"]

    expected_files = [
        "agent_config.json",
        "AGENT_PROMPT.md",
        "README.md",
        f"src/{slug}.py",
        f"tests/test_{slug}.py",
    ]
    assert config["files"] == expected_files

    validate_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "validate-agent-spec",
        "--config",
        str(config_path),
    ]
    validate = subprocess.run(validate_cmd, capture_output=True, text=True)
    assert validate.returncode == 0, validate.stdout + validate.stderr
    assert "Status: VALID" in validate.stdout

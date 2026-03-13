import json
from pathlib import Path

from moex_agent.agent_registry import validate_agent_spec


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _valid_payload() -> dict:
    return {
        "name": "Momentum Signal Agent",
        "slug": "momentum-signal-agent",
        "agent_type": "strategy",
        "input_data": ["candles", "volume"],
        "output_data": ["probability_long", "probability_short", "confidence"],
        "algorithms": ["ema momentum", "volume spike filter"],
        "risk_control": ["max_position", "kill_switch"],
        "files": ["moex_agent/agents/momentum_signal_agent.py"],
        "schema_version": "1.0",
    }


def test_validate_agent_spec_valid(tmp_path: Path):
    path = _write(tmp_path / "valid.json", _valid_payload())
    assert validate_agent_spec(path) == []


def test_validate_agent_spec_invalid_agent_type(tmp_path: Path):
    payload = _valid_payload()
    payload["agent_type"] = "signals"
    path = _write(tmp_path / "bad_type.json", payload)

    errors = validate_agent_spec(path)
    assert "invalid agent_type: signals" in errors


def test_validate_agent_spec_missing_required_field(tmp_path: Path):
    payload = _valid_payload()
    payload.pop("algorithms")
    path = _write(tmp_path / "missing_algorithms.json", payload)

    errors = validate_agent_spec(path)
    assert "missing field: algorithms" in errors


def test_validate_agent_spec_empty_algorithms_list(tmp_path: Path):
    payload = _valid_payload()
    payload["algorithms"] = []
    path = _write(tmp_path / "empty_algorithms.json", payload)

    errors = validate_agent_spec(path)
    assert "algorithms must be a non-empty list" in errors

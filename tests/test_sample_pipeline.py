import json
import subprocess
import sys
from pathlib import Path

from moex_agent.sample_data import build_sample_context, build_sample_market_data


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


def _set_adapter(cfg_path: Path, module: str):
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["runtime_adapter"] = {"module": module, "callable": "run_agent"}
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_sample_market_data_deterministic_structure():
    a = build_sample_market_data()
    b = build_sample_market_data()
    assert a == b
    assert {"ticker", "timeframe", "candles", "volume", "spread_bps", "session"}.issubset(a.keys())
    assert isinstance(build_sample_context(), dict)


def test_orchestrate_simulate_with_sample_data_succeeds(tmp_path: Path):
    registry = _create_system(tmp_path)

    adapter_map = {
        "data-collector": "moex_agent.adapters.data_adapter",
        "feature-builder": "moex_agent.adapters.feature_adapter",
        "signal-model": "moex_agent.adapters.ml_adapter",
        "risk-guard": "moex_agent.adapters.risk_adapter",
        "strategy-planner": "moex_agent.adapters.strategy_adapter",
        "portfolio-allocator": "moex_agent.adapters.portfolio_adapter",
        "execution-router": "moex_agent.adapters.execution_adapter",
        "system-monitor": "moex_agent.adapters.monitoring_adapter",
    }
    for slug, module in adapter_map.items():
        _set_adapter(tmp_path / slug / "agent_config.json", module)

    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "orchestrate",
        "--simulate",
        "--runtime-mode",
        "real",
        "--with-sample-data",
        "--registry",
        str(registry),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "STATUS: READY" in res.stdout


def test_artifact_flow_reaches_execution_report(tmp_path: Path):
    registry = _create_system(tmp_path)
    for slug, module in {
        "data-collector": "moex_agent.adapters.data_adapter",
        "feature-builder": "moex_agent.adapters.feature_adapter",
        "signal-model": "moex_agent.adapters.ml_adapter",
        "risk-guard": "moex_agent.adapters.risk_adapter",
        "strategy-planner": "moex_agent.adapters.strategy_adapter",
        "portfolio-allocator": "moex_agent.adapters.portfolio_adapter",
        "execution-router": "moex_agent.adapters.execution_adapter",
        "system-monitor": "moex_agent.adapters.monitoring_adapter",
    }.items():
        _set_adapter(tmp_path / slug / "agent_config.json", module)

    cmd = [sys.executable, "-m", "moex_agent", "orchestrate", "--simulate", "--runtime-mode", "real", "--with-sample-data", "--registry", str(registry)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert "execution-router [execution] -> execution_report" in res.stdout


def test_missing_upstream_artifact_causes_clear_runtime_error(tmp_path: Path):
    registry = _create_system(tmp_path)
    # break upstream by using feature adapter for data agent (won't provide market_data)
    _set_adapter(tmp_path / "data-collector" / "agent_config.json", "moex_agent.adapters.feature_adapter")
    _set_adapter(tmp_path / "feature-builder" / "agent_config.json", "moex_agent.adapters.feature_adapter")
    _set_adapter(tmp_path / "signal-model" / "agent_config.json", "moex_agent.adapters.ml_adapter")
    _set_adapter(tmp_path / "risk-guard" / "agent_config.json", "moex_agent.adapters.risk_adapter")
    _set_adapter(tmp_path / "strategy-planner" / "agent_config.json", "moex_agent.adapters.strategy_adapter")
    _set_adapter(tmp_path / "portfolio-allocator" / "agent_config.json", "moex_agent.adapters.portfolio_adapter")
    _set_adapter(tmp_path / "execution-router" / "agent_config.json", "moex_agent.adapters.execution_adapter")
    _set_adapter(tmp_path / "system-monitor" / "agent_config.json", "moex_agent.adapters.monitoring_adapter")

    cmd = [sys.executable, "-m", "moex_agent", "orchestrate", "--simulate", "--runtime-mode", "real", "--registry", str(registry)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "market_data is required" in res.stdout

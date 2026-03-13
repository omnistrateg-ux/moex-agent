from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from moex_agent.adapters.common import AdapterCompatibilityError
from moex_agent.adapters.data_adapter import run_agent as run_data_adapter
from moex_agent.adapters.feature_adapter import run_agent as run_feature_adapter
from moex_agent.adapters.risk_adapter import run_agent as run_risk_adapter
from moex_agent.adapters.monitoring_adapter import run_agent as run_monitoring_adapter
from moex_agent.orchestrator_loader import load_system_registry
from moex_agent.runtime_contracts import run_execution_plan
from moex_agent.semantic_contracts import validate_semantic_output


def _sample_market_data() -> dict:
    return {
        "ticker": "SBER",
        "timeframe": "5m",
        "candles": [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10000.0},
            {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 11000.0},
        ],
        "volume": 11000.0,
        "spread_bps": 10.0,
        "quote": {"bid": 101.4, "ask": 101.6},
    }


def test_real_adapter_executes_when_project_module_available() -> None:
    data_result = run_data_adapter({"slug": "data-collector"}, {"artifacts": {}})
    assert data_result["status"] == "ok"
    validate_semantic_output("data", data_result["provides"])

    feature_result = run_feature_adapter(
        {"slug": "feature-builder"},
        {"artifacts": {"market_data": _sample_market_data()}},
    )
    assert feature_result["status"] == "ok"
    validate_semantic_output("feature", feature_result["provides"])


def test_missing_module_fails_clearly() -> None:
    with patch("moex_agent.adapters.common.importlib.import_module", side_effect=ImportError("boom")):
        with pytest.raises(AdapterCompatibilityError, match="missing adapter dependency module"):
            run_feature_adapter({"slug": "feature-builder"}, {"artifacts": {"market_data": _sample_market_data()}})


def test_semantic_output_remains_valid_for_risk_and_monitoring() -> None:
    risk_result = run_risk_adapter(
        {"slug": "risk-guard", "p_threshold": 0.2},
        {
            "artifacts": {
                "market_data": _sample_market_data(),
                "signals": {"probability_long": 0.7, "probability_short": 0.2, "confidence": 0.6},
            }
        },
    )
    assert risk_result["status"] == "ok"
    validate_semantic_output("risk", risk_result["provides"])

    monitoring_result = run_monitoring_adapter(
        {"slug": "system-monitor"},
        {
            "artifacts": {
                "market_data": _sample_market_data(),
                "signals": {"probability_long": 0.7, "probability_short": 0.2, "confidence": 0.6},
                "risk_state": {"kill_switch": False, "max_drawdown": 0.0, "position_limit": 1.0},
                "execution_report": {"orders": 1, "status": "ok"},
            }
        },
    )
    assert monitoring_result["status"] == "ok"
    validate_semantic_output("monitoring", monitoring_result["provides"])


def test_stub_mode_remains_unchanged(tmp_path: Path) -> None:
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
    created = subprocess.run(cmd, capture_output=True, text=True)
    assert created.returncode == 0, created.stdout + created.stderr

    system = load_system_registry(tmp_path / "registry.json")
    results = run_execution_plan(system)
    assert len(results) == 8
    assert all(r.get("status") == "ok" for r in results)

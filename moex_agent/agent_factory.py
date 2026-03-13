"""Utilities for generating deterministic MOEX sub-agent templates."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


_SLUG_RE = re.compile(r"[^a-z0-9]+")

_AGENT_TYPE_PROFILES = {
    "data": {
        "input_data": ["MOEX ISS candles", "MOEX ISS orderbook", "MOEX ISS trades", "MOEX market stats"],
        "output_data": ["normalized_market_data", "data_quality_flags", "last_update_ts"],
        "algorithms": ["ISS polling", "schema validation", "session-aware resampling"],
        "risk_control": ["drop stale snapshots", "halt on schema mismatch", "session boundary checks"],
        "files": ["moex_agent/agents/data_agent.py"],
    },
    "feature": {
        "input_data": ["normalized candles", "trades", "orderbook snapshots"],
        "output_data": ["returns", "volatility", "momentum", "liquidity_features", "vwap_features"],
        "algorithms": ["rolling windows", "microstructure transforms", "outlier clipping"],
        "risk_control": ["feature null-rate threshold", "no future leakage checks"],
        "files": ["moex_agent/agents/feature_agent.py"],
    },
    "ml": {
        "input_data": ["engineered_features", "market regime tags"],
        "output_data": ["probability_long", "probability_short", "confidence"],
        "algorithms": ["calibrated classifier", "horizon ensemble", "probability calibration"],
        "risk_control": ["minimum confidence gate", "regime mismatch rejection"],
        "files": ["moex_agent/agents/ml_agent.py"],
    },
    "risk": {
        "input_data": ["candidate signal", "portfolio state", "broker limits"],
        "output_data": ["risk_approved", "position_size", "stop_loss", "kill_switch_state"],
        "algorithms": ["max drawdown checks", "leverage caps", "position sizing"],
        "risk_control": ["max_position", "max_drawdown", "max_leverage", "kill_switch"],
        "files": ["moex_agent/agents/risk_agent.py"],
    },
    "strategy": {
        "input_data": ["model probabilities", "regime classification", "liquidity filters"],
        "output_data": ["trade_plan", "entry_rules", "exit_rules", "invalidations"],
        "algorithms": ["signal fusion", "regime-aware rules", "cost-aware filtering"],
        "risk_control": ["reject weak edge", "require stop/take definition"],
        "files": ["moex_agent/agents/strategy_agent.py"],
    },
    "portfolio": {
        "input_data": ["approved trade plans", "current exposures", "PnL"],
        "output_data": ["capital_allocation", "target_weights", "rebalance_actions"],
        "algorithms": ["risk parity constraints", "sector exposure limits", "turnover control"],
        "risk_control": ["exposure cap per ticker", "daily loss budget", "liquidity floor"],
        "files": ["moex_agent/agents/portfolio_agent.py"],
    },
    "execution": {
        "input_data": ["approved orders", "orderbook", "broker API status"],
        "output_data": ["execution_report", "fill_price", "slippage_bps"],
        "algorithms": ["TWAP", "VWAP", "Iceberg", "limit execution"],
        "risk_control": ["max slippage", "retry budget", "cancel-on-disconnect"],
        "files": ["moex_agent/agents/execution_agent.py"],
    },
    "monitoring": {
        "input_data": ["system metrics", "strategy metrics", "risk events"],
        "output_data": ["alerts", "health_state", "incident_log"],
        "algorithms": ["heartbeat checks", "drift detection", "latency monitoring"],
        "risk_control": ["halt on missing data", "alert escalation matrix"],
        "files": ["moex_agent/agents/monitoring_agent.py"],
    },
}

AGENT_TYPES = tuple(sorted(_AGENT_TYPE_PROFILES.keys()))

_DEFAULT_SEMANTIC_PROVIDES = {
    "data": {"market_data": {"source": "moex_iss", "status": "ok"}},
    "feature": {"features": {"version": "1.0", "count": 0}},
    "ml": {"signals": {"probability_long": 0.5, "probability_short": 0.5, "confidence": 0.0}},
    "risk": {"risk_state": {"kill_switch": False, "max_drawdown": 0.0, "position_limit": 0.0}},
    "strategy": {"trade_plan": {"action": "HOLD", "reason": "stub"}},
    "portfolio": {"allocation": {"cash": 1.0}},
    "execution": {"execution_report": {"status": "SIMULATED", "fills": []}},
    "monitoring": {"alerts": {"level": "INFO", "messages": []}},
}

_DEFAULT_DEPENDENCIES = {
    "data": {"requires": [], "provides": ["market_data"]},
    "feature": {"requires": ["market_data"], "provides": ["features"]},
    "ml": {"requires": ["features"], "provides": ["signals"]},
    "risk": {"requires": ["market_data", "signals"], "provides": ["risk_state"]},
    "strategy": {"requires": ["signals", "risk_state"], "provides": ["trade_plan"]},
    "portfolio": {"requires": ["trade_plan", "risk_state"], "provides": ["allocation"]},
    "execution": {"requires": ["allocation"], "provides": ["execution_report"]},
    "monitoring": {"requires": ["market_data", "signals", "risk_state", "execution_report"], "provides": ["alerts"]},
}



@dataclass
class AgentBlueprint:
    """Configuration used to scaffold a deterministic MOEX agent."""

    name: str
    role: str
    agent_type: str
    timeframe: str
    risk_profile: str = "balanced"
    goals: List[str] | None = None
    input_data: List[str] | None = None
    output_data: List[str] | None = None
    algorithms: List[str] | None = None
    risk_limits: List[str] | None = None
    files: List[str] | None = None

    @property
    def slug(self) -> str:
        """Filesystem-safe slug used for filenames and folder names."""
        lowered = self.name.strip().lower()
        slug = _SLUG_RE.sub("-", lowered).strip("-")
        if not slug:
            raise ValueError("Agent name must contain letters or digits")
        return slug


def _normalize_list(items: Iterable[str] | None, fallback: List[str]) -> List[str]:
    normalized = [item.strip() for item in (items or []) if item and item.strip()]
    return normalized if normalized else fallback


def _profile_for(agent_type: str) -> dict:
    normalized_type = agent_type.strip().lower()
    if normalized_type not in _AGENT_TYPE_PROFILES:
        allowed = ", ".join(AGENT_TYPES)
        raise ValueError(f"Unknown agent type '{agent_type}'. Allowed: {allowed}")
    return _AGENT_TYPE_PROFILES[normalized_type]


def _pascal_case_from_slug(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("-") if part)


def _build_prompt_and_config(blueprint: AgentBlueprint, files: List[str]) -> Tuple[str, dict]:
    profile = _profile_for(blueprint.agent_type)
    goals = _normalize_list(
        blueprint.goals,
        [
            "Работать только с проверяемыми рыночными данными MOEX ISS",
            "Формировать детерминированный и воспроизводимый результат",
            "Явно объяснять риск и причины решения",
        ],
    )
    input_data = _normalize_list(blueprint.input_data, profile["input_data"])
    output_data = _normalize_list(blueprint.output_data, profile["output_data"])
    algorithms = _normalize_list(blueprint.algorithms, profile["algorithms"])
    risk_limits = _normalize_list(blueprint.risk_limits, profile["risk_control"])

    prompt_text = (
        f"# AGENT NAME\n{blueprint.name}\n\n"
        f"## ROLE\n{blueprint.role}\n\n"
        f"## AGENT TYPE\n{blueprint.agent_type.lower()}\n\n"
        f"## INPUT DATA\n" + "\n".join(f"- {row}" for row in input_data) + "\n\n"
        f"## OUTPUT DATA\n" + "\n".join(f"- {row}" for row in output_data) + "\n\n"
        f"## ALGORITHMS\n" + "\n".join(f"- {row}" for row in algorithms) + "\n\n"
        f"## RISK CONTROL\n" + "\n".join(f"- {row}" for row in risk_limits) + "\n\n"
        f"## FILES TO CREATE\n" + "\n".join(f"- {row}" for row in files) + "\n\n"
        f"## GOALS\n" + "\n".join(f"- {goal}" for goal in goals) + "\n\n"
        "## OPERATING CONSTRAINTS\n"
        "- Source of truth: MOEX ISS API only (https://iss.moex.com).\n"
        "- If data is missing, return exactly: \"Недостаточно данных\".\n"
        "- Respect MOEX specifics: trading sessions, liquidity, T+2, short-sale constraints, leverage.\n"
        "- Output must be deterministic: same input -> same output.\n\n"
        "## EXECUTION PARAMETERS\n"
        f"- Timeframe: {blueprint.timeframe}\n"
        f"- Risk profile: {blueprint.risk_profile}\n"
    )

    config = {
        "schema_version": "1.0",
        "name": blueprint.name,
        "slug": blueprint.slug,
        "role": blueprint.role,
        "agent_type": blueprint.agent_type.lower(),
        "timeframe": blueprint.timeframe,
        "input_data": input_data,
        "output_data": output_data,
        "algorithms": algorithms,
        "risk_control": risk_limits,
        "files": files,
        "dependencies": _DEFAULT_DEPENDENCIES[blueprint.agent_type.lower()],
        "runtime_mode": "stub",
        "runtime_adapter": None,
        "risk_profile": blueprint.risk_profile,
        "goals": goals,
    }
    return prompt_text, config


def create_agent_files(blueprint: AgentBlueprint, output_dir: Path | str = "agents") -> tuple[Path, Path]:
    """Create prompt + json config files for a newly defined sub-agent."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    legacy_files = _normalize_list(blueprint.files, _profile_for(blueprint.agent_type)["files"])
    prompt_text, config = _build_prompt_and_config(blueprint, legacy_files)
    config["files_to_create"] = legacy_files
    config["constraints"] = {
        "data_source": "MOEX ISS API",
        "if_missing_data": "Недостаточно данных",
        "deterministic": True,
        "moex_specifics": ["sessions", "liquidity", "T+2", "short_sales", "leverage"],
    }
    config["version"] = 2

    prompt_path = output_path / f"{blueprint.slug}_PROMPT.md"
    config_path = output_path / f"{blueprint.slug}.json"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prompt_path, config_path


def create_agent_bundle(blueprint: AgentBlueprint, output_dir: Path | str = "agents") -> Path:
    """Create executable agent bundle directory with config, docs, code and tests."""
    root = Path(output_dir)
    bundle_dir = root / blueprint.slug
    src_dir = bundle_dir / "src"
    tests_dir = bundle_dir / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "agent_config.json",
        "AGENT_PROMPT.md",
        "README.md",
        f"src/{blueprint.slug}.py",
        f"tests/test_{blueprint.slug}.py",
    ]

    prompt_text, config = _build_prompt_and_config(blueprint, files)

    class_name = _pascal_case_from_slug(blueprint.slug)
    src_path = src_dir / f"{blueprint.slug}.py"
    test_path = tests_dir / f"test_{blueprint.slug}.py"

    readme = (
        f"# {blueprint.name}\n\n"
        f"- **Role**: {blueprint.role}\n"
        f"- **Agent Type**: {blueprint.agent_type.lower()}\n"
        f"- **Timeframe**: {blueprint.timeframe}\n\n"
        "## Input Data\n" + "\n".join(f"- {item}" for item in config["input_data"]) + "\n\n"
        "## Output Data\n" + "\n".join(f"- {item}" for item in config["output_data"]) + "\n\n"
        "## Algorithms\n" + "\n".join(f"- {item}" for item in config["algorithms"]) + "\n\n"
        "## Risk Control\n" + "\n".join(f"- {item}" for item in config["risk_control"]) + "\n\n"
        "## File List\n" + "\n".join(f"- {item}" for item in files) + "\n"
    )

    source_code = f'''"""Executable scaffold for {blueprint.name}."""


_SEMANTIC_DEFAULTS = {_DEFAULT_SEMANTIC_PROVIDES!r}


class {class_name}:
    def __init__(self, config: dict):
        self.config = config

    def run(self, input_data: dict) -> dict:
        if not input_data:
            return {{"status": "error", "agent": "{blueprint.slug}", "message": "Недостаточно данных"}}

        agent_type = str(self.config.get("agent_type", "")).lower()
        provides = dict(_SEMANTIC_DEFAULTS.get(agent_type, {{}}))
        for token in self.config.get("dependencies", {{}}).get("provides", []):
            provides.setdefault(token, f"{blueprint.slug}:{{token}}")
        return {{"status": "ok", "agent": "{blueprint.slug}", "provides": provides}}
'''

    test_code = f'''import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "{blueprint.slug}.py"


def _load_agent_class():
    spec = importlib.util.spec_from_file_location("agent_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.{class_name}


def test_agent_instantiates():
    cls = _load_agent_class()
    agent = cls({{"name": "{blueprint.name}"}})
    assert agent is not None


def test_agent_returns_insufficient_data_on_empty_input():
    cls = _load_agent_class()
    agent = cls({{"name": "{blueprint.name}"}})
    result = agent.run({{}})
    assert result == {{"status": "error", "agent": "{blueprint.slug}", "message": "Недостаточно данных"}}
'''

    (bundle_dir / "agent_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / "AGENT_PROMPT.md").write_text(prompt_text, encoding="utf-8")
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")
    src_path.write_text(source_code, encoding="utf-8")
    test_path.write_text(test_code, encoding="utf-8")

    return bundle_dir

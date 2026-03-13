"""Runtime contracts and simulated execution for generated MOEX agent systems."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

from .orchestrator_loader import LoadedAgent, SystemRegistry, build_execution_plan
from .semantic_contracts import validate_semantic_output
from .sample_data import build_sample_context, build_sample_market_data
from .adapters.common import AdapterCompatibilityError


class AgentRuntimeError(RuntimeError):
    """Raised when runtime output violates contract."""


def validate_agent_result(result: dict, agent_slug: str, required_provides: List[str]) -> None:
    if not isinstance(result, dict):
        raise AgentRuntimeError(f"{agent_slug}: result must be dict")
    if "status" not in result:
        raise AgentRuntimeError(f"{agent_slug}: missing 'status'")
    if "agent" not in result:
        raise AgentRuntimeError(f"{agent_slug}: missing 'agent'")
    if result.get("agent") != agent_slug:
        raise AgentRuntimeError(f"{agent_slug}: result agent mismatch ({result.get('agent')})")

    status = result.get("status")
    if status == "ok":
        provides = result.get("provides")
        if not isinstance(provides, dict):
            raise AgentRuntimeError(f"{agent_slug}: 'provides' must be dict when status=ok")
        missing = [k for k in required_provides if k not in provides]
        if missing:
            raise AgentRuntimeError(f"{agent_slug}: missing provided tokens: {', '.join(missing)}")
    elif status == "error":
        if "message" not in result:
            raise AgentRuntimeError(f"{agent_slug}: missing 'message' when status=error")
    else:
        raise AgentRuntimeError(f"{agent_slug}: unsupported status '{status}'")


def _load_agent_config(agent: LoadedAgent) -> dict:
    return json.loads(agent.config_path.read_text(encoding="utf-8"))


def load_agent_runtime(agent: LoadedAgent):
    module_path = agent.config_path.parent / "src" / f"{agent.slug}.py"
    if not module_path.exists():
        raise AgentRuntimeError(f"{agent.slug}: runtime source not found at {module_path}")

    spec = importlib.util.spec_from_file_location(f"bundle_{agent.slug}", module_path)
    if not spec or not spec.loader:
        raise AgentRuntimeError(f"{agent.slug}: failed to create import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = "".join(part.capitalize() for part in agent.slug.split("-") if part)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AgentRuntimeError(f"{agent.slug}: class '{class_name}' not found in runtime module")

    cfg = _load_agent_config(agent)
    return cls(cfg), cfg


def _run_real_adapter(agent: LoadedAgent, cfg: dict, inputs: dict) -> dict:
    adapter = cfg.get("runtime_adapter")
    if not isinstance(adapter, dict):
        raise AgentRuntimeError(f"{agent.slug}: runtime_adapter is required for runtime_mode=real")

    module_name = adapter.get("module")
    callable_name = adapter.get("callable")
    if not isinstance(module_name, str) or not module_name:
        raise AgentRuntimeError(f"{agent.slug}: runtime_adapter.module must be non-empty string")
    if not isinstance(callable_name, str) or not callable_name:
        raise AgentRuntimeError(f"{agent.slug}: runtime_adapter.callable must be non-empty string")

    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        raise AgentRuntimeError(f"{agent.slug}: failed to import adapter module '{module_name}': {exc}") from exc

    fn = getattr(mod, callable_name, None)
    if fn is None or not callable(fn):
        raise AgentRuntimeError(f"{agent.slug}: adapter callable '{callable_name}' not found in '{module_name}'")

    try:
        return fn(cfg, inputs)
    except AdapterCompatibilityError as exc:
        raise AgentRuntimeError(f"{agent.slug}: adapter compatibility error: {exc}") from exc
    except Exception as exc:
        raise AgentRuntimeError(f"{agent.slug}: adapter execution failed: {exc}") from exc


def run_execution_plan_with_artifacts(
    system: SystemRegistry,
    runtime_mode_override: Optional[str] = None,
    with_sample_data: bool = False,
) -> tuple[List[dict], Dict[str, object], List[str]]:
    """Execute system plan and return outputs with final artifact store and executed slugs."""
    plan = build_execution_plan(system)

    artifact_store: Dict[str, object] = {}
    if with_sample_data:
        artifact_store["market_data"] = build_sample_market_data()
        artifact_store["context"] = build_sample_context()
    outputs: List[dict] = []
    executed_slugs: List[str] = []

    for agent in plan:
        runtime, cfg = load_agent_runtime(agent)
        mode = runtime_mode_override or cfg.get("runtime_mode", "stub")

        if mode == "stub":
            result = runtime.run({"artifacts": dict(artifact_store)})
        elif mode == "real":
            result = _run_real_adapter(agent, cfg, {"artifacts": dict(artifact_store)})
        else:
            raise AgentRuntimeError(f"{agent.slug}: invalid runtime_mode '{mode}'")

        validate_agent_result(result, agent.slug, agent.dependencies_provides)

        if result.get("status") == "ok":
            provides = result.get("provides", {})
            validate_semantic_output(agent.agent_type, provides)
            artifact_store.update(provides)

        outputs.append(result)
        executed_slugs.append(agent.slug)

        if result.get("status") == "error":
            raise AgentRuntimeError(f"{agent.slug}: {result.get('message', 'runtime error')}")

    return outputs, dict(artifact_store), executed_slugs


def run_execution_plan(system: SystemRegistry, runtime_mode_override: Optional[str] = None, with_sample_data: bool = False) -> List[dict]:
    outputs, _, _ = run_execution_plan_with_artifacts(
        system,
        runtime_mode_override=runtime_mode_override,
        with_sample_data=with_sample_data,
    )
    return outputs

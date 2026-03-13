"""Loader for generated MOEX agent systems (registry + bundles)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


REQUIRED_AGENT_TYPES = [
    "data",
    "feature",
    "ml",
    "risk",
    "strategy",
    "portfolio",
    "execution",
    "monitoring",
]


class SystemRegistryError(ValueError):
    """Raised when a generated system registry cannot be loaded/validated."""


@dataclass
class LoadedAgent:
    name: str
    slug: str
    agent_type: str
    role: str
    timeframe: str
    config_path: Path
    files: List[str]
    dependencies_requires: List[str]
    dependencies_provides: List[str]


@dataclass
class SystemRegistry:
    schema_version: str
    preset: str
    generated_at: str
    agents: List[LoadedAgent]
    agents_by_type: Dict[str, List[LoadedAgent]]
    status: str
    missing_types: List[str]


def _require_top_level(payload: dict, path: Path) -> None:
    for field in ("schema_version", "preset", "generated_at", "agents"):
        if field not in payload:
            raise SystemRegistryError(f"{path}: missing top-level field '{field}'")
    if not isinstance(payload["agents"], list):
        raise SystemRegistryError(f"{path}: 'agents' must be a list")


def _validate_bundle_layout(config_path: Path) -> None:
    if not config_path.exists():
        raise SystemRegistryError(f"missing config_path: {config_path}")

    bundle_dir = config_path.parent
    required = [
        bundle_dir / "AGENT_PROMPT.md",
        bundle_dir / "README.md",
        bundle_dir / "src",
        bundle_dir / "tests",
    ]
    for item in required:
        if not item.exists():
            raise SystemRegistryError(f"bundle validation failed, missing: {item}")


def _load_agent(config_path: Path) -> LoadedAgent:
    _validate_bundle_layout(config_path)

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemRegistryError(f"{config_path}: invalid json ({exc})") from exc

    for field in ("name", "slug", "agent_type", "role", "timeframe", "files", "dependencies"):
        if field not in payload:
            raise SystemRegistryError(f"{config_path}: missing field '{field}'")

    files = payload.get("files")
    if not isinstance(files, list):
        raise SystemRegistryError(f"{config_path}: 'files' must be a list")

    deps = payload.get("dependencies")
    if not isinstance(deps, dict):
        raise SystemRegistryError(f"{config_path}: 'dependencies' must be an object")
    req = deps.get("requires")
    prv = deps.get("provides")
    if not isinstance(req, list) or not isinstance(prv, list):
        raise SystemRegistryError(f"{config_path}: dependencies.requires/provides must be lists")

    return LoadedAgent(
        name=str(payload["name"]),
        slug=str(payload["slug"]),
        agent_type=str(payload["agent_type"]),
        role=str(payload["role"]),
        timeframe=str(payload["timeframe"]),
        config_path=config_path,
        files=[str(x) for x in files],
        dependencies_requires=[str(x) for x in req],
        dependencies_provides=[str(x) for x in prv],
    )


def load_system_registry(path: Path) -> SystemRegistry:
    """Load registry.json and normalize it for orchestration."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemRegistryError(f"{path}: invalid json ({exc})") from exc

    _require_top_level(payload, path)

    root = path.parent
    agents: List[LoadedAgent] = []
    for item in payload["agents"]:
        if not isinstance(item, dict):
            raise SystemRegistryError(f"{path}: each item in 'agents' must be an object")
        config_rel = item.get("config_path")
        if not isinstance(config_rel, str) or not config_rel.strip():
            raise SystemRegistryError(f"{path}: each agent must have non-empty 'config_path'")
        config_path = (root / config_rel).resolve()
        agents.append(_load_agent(config_path))

    by_type: Dict[str, List[LoadedAgent]] = {k: [] for k in REQUIRED_AGENT_TYPES}
    for agent in agents:
        by_type.setdefault(agent.agent_type, []).append(agent)

    missing = [t for t in REQUIRED_AGENT_TYPES if not by_type.get(t)]
    status = "READY" if not missing else "INCOMPLETE"

    return SystemRegistry(
        schema_version=str(payload["schema_version"]),
        preset=str(payload["preset"]),
        generated_at=str(payload["generated_at"]),
        agents=agents,
        agents_by_type=by_type,
        status=status,
        missing_types=missing,
    )


def build_execution_plan(system: SystemRegistry) -> List[LoadedAgent]:
    """Build deterministic dependency-aware execution order."""
    if system.status != "READY":
        missing = ", ".join(system.missing_types) if system.missing_types else "unknown"
        raise SystemRegistryError(f"system is incomplete, missing types: {missing}")

    agents = sorted(system.agents, key=lambda a: a.slug)
    by_slug = {a.slug: a for a in agents}

    provided_by: Dict[str, Set[str]] = {}
    for agent in agents:
        for token in agent.dependencies_provides:
            provided_by.setdefault(token, set()).add(agent.slug)

    # Validate that every dependency token has a provider
    unresolved: List[str] = []
    for agent in agents:
        for token in agent.dependencies_requires:
            providers = provided_by.get(token, set())
            if not providers:
                unresolved.append(f"{agent.slug} requires '{token}' but no provider exists")
    if unresolved:
        raise SystemRegistryError("dependency resolution failed: " + "; ".join(unresolved))

    # Build DAG edges provider -> consumer
    incoming: Dict[str, Set[str]] = {a.slug: set() for a in agents}
    outgoing: Dict[str, Set[str]] = {a.slug: set() for a in agents}
    for agent in agents:
        for token in agent.dependencies_requires:
            for provider_slug in sorted(provided_by.get(token, set())):
                if provider_slug == agent.slug:
                    continue
                incoming[agent.slug].add(provider_slug)
                outgoing[provider_slug].add(agent.slug)

    ready = sorted([slug for slug, deps in incoming.items() if not deps])
    ordered_slugs: List[str] = []

    while ready:
        slug = ready.pop(0)
        ordered_slugs.append(slug)

        for nxt in sorted(outgoing[slug]):
            incoming[nxt].discard(slug)
            if not incoming[nxt] and nxt not in ordered_slugs and nxt not in ready:
                ready.append(nxt)
                ready.sort()

    if len(ordered_slugs) != len(agents):
        remaining = sorted(set(by_slug.keys()) - set(ordered_slugs))
        raise SystemRegistryError(f"dependency cycle detected among agents: {', '.join(remaining)}")

    return [by_slug[slug] for slug in ordered_slugs]

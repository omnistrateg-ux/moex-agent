"""Agent registry loading/validation utilities for scaffolded MOEX agents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .agent_factory import AGENT_TYPES


@dataclass
class AgentSpec:
    """Validated agent specification loaded from a json config."""

    name: str
    slug: str
    agent_type: str
    role: str
    timeframe: str
    risk_profile: str
    input_data: List[str]
    output_data: List[str]
    algorithms: List[str]
    risk_control: List[str]
    files: List[str]
    source_file: Path


class RegistryValidationError(ValueError):
    """Raised when one or more registry files are invalid."""


def _validate_non_empty_list(value, field: str, source: Path) -> List[str]:
    if not isinstance(value, list) or not value:
        raise RegistryValidationError(f"{source}: field '{field}' must be a non-empty list")
    cleaned = [str(v).strip() for v in value if str(v).strip()]
    if not cleaned:
        raise RegistryValidationError(f"{source}: field '{field}' must contain non-empty strings")
    return cleaned


def validate_agent_spec(path: Path) -> List[str]:
    """Validate a single agent JSON specification and return error messages."""
    errors: List[str] = []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid json: {exc}"]

    required_fields = (
        "name",
        "slug",
        "agent_type",
        "input_data",
        "output_data",
        "algorithms",
        "risk_control",
        "files",
    )

    for field in required_fields:
        if field not in payload:
            errors.append(f"missing field: {field}")

    name = payload.get("name")
    if "name" in payload and (not isinstance(name, str) or not name.strip()):
        errors.append("name must be a non-empty string")

    slug = payload.get("slug")
    if "slug" in payload:
        if not isinstance(slug, str) or not slug.strip():
            errors.append("slug must be a non-empty string")
        else:
            if slug != slug.lower():
                errors.append("slug must be lowercase")
            if " " in slug:
                errors.append("slug must not contain spaces")

    agent_type = payload.get("agent_type")
    if "agent_type" in payload:
        if not isinstance(agent_type, str) or not agent_type.strip():
            errors.append("agent_type must be a non-empty string")
        elif agent_type not in AGENT_TYPES:
            errors.append(f"invalid agent_type: {agent_type}")

    for field in ("input_data", "output_data", "algorithms", "risk_control", "files"):
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, list) or len(value) == 0:
            errors.append(f"{field} must be a non-empty list")

    schema_version = payload.get("schema_version")
    if schema_version is not None and not isinstance(schema_version, str):
        errors.append("schema_version must be a string")

    return errors


def _load_one(path: Path) -> AgentSpec:
    errors = validate_agent_spec(path)
    if errors:
        raise RegistryValidationError(f"{path}: " + "; ".join(errors))

    payload = json.loads(path.read_text(encoding="utf-8"))

    for field in ("role", "timeframe"):
        value = str(payload.get(field, "")).strip()
        if not value:
            raise RegistryValidationError(f"{path}: field '{field}' is required and must be non-empty")

    risk_profile = str(payload.get("risk_profile", "balanced")).strip() or "balanced"

    files = payload.get("files")
    if files is None:
        files = payload.get("files_to_create")
    if files is None:
        raise RegistryValidationError(f"{path}: field 'files' is required")

    return AgentSpec(
        name=str(payload["name"]).strip(),
        slug=str(payload["slug"]).strip(),
        agent_type=str(payload["agent_type"]).strip().lower(),
        role=str(payload["role"]).strip(),
        timeframe=str(payload["timeframe"]).strip(),
        risk_profile=risk_profile,
        input_data=_validate_non_empty_list(payload.get("input_data"), "input_data", path),
        output_data=_validate_non_empty_list(payload.get("output_data"), "output_data", path),
        algorithms=_validate_non_empty_list(payload.get("algorithms"), "algorithms", path),
        risk_control=_validate_non_empty_list(payload.get("risk_control"), "risk_control", path),
        files=_validate_non_empty_list(files, "files", path),
        source_file=path,
    )


def _discover_config_paths(root: Path) -> List[Path]:
    candidates = set(root.glob("*.json"))
    candidates.update(root.rglob("agent_config.json"))
    paths = sorted(path for path in candidates if path.name != "registry.json")
    return paths


def load_registry(agents_dir: Path | str = "agents") -> List[AgentSpec]:
    """Load and validate generated agent configs from directory tree."""
    root = Path(agents_dir)
    if not root.exists():
        raise RegistryValidationError(f"agents directory does not exist: {root}")

    specs: List[AgentSpec] = []
    errors: List[str] = []
    for path in _discover_config_paths(root):
        try:
            specs.append(_load_one(path))
        except RegistryValidationError as exc:
            errors.append(str(exc))

    if not specs and not errors:
        raise RegistryValidationError(f"no agent config json files found in: {root}")

    if errors:
        raise RegistryValidationError("\n".join(errors))

    slugs = [spec.slug for spec in specs]
    duplicate_slugs = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicate_slugs:
        raise RegistryValidationError(f"duplicate slugs found: {', '.join(duplicate_slugs)}")

    return specs


def build_dry_run_plan(specs: List[AgentSpec]) -> str:
    """Build deterministic orchestration plan text for review without execution."""
    by_type = {agent_type: [] for agent_type in AGENT_TYPES}
    for spec in specs:
        by_type[spec.agent_type].append(spec)

    lines = ["MOEX META-AGENT DRY-RUN PLAN", "=" * 32]
    for agent_type in AGENT_TYPES:
        rows = by_type[agent_type]
        if not rows:
            lines.append(f"- {agent_type}: MISSING")
            continue
        lines.append(f"- {agent_type}: {len(rows)} agent(s)")
        for row in rows:
            lines.append(f"  • {row.name} ({row.slug}) | tf={row.timeframe} | risk={row.risk_profile}")

    missing = [agent_type for agent_type, rows in by_type.items() if not rows]
    if missing:
        lines.append("")
        lines.append("RESULT: INCOMPLETE (missing required agent types)")
        lines.append(f"Missing: {', '.join(missing)}")
    else:
        lines.append("")
        lines.append("RESULT: READY (all required agent types present)")

    return "\n".join(lines)

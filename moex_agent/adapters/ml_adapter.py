"""Real-bound ML adapter using project predictor/model registry."""

from __future__ import annotations

import numpy as np

from .common import AdapterCompatibilityError, require_symbol


_DIAG_OK = "OK"
_DIAG_ERROR = "ERROR"


def get_real_adapter_diagnostics(config: dict | None = None) -> list[tuple[str, str]]:
    checks = [
        ("moex_agent.predictor", "FEATURE_COLS"),
        ("moex_agent.predictor", "get_registry"),
    ]
    out: list[tuple[str, str]] = []
    for module_name, symbol_name in checks:
        try:
            require_symbol(module_name, symbol_name)
            out.append((_DIAG_OK, f"{module_name}.{symbol_name} available"))
        except AdapterCompatibilityError as exc:
            out.append((_DIAG_ERROR, str(exc)))
    return out


def run_agent(config: dict, inputs: dict) -> dict:
    slug = config.get("slug", "unknown")
    diag = get_real_adapter_diagnostics(config)
    errors = [m for lvl, m in diag if lvl == _DIAG_ERROR]
    if errors:
        raise AdapterCompatibilityError("; ".join(errors))

    FEATURE_COLS = require_symbol("moex_agent.predictor", "FEATURE_COLS")
    get_registry = require_symbol("moex_agent.predictor", "get_registry")

    artifacts = (inputs or {}).get("artifacts", {})
    features = artifacts.get("features")
    if not isinstance(features, dict):
        return {"status": "error", "agent": slug, "message": "features is required"}

    vec = np.zeros((1, len(FEATURE_COLS)), dtype=float)
    for idx, name in enumerate(FEATURE_COLS):
        val = features.get(name)
        if isinstance(val, (int, float)):
            vec[0, idx] = float(val)

    registry = get_registry()
    try:
        best_h, best_p = registry.best_horizon(vec)
    except Exception as exc:
        raise AdapterCompatibilityError(f"predictor scoring failed: {exc}") from exc

    p_long = float(best_p if best_h is not None else 0.5)
    p_short = float(max(0.0, min(1.0, 1.0 - p_long)))
    confidence = float(abs(p_long - p_short))
    signals = {
        "probability_long": round(p_long, 6),
        "probability_short": round(p_short, 6),
        "confidence": round(confidence, 6),
    }
    return {"status": "ok", "agent": slug, "provides": {"signals": signals}}

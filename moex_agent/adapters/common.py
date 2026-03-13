"""Shared helpers for real adapter dependency checks."""

from __future__ import annotations

import importlib
from typing import Any


class AdapterCompatibilityError(RuntimeError):
    """Raised when a real adapter dependency is unavailable/incompatible."""


def require_symbol(module_name: str, symbol_name: str) -> Any:
    """Import module and return required symbol or raise compatibility error."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        raise AdapterCompatibilityError(
            f"missing adapter dependency module '{module_name}': {exc}"
        ) from exc

    sym = getattr(mod, symbol_name, None)
    if sym is None:
        raise AdapterCompatibilityError(
            f"missing adapter dependency symbol '{module_name}.{symbol_name}'"
        )
    return sym

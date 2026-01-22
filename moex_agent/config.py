"""
MOEX Agent Configuration (Backward Compatibility Layer)

DEPRECATED: Import from config_schema instead.

    from moex_agent.config_schema import AppConfig, load_config

This module re-exports the new Pydantic-based config for backward compatibility.
"""
from __future__ import annotations

import warnings

from .config_schema import (
    AppConfig,
    HorizonConfig,
    load_config,
    Config,
    StorageConfig,
    UniverseConfig,
    SignalsConfig,
    RiskConfig,
    QwenConfig,
    TelegramConfig,
)

__all__ = [
    "AppConfig",
    "Config",
    "HorizonConfig",
    "load_config",
    "StorageConfig",
    "UniverseConfig",
    "SignalsConfig",
    "RiskConfig",
    "QwenConfig",
    "TelegramConfig",
]

# Alias for backward compatibility
Horizon = HorizonConfig


def _deprecated_warning():
    warnings.warn(
        "Importing from moex_agent.config is deprecated. "
        "Use moex_agent.config_schema instead.",
        DeprecationWarning,
        stacklevel=3,
    )

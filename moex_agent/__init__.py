"""
MOEX Agent - Trading Signal Generator for Moscow Exchange

A real-time trading signal generator that detects price/volume anomalies,
predicts continuation probability via ML models, and sends alerts to Telegram.

Usage:
    python -m moex_agent init-db
    python -m moex_agent bootstrap --days 180
    python -m moex_agent train
    python -m moex_agent live
    python -m moex_agent web --port 8000
"""

__version__ = "1.0.0"

# Core exports
from .config_schema import AppConfig, load_config
from .storage import connect, database
from .engine import PipelineEngine, Signal, CycleResult
from .predictor import ModelRegistry, safe_predict_proba

__all__ = [
    # Version
    "__version__",
    # Config
    "AppConfig",
    "load_config",
    # Storage
    "connect",
    "database",
    # Engine
    "PipelineEngine",
    "Signal",
    "CycleResult",
    # Predictor
    "ModelRegistry",
    "safe_predict_proba",
]

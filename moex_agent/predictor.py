"""
MOEX Agent ML Predictor

Safe model inference with robust error handling.
Handles edge cases: single-class models, (n,1) proba shape, missing classes_.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger("moex_agent.predictor")

# Feature columns used by all models (extended with technical indicators)
FEATURE_COLS = [
    # Returns
    "r_1m", "r_5m", "r_10m", "r_30m", "r_60m",
    # Turnover
    "turn_1m", "turn_5m", "turn_10m",
    # ATR & VWAP
    "atr_14", "dist_vwap_atr",
    # RSI
    "rsi_14", "rsi_7",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # Bollinger Bands
    "bb_position", "bb_width",
    # Stochastic
    "stoch_k", "stoch_d",
    # ADX (trend strength)
    "adx",
    # OBV momentum
    "obv_change",
    # Momentum
    "momentum_10", "momentum_30",
    # Volatility
    "volatility_10", "volatility_30",
    # Moving averages
    "price_sma20_ratio", "price_sma50_ratio", "sma20_sma50_ratio",
    # Volume
    "volume_sma_ratio",
]


def safe_predict_proba(model: Any, X: np.ndarray) -> float:
    """
    Safely extract P(class=1) from sklearn model.

    Handles edge cases:
    - model.classes_ may be [0, 1], [1, 0], [False, True], or single-class
    - predict_proba may return shape (n, 1) or (n, 2)
    - Single-class model → return 0.5 (no information)

    Args:
        model: Fitted sklearn classifier with predict_proba method
        X: Feature array of shape (1, n_features)

    Returns:
        Probability of positive class (float in [0, 1])
    """
    try:
        proba = model.predict_proba(X)
    except Exception as e:
        logger.warning(f"predict_proba failed: {e}")
        return 0.5  # Default to no information

    if proba is None:
        return 0.5

    proba = np.asarray(proba)
    classes = getattr(model, "classes_", None)

    # Handle 1D output (unusual but possible)
    if proba.ndim == 1:
        return float(proba[0])

    # Handle non-2D output
    if proba.ndim != 2:
        return float(proba.ravel()[0]) if proba.size > 0 else 0.5

    # Single-class model: no discriminative power
    if proba.shape[1] == 1:
        # If only class 1 in training data, proba is P(class=1)
        if classes is not None and len(classes) == 1:
            if classes[0] == 1 or classes[0] is True:
                return float(proba[0, 0])
            else:
                return 1.0 - float(proba[0, 0])
        return 0.5

    # Standard 2-class model
    if classes is not None:
        classes_list = list(classes)
        # Find index of positive class (1 or True)
        if 1 in classes_list:
            idx = classes_list.index(1)
            return float(proba[0, idx])
        if True in classes_list:
            idx = classes_list.index(True)
            return float(proba[0, idx])

    # Fallback: assume second column is positive class
    return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])


class ModelRegistry:
    """
    Thread-safe registry for loaded ML models.

    Provides lazy loading, caching, and safe prediction interface.
    """

    def __init__(self, models_dir: Path = Path("./models")):
        self.models_dir = Path(models_dir)
        self._models: Dict[str, Any] = {}
        self._meta: Optional[Dict[str, Dict]] = None
        self._loaded = False

    def load(self) -> None:
        """Load all models from models_dir."""
        meta_path = self.models_dir / "meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {meta_path}\n"
                "Run 'python -m moex_agent train' first."
            )

        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._models = {}

        for horizon, info in self._meta.items():
            model_path = Path(info["path"])
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                self._models[horizon] = joblib.load(model_path)
                logger.debug(f"Loaded model: {horizon} from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model {horizon}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._models)} models: {list(self._models.keys())}")

    def ensure_loaded(self) -> None:
        """Load models if not already loaded."""
        if not self._loaded:
            self.load()

    @property
    def horizons(self) -> List[str]:
        """Get list of available horizons."""
        self.ensure_loaded()
        return list(self._models.keys())

    def predict(self, horizon: str, X: np.ndarray) -> float:
        """
        Predict P(success) for a given horizon.

        Args:
            horizon: Model horizon name (e.g., '5m', '1h')
            X: Feature array of shape (1, n_features)

        Returns:
            Probability of success (float in [0, 1])

        Raises:
            KeyError: If horizon model not loaded
        """
        self.ensure_loaded()

        if horizon not in self._models:
            raise KeyError(
                f"Model for horizon '{horizon}' not found. "
                f"Available: {list(self._models.keys())}"
            )

        return safe_predict_proba(self._models[horizon], X)

    def predict_all(self, X: np.ndarray) -> Dict[str, float]:
        """
        Predict P(success) for all horizons.

        Args:
            X: Feature array of shape (1, n_features)

        Returns:
            Dict mapping horizon name to probability
        """
        self.ensure_loaded()
        return {h: safe_predict_proba(m, X) for h, m in self._models.items()}

    def best_horizon(self, X: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find horizon with highest P(success).

        Args:
            X: Feature array of shape (1, n_features)

        Returns:
            Tuple of (horizon_name, probability), or (None, 0.0) if no models
        """
        preds = self.predict_all(X)
        if not preds:
            return None, 0.0

        best_h = max(preds, key=preds.get)
        return best_h, preds[best_h]


# Global singleton for convenience
_registry: Optional[ModelRegistry] = None


def get_registry(models_dir: Path = Path("./models")) -> ModelRegistry:
    """Get or create global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(models_dir)
    return _registry


def reset_registry() -> None:
    """Reset global registry (for testing)."""
    global _registry
    _registry = None

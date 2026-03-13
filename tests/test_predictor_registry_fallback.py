from __future__ import annotations

import json
from pathlib import Path

import joblib

from moex_agent.predictor import ModelRegistry


class _DummyModel:
    def __init__(self) -> None:
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.3, 0.7] for _ in range(len(X))]


def test_model_registry_loads_orphan_models(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # Empty metadata: no model entries.
    (models_dir / "meta.json").write_text(json.dumps({}), encoding="utf-8")

    # A model file present on disk should still be discoverable.
    joblib.dump(_DummyModel(), models_dir / "model_time_1d.joblib")

    registry = ModelRegistry(models_dir=models_dir)
    registry.load()

    assert "1d" in registry.horizons

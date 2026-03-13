from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


DEFAULT_META = {"schema_version": "1.0", "horizons": {}}


def _write_config(path: Path, sqlite_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "storage:",
                f"  sqlite_path: \"{sqlite_path}\"",
                "universe:",
                "  tickers:",
                "    - SBER",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_repair(config_path: Path, models_dir: Path, registry_path: Path, output_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "--config",
        str(config_path),
        "repair",
        "--models-dir",
        str(models_dir),
        "--registry",
        str(registry_path),
    ]
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    return subprocess.run(cmd, capture_output=True, text=True)


def test_repair_missing_db_and_tables(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "db" / "runtime.sqlite"
    config = tmp_path / "config.yaml"
    models = tmp_path / "models"
    _write_config(config, sqlite_path)

    res = _run_repair(config, models, tmp_path / "registry.json")
    assert res.returncode == 0, res.stdout + res.stderr
    assert sqlite_path.exists()

    conn = sqlite3.connect(sqlite_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"candles", "quotes", "alerts", "state"}.issubset(tables)


def test_repair_invalid_meta_json_replaced(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite"
    config = tmp_path / "config.yaml"
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "meta.json").write_text("{invalid", encoding="utf-8")
    _write_config(config, sqlite_path)

    res = _run_repair(config, models, tmp_path / "registry.json")
    assert res.returncode == 0, res.stdout + res.stderr

    payload = json.loads((models / "meta.json").read_text(encoding="utf-8"))
    assert payload == DEFAULT_META


def test_repair_orphan_model_files_synced_into_meta(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite"
    config = tmp_path / "config.yaml"
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "meta.json").write_text(json.dumps(DEFAULT_META), encoding="utf-8")
    (models / "model_time_1d.joblib").write_bytes(b"x")
    (models / "model_time_5m.joblib").write_bytes(b"x")
    _write_config(config, sqlite_path)

    res = _run_repair(config, models, tmp_path / "registry.json")
    assert res.returncode == 0, res.stdout + res.stderr

    payload = json.loads((models / "meta.json").read_text(encoding="utf-8"))
    assert sorted(payload["horizons"].keys()) == ["1d", "5m"]


def test_repair_rebuilds_registry_from_bundles(tmp_path: Path) -> None:
    system_dir = tmp_path / "system"
    create_cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "create-system",
        "--preset",
        "core_moex_v1",
        "--output-dir",
        str(system_dir),
    ]
    created = subprocess.run(create_cmd, capture_output=True, text=True)
    assert created.returncode == 0, created.stdout + created.stderr

    registry = system_dir / "registry.json"
    registry.unlink()

    sqlite_path = tmp_path / "runtime.sqlite"
    config = tmp_path / "config.yaml"
    models = tmp_path / "models"
    _write_config(config, sqlite_path)

    res = _run_repair(config, models, registry, output_dir=system_dir)
    assert res.returncode == 0, res.stdout + res.stderr
    assert registry.exists()

    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["generated_at"] == "static"
    assert len(payload["agents"]) == 8


def test_repair_idempotent_second_run(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite"
    config = tmp_path / "config.yaml"
    models = tmp_path / "models"
    _write_config(config, sqlite_path)

    first = _run_repair(config, models, tmp_path / "registry.json")
    assert first.returncode == 0, first.stdout + first.stderr

    second = _run_repair(config, models, tmp_path / "registry.json")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Overall" in second.stdout
    assert "EXISTS:" in second.stdout or "SKIPPED:" in second.stdout

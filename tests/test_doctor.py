from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


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


def _run_doctor(config_path: Path, models_dir: Path, registry_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "--config",
        str(config_path),
        "doctor",
        "--models-dir",
        str(models_dir),
        "--registry",
        str(registry_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_doctor_empty_sqlite_reports_missing_tables_warning(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "empty.sqlite"
    sqlite3.connect(sqlite_path).close()

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, sqlite_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "meta.json").write_text("{}", encoding="utf-8")

    result = _run_doctor(config_path, models_dir, tmp_path / "missing_registry.json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Storage" in result.stdout
    assert "WARNING: missing required tables" in result.stdout


def test_doctor_orphan_model_file_reports_warning(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "ok.sqlite"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE candles(ts TEXT)")
    conn.execute("CREATE TABLE quotes(ts TEXT)")
    conn.execute("CREATE TABLE alerts(id INTEGER)")
    conn.execute("CREATE TABLE state(key TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, sqlite_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "meta.json").write_text("{}", encoding="utf-8")
    (models_dir / "model_time_1d.joblib").write_bytes(b"placeholder")

    result = _run_doctor(config_path, models_dir, tmp_path / "missing_registry.json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING: model files missing in meta.json: 1d" in result.stdout


def test_doctor_invalid_meta_json_reports_error(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "ok.sqlite"
    sqlite3.connect(sqlite_path).close()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, sqlite_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "meta.json").write_text("{invalid", encoding="utf-8")

    result = _run_doctor(config_path, models_dir, tmp_path / "missing_registry.json")

    assert result.returncode == 1
    assert "ERROR: meta.json invalid JSON" in result.stdout
    assert "Overall" in result.stdout and "ERROR:" in result.stdout


def test_doctor_healthy_generated_system_reports_ok(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "ready.sqlite"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE candles(ts TEXT)")
    conn.execute("CREATE TABLE quotes(ts TEXT)")
    conn.execute("CREATE TABLE alerts(id INTEGER)")
    conn.execute("CREATE TABLE state(key TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, sqlite_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_time_5m.joblib").write_bytes(b"placeholder")
    meta = {"5m": {"path": str(models_dir / "model_time_5m.joblib")}}
    (models_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

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

    result = _run_doctor(config_path, models_dir, system_dir / "registry.json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: execution plan READY" in result.stdout
    assert "Overall" in result.stdout
    assert "OK: runtime readiness OK" in result.stdout

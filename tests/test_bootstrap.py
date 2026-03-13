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


def _run_bootstrap(config_path: Path, models_dir: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "moex_agent",
        "--config",
        str(config_path),
        "bootstrap",
        "--models-dir",
        str(models_dir),
    ]
    if extra:
        cmd.extend(extra)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_bootstrap_creates_sqlite_file_and_tables(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "data" / "runtime.sqlite"
    config_path = tmp_path / "config.yaml"
    models_dir = tmp_path / "models"
    _write_config(config_path, sqlite_path)

    result = _run_bootstrap(config_path, models_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sqlite_path.exists()

    conn = sqlite3.connect(sqlite_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()

    for required in {"candles", "quotes", "alerts", "state"}:
        assert required in tables


def test_bootstrap_creates_models_dir_and_meta_json(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite"
    config_path = tmp_path / "config.yaml"
    models_dir = tmp_path / "models"
    _write_config(config_path, sqlite_path)

    result = _run_bootstrap(config_path, models_dir)
    assert result.returncode == 0, result.stdout + result.stderr

    meta_path = models_dir / "meta.json"
    assert models_dir.exists()
    assert meta_path.exists()

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload == {"schema_version": "1.0", "horizons": {}}


def test_bootstrap_is_idempotent_on_second_run(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite"
    config_path = tmp_path / "config.yaml"
    models_dir = tmp_path / "models"
    _write_config(config_path, sqlite_path)

    first = _run_bootstrap(config_path, models_dir)
    assert first.returncode == 0, first.stdout + first.stderr

    second = _run_bootstrap(config_path, models_dir)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "EXISTS:" in second.stdout
    assert "Overall" in second.stdout


def test_bootstrap_with_create_system_creates_registry_and_bundles(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite"
    config_path = tmp_path / "config.yaml"
    models_dir = tmp_path / "models"
    out_dir = tmp_path / "system"
    _write_config(config_path, sqlite_path)

    result = _run_bootstrap(
        config_path,
        models_dir,
        ["--create-system", "--preset", "core_moex_v1", "--output-dir", str(out_dir)],
    )
    assert result.returncode == 0, result.stdout + result.stderr

    registry = out_dir / "registry.json"
    assert registry.exists()

    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert payload.get("preset") == "core_moex_v1"
    assert isinstance(payload.get("agents"), list)
    assert len(payload["agents"]) == 8

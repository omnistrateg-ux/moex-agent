from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(tmp_cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_repo_root())
    cmd = [sys.executable, "-m", "moex_agent", *args]
    return subprocess.run(cmd, cwd=tmp_cwd, env=env, capture_output=True, text=True)




def _ensure_models(tmp_cwd: Path) -> None:
    src = _repo_root() / "models"
    dst = tmp_cwd / "models"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["meta.json", "model_time_5m.joblib", "model_time_10m.joblib", "model_time_30m.joblib", "model_time_1h.joblib"]:
        src_file = src / name
        if src_file.exists():
            (dst / name).write_bytes(src_file.read_bytes())


def _create_system(tmp_cwd: Path) -> Path:
    system_dir = tmp_cwd / "system"
    res = _run(tmp_cwd, ["create-system", "--preset", "core_moex_v1", "--output-dir", str(system_dir)])
    assert res.returncode == 0, res.stdout + res.stderr
    return system_dir


def _set_real_adapters(system_dir: Path) -> None:
    adapter_map = {
        "data-collector": "moex_agent.adapters.data_adapter",
        "feature-builder": "moex_agent.adapters.feature_adapter",
        "signal-model": "moex_agent.adapters.ml_adapter",
        "risk-guard": "moex_agent.adapters.risk_adapter",
        "strategy-planner": "moex_agent.adapters.strategy_adapter",
        "portfolio-allocator": "moex_agent.adapters.portfolio_adapter",
        "execution-router": "moex_agent.adapters.execution_adapter",
        "system-monitor": "moex_agent.adapters.monitoring_adapter",
    }
    for slug, module in adapter_map.items():
        cfg_path = system_dir / slug / "agent_config.json"
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        payload["runtime_mode"] = "real"
        payload["runtime_adapter"] = {"module": module, "callable": "run_agent"}
        cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_run(tmp_cwd: Path) -> Path:
    _ensure_models(tmp_cwd)
    system_dir = _create_system(tmp_cwd)
    _set_real_adapters(system_dir)
    runs_dir = tmp_cwd / "runs"
    res = _run(
        tmp_cwd,
        [
            "orchestrate",
            "--paper",
            "--registry",
            str(system_dir / "registry.json"),
            "--runs-dir",
            str(runs_dir),
            "--with-sample-data",
        ],
    )
    assert res.returncode == 0, res.stdout + res.stderr
    return runs_dir / "run_000001"


def test_mark_valid_run_as_baseline(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    res = _run(tmp_path, ["mark-baseline", "--run", str(run_dir), "--name", "main"])
    assert res.returncode == 0, res.stdout + res.stderr
    baselines_path = tmp_path / "baselines" / "baselines.json"
    payload = json.loads(baselines_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["baselines"]["main"] == str(run_dir)


def test_list_baselines_sorted(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    _run(tmp_path, ["mark-baseline", "--run", str(run_dir), "--name", "zeta"])
    _run(tmp_path, ["mark-baseline", "--run", str(run_dir), "--name", "alpha"])

    res = _run(tmp_path, ["list-baselines"])
    assert res.returncode == 0, res.stdout + res.stderr
    alpha_idx = res.stdout.index("OK: alpha ->")
    zeta_idx = res.stdout.index("OK: zeta ->")
    assert alpha_idx < zeta_idx


def test_run_regression_with_baseline_name(tmp_path: Path) -> None:
    baseline_run = _make_run(tmp_path)
    _run(tmp_path, ["mark-baseline", "--run", str(baseline_run), "--name", "main"])

    _ensure_models(tmp_path)
    system_dir = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir = tmp_path / "runs2"
    res_run = _run(
        tmp_path,
        [
            "orchestrate",
            "--paper",
            "--registry",
            str(system_dir / "registry.json"),
            "--runs-dir",
            str(runs_dir),
            "--with-sample-data",
        ],
    )
    assert res_run.returncode == 0, res_run.stdout + res_run.stderr

    candidate = runs_dir / "run_000001"
    res = _run(tmp_path, ["run-regression", "--baseline-name", "main", "--candidate", str(candidate)])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: regression passed" in res.stdout


def test_invalid_run_cannot_be_marked(tmp_path: Path) -> None:
    invalid_run = tmp_path / "bad_run"
    invalid_run.mkdir(parents=True, exist_ok=True)
    res = _run(tmp_path, ["mark-baseline", "--run", str(invalid_run), "--name", "bad"])
    assert res.returncode != 0
    assert "ERROR: baseline not updated" in res.stdout


def test_remark_same_name_updates_deterministically(tmp_path: Path) -> None:
    run1 = _make_run(tmp_path)
    # make second run under another runs dir to get different path
    _ensure_models(tmp_path)
    system_dir = _create_system(tmp_path)
    _set_real_adapters(system_dir)
    runs_dir2 = tmp_path / "runs_alt"
    res2 = _run(
        tmp_path,
        [
            "orchestrate",
            "--paper",
            "--registry",
            str(system_dir / "registry.json"),
            "--runs-dir",
            str(runs_dir2),
            "--with-sample-data",
        ],
    )
    assert res2.returncode == 0, res2.stdout + res2.stderr
    run2 = runs_dir2 / "run_000001"

    r1 = _run(tmp_path, ["mark-baseline", "--run", str(run1), "--name", "main"])
    r2 = _run(tmp_path, ["mark-baseline", "--run", str(run2), "--name", "main"])
    assert r1.returncode == 0, r1.stdout + r1.stderr
    assert r2.returncode == 0, r2.stdout + r2.stderr

    payload = json.loads((tmp_path / "baselines" / "baselines.json").read_text(encoding="utf-8"))
    assert payload["baselines"] == {"main": str(run2)}

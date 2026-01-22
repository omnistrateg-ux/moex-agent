"""
MOEX Agent Smoke Tests

Basic tests to verify all modules load and work correctly.
Run with: pytest tests/test_smoke.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────────────────────

def test_config_schema_imports():
    """Test config_schema module imports."""
    from moex_agent.config_schema import (
        AppConfig,
        HorizonConfig,
        StorageConfig,
        UniverseConfig,
        SignalsConfig,
        RiskConfig,
        QwenConfig,
        TelegramConfig,
        load_config,
    )
    assert AppConfig is not None
    assert load_config is not None


def test_config_loads():
    """Test config.yaml loads without errors."""
    from moex_agent.config_schema import load_config

    cfg = load_config()
    assert cfg is not None
    assert len(cfg.tickers) > 0
    assert cfg.poll_seconds > 0
    assert 0 < cfg.p_threshold < 1


def test_config_properties():
    """Test all config properties are accessible."""
    from moex_agent.config_schema import load_config

    cfg = load_config()

    # App-level
    assert isinstance(cfg.poll_seconds, int)
    assert isinstance(cfg.cooldown_minutes, int)
    assert isinstance(cfg.max_workers, int)

    # Sub-configs
    assert cfg.risk.max_spread_bps >= 0
    assert cfg.risk.min_turnover_rub_5m >= 0
    assert isinstance(cfg.qwen.enabled, bool)
    assert isinstance(cfg.telegram.enabled, bool)

    # Backward compat properties
    assert cfg.sqlite_path is not None
    assert cfg.engine == "stock"
    assert cfg.board == "TQBR"


# ─────────────────────────────────────────────────────────────
# Storage Tests
# ─────────────────────────────────────────────────────────────

def test_storage_imports():
    """Test storage module imports."""
    from moex_agent.storage import (
        connect,
        database,
        init_db,
        upsert_many,
        get_state,
        set_state,
        get_max_ts,
        get_window,
        get_recent_candles,
        save_alert,
        mark_alert_sent,
        get_alerts,
    )
    assert connect is not None
    assert database is not None


def test_storage_connect():
    """Test database connection."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect

    cfg = load_config()
    conn = connect(cfg.sqlite_path)
    assert conn is not None

    # Check WAL mode is enabled
    cur = conn.execute("PRAGMA journal_mode")
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"

    conn.close()


def test_storage_get_window():
    """Test get_window returns data."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect, get_window, get_max_ts

    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    max_ts = get_max_ts(conn)
    assert max_ts is not None

    df = get_window(conn, minutes=60)
    assert isinstance(df, pd.DataFrame)
    assert "secid" in df.columns
    assert "close" in df.columns

    conn.close()


def test_storage_context_manager():
    """Test database context manager."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import database

    cfg = load_config()

    with database(cfg.sqlite_path) as conn:
        cur = conn.execute("SELECT 1")
        assert cur.fetchone()[0] == 1


# ─────────────────────────────────────────────────────────────
# MOEX ISS Tests
# ─────────────────────────────────────────────────────────────

def test_moex_iss_imports():
    """Test moex_iss module imports."""
    from moex_agent.moex_iss import (
        Candle,
        fetch_candles,
        fetch_quote,
        close_session,
    )
    assert Candle is not None
    assert fetch_candles is not None


# ─────────────────────────────────────────────────────────────
# Predictor Tests
# ─────────────────────────────────────────────────────────────

def test_predictor_imports():
    """Test predictor module imports."""
    from moex_agent.predictor import (
        safe_predict_proba,
        ModelRegistry,
        FEATURE_COLS,
        get_registry,
        reset_registry,
    )
    assert safe_predict_proba is not None
    assert len(FEATURE_COLS) == 10


def test_predictor_model_registry():
    """Test ModelRegistry loads models."""
    from moex_agent.predictor import ModelRegistry

    registry = ModelRegistry()
    registry.load()

    assert len(registry.horizons) > 0
    assert "5m" in registry.horizons
    assert "1d" in registry.horizons


def test_predictor_predict():
    """Test prediction works."""
    from moex_agent.predictor import ModelRegistry, FEATURE_COLS

    registry = ModelRegistry()
    registry.load()

    # Create random feature vector
    X = np.random.randn(1, len(FEATURE_COLS))

    # Predict for each horizon
    for h in registry.horizons:
        p = registry.predict(h, X)
        assert 0 <= p <= 1, f"Probability {p} out of range for {h}"


def test_predictor_best_horizon():
    """Test best_horizon selection."""
    from moex_agent.predictor import ModelRegistry, FEATURE_COLS

    registry = ModelRegistry()
    registry.load()

    X = np.random.randn(1, len(FEATURE_COLS))
    best_h, best_p = registry.best_horizon(X)

    assert best_h in registry.horizons
    assert 0 <= best_p <= 1


# ─────────────────────────────────────────────────────────────
# Engine Tests
# ─────────────────────────────────────────────────────────────

def test_engine_imports():
    """Test engine module imports."""
    from moex_agent.engine import (
        PipelineEngine,
        Signal,
        CycleResult,
        create_engine,
    )
    assert PipelineEngine is not None
    assert Signal is not None


def test_engine_creation():
    """Test PipelineEngine creation."""
    from moex_agent.config_schema import load_config
    from moex_agent.engine import PipelineEngine

    cfg = load_config()
    engine = PipelineEngine(cfg)
    engine.load_models()

    assert len(engine.models.horizons) > 0
    assert engine.risk_params is not None


def test_engine_detect_anomalies():
    """Test anomaly detection."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect, get_window
    from moex_agent.engine import PipelineEngine

    cfg = load_config()
    conn = connect(cfg.sqlite_path)
    engine = PipelineEngine(cfg)

    df = get_window(conn, minutes=60*24*3)
    quotes = {secid: {"secid": secid} for secid in df["secid"].unique()}

    anomalies = engine.detect_anomalies(df, quotes)
    assert isinstance(anomalies, list)

    conn.close()


# ─────────────────────────────────────────────────────────────
# Anomaly Tests
# ─────────────────────────────────────────────────────────────

def test_anomaly_imports():
    """Test anomaly module imports."""
    from moex_agent.anomaly import (
        AnomalyResult,
        Direction,
        robust_z,
        compute_anomalies,
    )
    assert AnomalyResult is not None
    assert Direction is not None


def test_anomaly_robust_z():
    """Test robust z-score calculation."""
    from moex_agent.anomaly import robust_z

    # Normal distribution
    hist = np.random.randn(1000)
    z = robust_z(2.5, hist)
    assert isinstance(z, float)
    assert z > 0  # 2.5 should be positive z-score


def test_anomaly_direction_enum():
    """Test Direction enum."""
    from moex_agent.anomaly import Direction

    assert Direction.LONG.value == "LONG"
    assert Direction.SHORT.value == "SHORT"


# ─────────────────────────────────────────────────────────────
# Features Tests
# ─────────────────────────────────────────────────────────────

def test_features_imports():
    """Test features module imports."""
    from moex_agent.features import (
        build_feature_frame,
        compute_atr,
    )
    assert build_feature_frame is not None


def test_features_build():
    """Test feature building."""
    from moex_agent.features import build_feature_frame

    # Create sample data
    n = 200
    df = pd.DataFrame({
        "secid": ["TEST"] * n,
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": 100 + np.cumsum(np.random.randn(n) * 0.1),
        "high": 101 + np.cumsum(np.random.randn(n) * 0.1),
        "low": 99 + np.cumsum(np.random.randn(n) * 0.1),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.1),
        "value": np.random.randint(1000000, 10000000, n),
        "volume": np.random.randint(1000, 10000, n),
    })

    features = build_feature_frame(df)

    assert "r_1m" in features.columns
    assert "r_5m" in features.columns
    assert "atr_14" in features.columns
    assert "dist_vwap_atr" in features.columns


# ─────────────────────────────────────────────────────────────
# Risk Tests
# ─────────────────────────────────────────────────────────────

def test_risk_imports():
    """Test risk module imports."""
    from moex_agent.risk import (
        RiskParams,
        spread_bps,
        pass_gatekeeper,
    )
    assert RiskParams is not None


def test_risk_spread_bps():
    """Test spread calculation."""
    from moex_agent.risk import spread_bps

    # Normal spread
    s = spread_bps(100.0, 100.1)
    assert s is not None
    assert s > 0

    # None handling
    assert spread_bps(None, 100.0) is None
    assert spread_bps(100.0, None) is None


def test_risk_gatekeeper():
    """Test risk gatekeeper."""
    from moex_agent.risk import RiskParams, pass_gatekeeper

    risk = RiskParams(max_spread_bps=50, min_turnover_rub_5m=1000000)

    # Should pass
    assert pass_gatekeeper(
        p=0.45, p_threshold=0.35,
        turnover_5m=2000000, spread=20, risk=risk
    )

    # Should fail - low p
    assert not pass_gatekeeper(
        p=0.30, p_threshold=0.35,
        turnover_5m=2000000, spread=20, risk=risk
    )

    # Should fail - low turnover
    assert not pass_gatekeeper(
        p=0.45, p_threshold=0.35,
        turnover_5m=500000, spread=20, risk=risk
    )

    # Should fail - wide spread
    assert not pass_gatekeeper(
        p=0.45, p_threshold=0.35,
        turnover_5m=2000000, spread=100, risk=risk
    )


# ─────────────────────────────────────────────────────────────
# Qwen Tests
# ─────────────────────────────────────────────────────────────

def test_qwen_imports():
    """Test qwen module imports."""
    from moex_agent.qwen import (
        QwenAnalysis,
        analyze_signal,
        format_telegram_message,
    )
    assert QwenAnalysis is not None


def test_qwen_rule_based():
    """Test rule-based analysis (without LLM)."""
    from moex_agent.qwen import analyze_signal

    payload = {
        "ticker": "SBER",
        "direction": "LONG",
        "horizon": "5m",
        "p": 0.45,
        "anomaly": {
            "z_ret_5m": 2.0,
            "z_vol_5m": 1.5,
            "volume_spike": 1.8,
            "spread_bps": 15,
        }
    }

    result = analyze_signal(
        ollama_url="http://localhost:11434",
        model="test",
        payload=payload,
        use_rules_only=True,
    )

    assert result is not None
    assert result.recommendation in ["STRONG_BUY", "BUY", "WEAK_BUY", "SKIP"]


def test_qwen_skip_low_p():
    """Test rule skips low probability signals."""
    from moex_agent.qwen import analyze_signal

    payload = {
        "ticker": "TEST",
        "direction": "LONG",
        "horizon": "5m",
        "p": 0.20,  # Below threshold
        "anomaly": {"z_ret_5m": 2.0, "z_vol_5m": 1.5}
    }

    result = analyze_signal(
        ollama_url="http://localhost:11434",
        model="test",
        payload=payload,
        use_rules_only=True,
    )

    assert result.skip is True
    assert "p=" in result.skip_reason


# ─────────────────────────────────────────────────────────────
# Telegram Tests
# ─────────────────────────────────────────────────────────────

def test_telegram_imports():
    """Test telegram module imports."""
    from moex_agent.telegram import (
        send_telegram,
        send_signal_alert,
    )
    assert send_telegram is not None


# ─────────────────────────────────────────────────────────────
# Webapp Tests
# ─────────────────────────────────────────────────────────────

def test_webapp_imports():
    """Test webapp module imports."""
    from moex_agent.webapp import app
    assert app is not None


def test_webapp_health():
    """Test /api/health endpoint."""
    from fastapi.testclient import TestClient
    from moex_agent.webapp import app

    client = TestClient(app)
    r = client.get("/api/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_webapp_status():
    """Test /api/status endpoint."""
    from fastapi.testclient import TestClient
    from moex_agent.webapp import app

    client = TestClient(app)
    r = client.get("/api/status")

    assert r.status_code == 200
    data = r.json()
    assert "candles_count" in data
    assert "models_loaded" in data


def test_webapp_dashboard():
    """Test HTML dashboard."""
    from fastapi.testclient import TestClient
    from moex_agent.webapp import app

    client = TestClient(app)
    r = client.get("/")

    assert r.status_code == 200
    assert "MOEX Agent Dashboard" in r.text


# ─────────────────────────────────────────────────────────────
# CLI Tests
# ─────────────────────────────────────────────────────────────

def test_cli_main_imports():
    """Test __main__ module imports."""
    from moex_agent.__main__ import main
    assert main is not None


# ─────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────

def test_full_pipeline():
    """Test full signal generation pipeline (without network)."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect, get_window
    from moex_agent.features import build_feature_frame
    from moex_agent.anomaly import compute_anomalies
    from moex_agent.predictor import ModelRegistry, FEATURE_COLS
    from moex_agent.risk import RiskParams, pass_gatekeeper

    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    # Load data
    df = get_window(conn, minutes=60*24)
    assert len(df) > 0

    # Build features
    features = build_feature_frame(df)
    features = features.dropna()
    assert len(features) > 0

    # Detect anomalies
    quotes = {secid: {"secid": secid} for secid in df["secid"].unique()}
    anomalies = compute_anomalies(
        candles_1m=df[["secid", "ts", "close", "value", "volume"]],
        quotes=quotes,
        min_turnover_rub_5m=cfg.risk.min_turnover_rub_5m,
        max_spread_bps=cfg.risk.max_spread_bps,
        top_n=cfg.top_n_anomalies,
    )

    # Load models
    registry = ModelRegistry()
    registry.load()

    # Generate predictions for anomalies
    latest = features.sort_values(["secid", "ts"]).groupby("secid").tail(1)

    for anomaly in anomalies[:3]:
        row = latest[latest["secid"] == anomaly.secid]
        if row.empty:
            continue

        X = row[FEATURE_COLS].to_numpy(dtype=float)
        best_h, best_p = registry.best_horizon(X)

        assert best_h in registry.horizons
        assert 0 <= best_p <= 1

    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

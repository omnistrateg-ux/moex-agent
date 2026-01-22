"""
MOEX Agent Pipeline Engine

Encapsulates the core signal generation pipeline:
1. Fetch candles and quotes (parallel)
2. Detect anomalies
3. Build features
4. Predict probabilities
5. Apply risk gatekeeper
6. Return signal candidates

Designed for reuse in CLI, web API, and testing.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .anomaly import AnomalyResult, Direction, compute_anomalies
from .config_schema import AppConfig
from .features import build_feature_frame
from .moex_iss import fetch_candles, fetch_quote
from .predictor import FEATURE_COLS, ModelRegistry, safe_predict_proba
from .risk import RiskParams, pass_gatekeeper
from .storage import get_window, upsert_many

logger = logging.getLogger("moex_agent.engine")


@dataclass
class Signal:
    """Generated trading signal."""

    secid: str
    direction: Direction
    horizon: str
    probability: float
    signal_type: str  # 'price-exit' or 'time-exit'
    entry: Optional[float] = None
    take: Optional[float] = None
    stop: Optional[float] = None
    ttl_minutes: Optional[int] = None

    # Anomaly data
    anomaly_score: float = 0.0
    z_ret_5m: float = 0.0
    z_vol_5m: float = 0.0
    ret_5m: float = 0.0
    turnover_5m: float = 0.0
    spread_bps: Optional[float] = None
    volume_spike: float = 1.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.secid,
            "direction": self.direction.value if isinstance(self.direction, Direction) else self.direction,
            "horizon": self.horizon,
            "p": round(self.probability, 4),
            "signal_type": self.signal_type,
            "entry": self.entry,
            "take": self.take,
            "stop": self.stop,
            "ttl_minutes": self.ttl_minutes,
            "anomaly": {
                "score": round(self.anomaly_score, 3),
                "z_ret_5m": round(self.z_ret_5m, 3),
                "z_vol_5m": round(self.z_vol_5m, 3),
                "ret_5m": round(self.ret_5m, 5),
                "turnover_5m": int(self.turnover_5m),
                "spread_bps": None if self.spread_bps is None else round(self.spread_bps, 1),
                "volume_spike": round(self.volume_spike, 2),
            },
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CycleResult:
    """Result of a single pipeline cycle."""

    signals: List[Signal]
    anomalies_count: int
    candles_fetched: int
    quotes_fetched: int
    duration_ms: float
    errors: List[str] = field(default_factory=list)


class PipelineEngine:
    """
    Core signal generation engine.

    Thread-safe, stateless per cycle. Maintains model registry
    and provides methods for parallel data fetching.
    """

    def __init__(
        self,
        config: AppConfig,
        models_dir: Path = Path("./models"),
    ):
        self.config = config
        self.models = ModelRegistry(models_dir)
        self.risk_params = RiskParams(
            max_spread_bps=config.risk.max_spread_bps,
            min_turnover_rub_5m=config.risk.min_turnover_rub_5m,
        )

    def load_models(self) -> None:
        """Pre-load ML models."""
        self.models.load()

    def fetch_candles_parallel(
        self,
        from_date: str,
        till_date: str,
    ) -> Dict[str, List]:
        """
        Fetch candles for all tickers in parallel.

        Args:
            from_date: Start date (YYYY-MM-DD)
            till_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping secid to list of Candle objects
        """
        results = {}

        def fetch_one(secid: str):
            try:
                candles = fetch_candles(
                    self.config.engine,
                    self.config.market,
                    self.config.board,
                    secid,
                    interval=1,
                    from_date=from_date,
                    till_date=till_date,
                )
                return secid, candles, None
            except Exception as e:
                return secid, [], e

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(fetch_one, s): s for s in self.config.tickers}
            for future in as_completed(futures):
                secid, candles, err = future.result()
                if err:
                    logger.warning(f"Failed to fetch candles for {secid}: {err}")
                results[secid] = candles

        return results

    def fetch_quotes_parallel(self) -> Dict[str, Dict]:
        """
        Fetch quotes for all tickers in parallel.

        Returns:
            Dict mapping secid to quote dict
        """
        results = {}

        def fetch_one(secid: str):
            try:
                q = fetch_quote(
                    self.config.engine,
                    self.config.market,
                    self.config.board,
                    secid,
                )
                return secid, q, None
            except Exception as e:
                return secid, {"secid": secid}, e

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(fetch_one, s): s for s in self.config.tickers}
            for future in as_completed(futures):
                secid, quote, err = future.result()
                if err:
                    logger.warning(f"Failed to fetch quote for {secid}: {err}")
                results[secid] = quote

        return results

    def detect_anomalies(
        self,
        candles_df: pd.DataFrame,
        quotes: Dict[str, Dict],
    ) -> List[AnomalyResult]:
        """
        Detect price/volume anomalies.

        Args:
            candles_df: DataFrame with candle data
            quotes: Dict of quotes

        Returns:
            List of AnomalyResult sorted by score
        """
        return compute_anomalies(
            candles_1m=candles_df[["secid", "ts", "close", "value", "volume"]],
            quotes=quotes,
            min_turnover_rub_5m=self.risk_params.min_turnover_rub_5m,
            max_spread_bps=self.risk_params.max_spread_bps,
            top_n=self.config.top_n_anomalies,
        )

    def generate_signals(
        self,
        anomalies: List[AnomalyResult],
        features_df: pd.DataFrame,
        quotes: Dict[str, Dict],
        cooldown_map: Optional[Dict[str, datetime]] = None,
    ) -> List[Signal]:
        """
        Generate signals from anomalies.

        Args:
            anomalies: List of detected anomalies
            features_df: DataFrame with computed features
            quotes: Dict of quotes
            cooldown_map: Optional dict of secid -> last alert time

        Returns:
            List of generated Signal objects
        """
        self.models.ensure_loaded()

        if cooldown_map is None:
            cooldown_map = {}

        signals = []
        now = datetime.now(timezone.utc)
        cooldown_td = timedelta(minutes=self.config.cooldown_minutes)

        # Get latest feature row per secid
        latest = features_df.sort_values(["secid", "ts"]).groupby("secid").tail(1)

        for anomaly in anomalies:
            secid = anomaly.secid

            # Check cooldown
            last_alert = cooldown_map.get(secid, datetime.min.replace(tzinfo=timezone.utc))
            if (now - last_alert) < cooldown_td:
                continue

            # Get feature row
            row = latest[latest["secid"] == secid]
            if row.empty:
                continue

            # Prepare feature vector
            try:
                X = row[FEATURE_COLS].to_numpy(dtype=float)
            except KeyError as e:
                logger.warning(f"Missing feature column for {secid}: {e}")
                continue

            # Find best horizon
            best_h, best_p = self.models.best_horizon(X)
            if best_h is None:
                continue

            # Risk gatekeeper
            if not pass_gatekeeper(
                p=best_p,
                p_threshold=self.config.p_threshold,
                turnover_5m=anomaly.turnover_5m,
                spread=anomaly.spread_bps,
                risk=self.risk_params,
            ):
                continue

            # Compute price targets
            last_price = quotes.get(secid, {}).get("last")
            atr = float(row["atr_14"].iloc[0]) if "atr_14" in row.columns else None

            entry = float(last_price) if last_price else None
            take = None
            stop = None
            signal_type = "time-exit"

            price_exit_cfg = self.config.signals.price_exit
            if price_exit_cfg.enabled and last_price and atr and atr > 0:
                take_atr = price_exit_cfg.take_atr
                stop_atr = price_exit_cfg.stop_atr

                if anomaly.direction == Direction.LONG:
                    take = float(last_price + take_atr * atr)
                    stop = float(last_price - stop_atr * atr)
                else:  # SHORT
                    take = float(last_price - take_atr * atr)
                    stop = float(last_price + stop_atr * atr)

                signal_type = "price-exit"

            # Get TTL from horizon config
            ttl = next(
                (h.minutes for h in self.config.horizons if h.name == best_h),
                60,
            )

            # Create signal
            signal = Signal(
                secid=secid,
                direction=anomaly.direction,
                horizon=best_h,
                probability=best_p,
                signal_type=signal_type,
                entry=entry,
                take=take,
                stop=stop,
                ttl_minutes=ttl,
                anomaly_score=anomaly.score,
                z_ret_5m=anomaly.z_ret_5m,
                z_vol_5m=anomaly.z_vol_5m,
                ret_5m=anomaly.ret_5m,
                turnover_5m=anomaly.turnover_5m,
                spread_bps=anomaly.spread_bps,
                volume_spike=anomaly.volume_spike,
            )

            signals.append(signal)

        return signals

    def run_cycle(
        self,
        conn,
        candles_df: Optional[pd.DataFrame] = None,
        quotes: Optional[Dict[str, Dict]] = None,
        cooldown_map: Optional[Dict[str, datetime]] = None,
    ) -> CycleResult:
        """
        Run a single pipeline cycle.

        This is the main entry point for signal generation.
        Can accept pre-fetched data for testing.

        Args:
            conn: SQLite connection for upserts
            candles_df: Optional pre-loaded candles DataFrame
            quotes: Optional pre-loaded quotes
            cooldown_map: Optional cooldown state

        Returns:
            CycleResult with signals and metrics
        """
        import time

        start = time.perf_counter()
        errors = []

        # 1. Fetch data if not provided
        today = datetime.now(timezone.utc).date()
        from_date = (today - timedelta(days=3)).isoformat()
        till_date = today.isoformat()

        candles_fetched = 0
        quotes_fetched = 0

        if candles_df is None:
            all_candles = self.fetch_candles_parallel(from_date, till_date)

            # Upsert to database
            for secid, candles in all_candles.items():
                if candles:
                    candles_fetched += len(candles)
                    rows = [
                        (secid, self.config.board, 1, c.ts, c.open, c.high, c.low, c.close, c.value, c.volume)
                        for c in candles
                    ]
                    try:
                        upsert_many(
                            conn,
                            "candles",
                            ("secid", "board", "interval", "ts", "open", "high", "low", "close", "value", "volume"),
                            rows,
                        )
                    except Exception as e:
                        errors.append(f"Upsert candles {secid}: {e}")

            # Load from DB using anchor-based window
            candles_df = get_window(conn, minutes=3 * 24 * 60, interval=1)

        if quotes is None:
            quotes = self.fetch_quotes_parallel()
            quotes_fetched = len(quotes)

            # Upsert quotes
            now_ts = datetime.now(timezone.utc).isoformat()
            qrows = [
                (secid, self.config.board, now_ts, q.get("last"), q.get("bid"), q.get("ask"),
                 q.get("numtrades"), q.get("voltoday"), q.get("valtoday"))
                for secid, q in quotes.items()
            ]
            try:
                upsert_many(
                    conn,
                    "quotes",
                    ("secid", "board", "ts", "last", "bid", "ask", "numtrades", "voltoday", "valtoday"),
                    qrows,
                )
            except Exception as e:
                errors.append(f"Upsert quotes: {e}")

        # 2. Detect anomalies
        anomalies = self.detect_anomalies(candles_df, quotes)

        # 3. Build features
        features_df = build_feature_frame(candles_df)
        features_df = features_df.dropna()

        # 4. Generate signals
        signals = self.generate_signals(
            anomalies=anomalies,
            features_df=features_df,
            quotes=quotes,
            cooldown_map=cooldown_map,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        return CycleResult(
            signals=signals,
            anomalies_count=len(anomalies),
            candles_fetched=candles_fetched,
            quotes_fetched=quotes_fetched,
            duration_ms=duration_ms,
            errors=errors,
        )


def create_engine(config_path: str = "config.yaml") -> PipelineEngine:
    """
    Factory function to create a pipeline engine.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configured PipelineEngine instance
    """
    from .config_schema import load_config

    config = load_config(config_path)
    engine = PipelineEngine(config)
    engine.load_models()
    return engine

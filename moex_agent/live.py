from __future__ import annotations

import json
import logging
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .anomaly import compute_anomalies
from .config import load_config
from .features import build_feature_frame
from .moex_iss import fetch_candles, fetch_quote
from .qwen import analyze_signal, format_telegram_message, QwenAnalysis
from .risk import RiskParams, pass_gatekeeper
from .storage import connect, upsert_many
from .telegram import send_telegram, send_signal_alert

# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("moex_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("moex_agent")

# ─────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown signal received, finishing current cycle...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def p_pos1(clf, X) -> float:
    """Return P(class==1) for single-row X, robust to classes_ order and (n,1) proba."""
    import numpy as np
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))

    if proba is None:
        return 0.0

    proba = np.asarray(proba)

    if getattr(proba, "ndim", 0) == 1:
        return float(proba[0])

    if proba.ndim != 2:
        return float(proba.ravel()[0])

    if proba.shape[1] == 1:
        return float(proba[0, 0])

    if 1 in classes:
        return float(proba[0, classes.index(1)])
    if True in classes:
        return float(proba[0, classes.index(True)])

    return float(proba[0, 1])



def _load_models(models_dir: Path) -> Dict[str, Any]:
    meta = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
    out = {}
    for h, info in meta.items():
        out[h] = joblib.load(info["path"])
    return out


def _load_recent_1m(conn) -> pd.DataFrame:
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval=1
    AND ts >= datetime('now','-3 days')
    ORDER BY secid, ts
    """
    return pd.read_sql_query(q, conn)


# ─────────────────────────────────────────────────────────────
# Parallel data fetching
# ─────────────────────────────────────────────────────────────
def _fetch_candles_parallel(
    cfg, from_date: str, till_date: str, max_workers: int = 10
) -> Dict[str, List]:
    """Fetch candles for all tickers in parallel. Returns {secid: [Candle, ...]}"""
    results = {}

    def fetch_one(secid: str):
        try:
            candles = fetch_candles(
                cfg.engine, cfg.market, cfg.board, secid,
                interval=1, from_date=from_date, till_date=till_date
            )
            return secid, candles, None
        except Exception as e:
            return secid, [], e

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, s): s for s in cfg.tickers}
        for future in as_completed(futures):
            secid, candles, err = future.result()
            if err:
                logger.warning(f"Failed to fetch candles for {secid}: {err}")
            results[secid] = candles

    return results


def _fetch_quotes_parallel(cfg, max_workers: int = 10) -> Dict[str, Dict]:
    """Fetch quotes for all tickers in parallel. Returns {secid: quote_dict}"""
    results = {}

    def fetch_one(secid: str):
        try:
            q = fetch_quote(cfg.engine, cfg.market, cfg.board, secid)
            return secid, q, None
        except Exception as e:
            return secid, {"secid": secid}, e

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, s): s for s in cfg.tickers}
        for future in as_completed(futures):
            secid, quote, err = future.result()
            if err:
                logger.warning(f"Failed to fetch quote for {secid}: {err}")
            results[secid] = quote

    return results


# ─────────────────────────────────────────────────────────────
# In-memory candles cache (avoid reloading 50K rows each cycle)
# ─────────────────────────────────────────────────────────────
class CandlesCache:
    """Incremental cache for recent candles. Avoids full DB reload each cycle."""

    def __init__(self, lookback_days: int = 3):
        self.lookback_days = lookback_days
        self._df: Optional[pd.DataFrame] = None
        self._last_ts: Dict[str, datetime] = {}  # secid -> last known ts

    def update(self, new_candles: Dict[str, List], conn) -> pd.DataFrame:
        """Merge new candles into cache and return full DataFrame."""
        # First call: load from DB
        if self._df is None:
            self._df = self._load_from_db(conn)
            for secid in self._df["secid"].unique():
                grp = self._df[self._df["secid"] == secid]
                if not grp.empty:
                    self._last_ts[secid] = grp["ts"].max()

        # Append new candles
        new_rows = []
        for secid, candles in new_candles.items():
            last_known = self._last_ts.get(secid, datetime.min.replace(tzinfo=timezone.utc))
            for c in candles:
                ts = pd.to_datetime(c.ts, utc=True)
                if ts > last_known:
                    new_rows.append({
                        "secid": secid,
                        "ts": ts,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "value": c.value,
                        "volume": c.volume,
                    })
            if candles:
                max_ts = max(pd.to_datetime(c.ts, utc=True) for c in candles)
                if max_ts > last_known:
                    self._last_ts[secid] = max_ts

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self._df = pd.concat([self._df, new_df], ignore_index=True)

        # Trim old data (keep only lookback_days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        self._df = self._df[self._df["ts"] >= cutoff].copy()

        return self._df

    def _load_from_db(self, conn) -> pd.DataFrame:
        q = """
        SELECT secid, ts, open, high, low, close, value, volume
        FROM candles
        WHERE interval=1 AND ts >= datetime('now','-3 days')
        ORDER BY secid, ts
        """
        df = pd.read_sql_query(q, conn)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df


# ─────────────────────────────────────────────────────────────
# Metrics tracking
# ─────────────────────────────────────────────────────────────
class Metrics:
    """Simple metrics collector for monitoring."""
    def __init__(self):
        self.cycles = 0
        self.alerts_sent = 0
        self.errors = 0
        self.last_cycle_ms = 0
        self.avg_cycle_ms = 0.0
        self.start_time = datetime.now(timezone.utc)

    def record_cycle(self, duration_ms: float):
        self.cycles += 1
        self.last_cycle_ms = duration_ms
        # Exponential moving average
        alpha = 0.1
        self.avg_cycle_ms = alpha * duration_ms + (1 - alpha) * self.avg_cycle_ms

    def to_dict(self) -> Dict[str, Any]:
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "uptime_seconds": int(uptime),
            "cycles": self.cycles,
            "alerts_sent": self.alerts_sent,
            "errors": self.errors,
            "last_cycle_ms": round(self.last_cycle_ms, 1),
            "avg_cycle_ms": round(self.avg_cycle_ms, 1),
        }


def main() -> None:
    global _shutdown_requested

    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    models_dir = Path("./models")
    models = _load_models(models_dir)

    last_alert_ts: Dict[str, datetime] = defaultdict(lambda: datetime(1970,1,1,tzinfo=timezone.utc))

    risk_params = RiskParams(
        max_spread_bps=float(cfg.risk.max_spread_bps),
        min_turnover_rub_5m=float(cfg.risk.min_turnover_rub_5m),
    )

    metrics = Metrics()
    candles_cache = CandlesCache(lookback_days=3)

    logger.info("Live loop started")
    logger.info(f"Config: poll={cfg.poll_seconds}s | tickers={len(cfg.tickers)} | p_threshold={cfg.p_threshold}")

    while not _shutdown_requested:

        p_map = {}  # secid -> p
        cycle_start = time.perf_counter()

        try:
            # 1) Refresh latest candles (parallel fetch)
            today = datetime.now(timezone.utc).date()
            from_date = (today - timedelta(days=3)).isoformat()
            till_date = today.isoformat()

            # PARALLEL: fetch all candles at once
            all_candles = _fetch_candles_parallel(cfg, from_date, till_date, max_workers=cfg.max_workers)
            for secid, candles in all_candles.items():
                if candles:
                    rows = [(secid, cfg.board, 1, c.ts, c.open, c.high, c.low, c.close, c.value, c.volume) for c in candles]
                    upsert_many(conn, "candles", ("secid","board","interval","ts","open","high","low","close","value","volume"), rows)

            # 2) Fetch quotes (parallel)
            now_ts = datetime.now(timezone.utc).isoformat()
            quotes = _fetch_quotes_parallel(cfg, max_workers=cfg.max_workers)
            qrows = [
                (secid, cfg.board, now_ts, q.get("last"), q.get("bid"), q.get("ask"),
                 q.get("numtrades"), q.get("voltoday"), q.get("valtoday"))
                for secid, q in quotes.items()
            ]
            upsert_many(conn, "quotes", ("secid","board","ts","last","bid","ask","numtrades","voltoday","valtoday"), qrows)

            # 3) Use cached candles (incremental update, not full DB reload)
            candles_1m = candles_cache.update(all_candles, conn)
            anomalies = compute_anomalies(
                candles_1m=candles_1m[["secid","ts","close","value","volume"]],
                quotes=quotes,
                min_turnover_rub_5m=risk_params.min_turnover_rub_5m,
                max_spread_bps=risk_params.max_spread_bps,
                top_n=cfg.top_n_anomalies,
            )
            if anomalies:
                logger.debug(f"TOP_ANOMALIES: {[(a.secid, round(a.score, 2)) for a in anomalies]}")
            else:
                # Нет аномалий — рынок спокойный, это нормально
                cycle_ms = (time.perf_counter() - cycle_start) * 1000
                metrics.record_cycle(cycle_ms)

                # Логируем каждые 12 циклов (~1 мин) что нет аномалий
                if metrics.cycles % 12 == 0:
                    logger.info(f"No anomalies detected (cycle {metrics.cycles})")

                if __import__("os").getenv("MOEX_AGENT_ONCE") == "1":
                    logger.info("ONCE_MODE: exiting after one cycle (no anomalies)")
                    break
                time.sleep(cfg.poll_seconds)
                continue

            # 4) build features for last row per secid
            feats = build_feature_frame(candles_1m)
            feats = feats.dropna()
            latest = feats.sort_values(["secid","ts"]).groupby("secid").tail(1)

            # 5) for each anomaly: compute p(H) and choose best horizon
            for a in anomalies:
                secid = a.secid
                if (datetime.now(timezone.utc) - last_alert_ts[secid]) < timedelta(minutes=cfg.cooldown_minutes):
                    continue

                row = latest[latest["secid"] == secid]
                if row.empty:
                    continue

                X = row[[
                    "r_1m","r_5m","r_10m","r_30m","r_60m",
                    "turn_1m","turn_5m","turn_10m",
                    "atr_14","dist_vwap_atr",
                ]].to_numpy(dtype=float)

                best = (None, 0.0)
                for hname, clf in models.items():
                    p = p_pos1(clf, X)
                    p_map[secid] = float(p)
                    if not globals().get('_CLF_CLASSES_ONCE'):
                        globals()['_CLF_CLASSES_ONCE'] = True
                        logger.debug(f"CLF_CLASSES: {getattr(clf, 'classes_', None)}")
                    if p > best[1]:
                        best = (hname, p)

                hname, pbest = best
                if hname is None:
                    continue

                spread = a.spread_bps
                if not pass_gatekeeper(
                    p=pbest,
                    p_threshold=cfg.p_threshold,
                    turnover_5m=a.turnover_5m,
                    spread=spread,
                    risk=risk_params,
                ):
                    continue

                last_price = quotes.get(secid, {}).get("last")
                atr = float(row["atr_14"].iloc[0]) if "atr_14" in row else None

                entry = float(last_price) if last_price else float(row["dist_vwap_atr"].iloc[0])
                take = None
                stop = None
                ttl = next((h.minutes for h in cfg.horizons if h.name == hname), 60)

                if cfg.price_exit.get("enabled", True) and last_price and atr and atr > 0:
                    take = float(last_price + cfg.price_exit.get("take_atr", 0.8) * atr)
                    stop = float(last_price - cfg.price_exit.get("stop_atr", 0.6) * atr)

                # Direction из аномалии
                direction = getattr(a, 'direction', 'LONG')
                if hasattr(direction, 'value'):
                    direction = direction.value  # Enum -> str

                volume_spike = getattr(a, 'volume_spike', 1.0)

                payload = {
                    "ticker": secid,
                    "direction": direction,
                    "horizon": hname,
                    "p": round(pbest, 4),
                    "signal_type": "price-exit" if take and stop else "time-exit",
                    "entry": float(last_price) if last_price else None,
                    "take": take,
                    "stop": stop,
                    "ttl_minutes": ttl,
                    "anomaly": {
                        "score": round(a.score, 3),
                        "z_ret_5m": round(a.z_ret_5m, 3),
                        "z_vol_5m": round(a.z_vol_5m, 3),
                        "ret_5m": round(a.ret_5m, 5),
                        "turnover_5m": int(a.turnover_5m),
                        "spread_bps": None if spread is None else round(spread, 1),
                        "volume_spike": round(volume_spike, 2),
                    },
                }

                # ─────────────────────────────────────────────────────
                # Qwen Analysis (с правилами + LLM)
                # ─────────────────────────────────────────────────────
                analysis: QwenAnalysis = None
                if cfg.qwen.enabled:
                    try:
                        analysis = analyze_signal(
                            ollama_url=cfg.qwen.ollama_url,
                            model=cfg.qwen.model,
                            payload=payload,
                            max_tokens=cfg.qwen.max_tokens,
                            temperature=cfg.qwen.temperature,
                        )
                        if analysis.skip:
                            logger.debug(f"Signal {secid} skipped by Qwen: {analysis.skip_reason}")
                            continue
                    except Exception as e:
                        logger.warning(f"Qwen analysis failed for {secid}: {e}")
                        analysis = None

                # Форматирование сообщения
                dir_emoji = "📈" if direction == "LONG" else "📉"

                if analysis:
                    text = format_telegram_message(
                        ticker=secid,
                        horizon=hname,
                        p=pbest,
                        analysis=analysis,
                        direction=direction,
                        entry=payload.get("entry"),
                        take=payload.get("take"),
                        stop=payload.get("stop"),
                        anomaly_score=a.score,
                        volume_spike=volume_spike,
                    )
                else:
                    # Fallback без Qwen
                    if payload["signal_type"] == "price-exit":
                        text = (
                            f"🟢 {secid} {dir_emoji} {direction} | {hname}\n"
                            f"📊 p={pbest:.0%} | score={a.score:.1f} | vol={volume_spike:.1f}x\n"
                            f"💰 Entry: {payload['entry']:.2f} → Take: {payload['take']:.2f} | Stop: {payload['stop']:.2f}\n"
                            f"z_ret={a.z_ret_5m:.1f} z_vol={a.z_vol_5m:.1f}"
                        )
                    else:
                        text = (
                            f"🟢 {secid} {dir_emoji} {direction} | {hname}\n"
                            f"📊 p={pbest:.0%} | score={a.score:.1f} | vol={volume_spike:.1f}x\n"
                            f"⏱ Time-exit: close after {ttl} min"
                        )

                # Отправка в Telegram (с фильтром по рекомендациям)
                if cfg.telegram_enabled:
                    recommendation = analysis.recommendation if analysis else "BUY"
                    allowed = cfg.telegram_send_recommendations

                    if recommendation in allowed:
                        sent = send_signal_alert(
                            bot_token=cfg.telegram_bot_token,
                            chat_id=cfg.telegram_chat_id,
                            ticker=secid,
                            direction=direction,
                            horizon=hname,
                            p=pbest,
                            score=a.score,
                            recommendation=recommendation,
                            risk_level=analysis.risk_level if analysis else "MEDIUM",
                            reasoning=analysis.reasoning if analysis else "",
                            entry=payload.get("entry"),
                            take=payload.get("take"),
                            stop=payload.get("stop"),
                            volume_spike=volume_spike,
                            risk_note=analysis.risk_note if analysis else "",
                        )
                        if sent:
                            logger.info(f"Telegram sent: {secid} {direction} {recommendation}")
                    else:
                        logger.debug(f"Telegram skipped: {recommendation} not in {allowed}")

                last_alert_ts[secid] = datetime.now(timezone.utc)
                metrics.alerts_sent += 1
                logger.info(f"ALERT: {text.replace(chr(10), ' | ')}")

            # Record cycle metrics
            cycle_ms = (time.perf_counter() - cycle_start) * 1000
            metrics.record_cycle(cycle_ms)

            # Periodic heartbeat (every 60 cycles ~= 5 min at 5s poll)
            if metrics.cycles % 60 == 0:
                m = metrics.to_dict()
                logger.info(f"HEARTBEAT cycles={m['cycles']} alerts={m['alerts_sent']} avg_cycle={m['avg_cycle_ms']:.0f}ms uptime={m['uptime_seconds']}s")

            # ONCE mode: exit after first complete cycle
            if __import__("os").getenv("MOEX_AGENT_ONCE") == "1":
                logger.info(f"ONCE_MODE: completed cycle in {cycle_ms:.0f}ms, exiting")
                break

            time.sleep(cfg.poll_seconds)

        except KeyboardInterrupt:
            break
        except Exception as e:
            metrics.errors += 1
            logger.error(f"Cycle error: {repr(e)}")
            time.sleep(max(5, cfg.poll_seconds))

    # Graceful shutdown
    conn.close()
    logger.info(f"Shutdown complete. Final metrics: {metrics.to_dict()}")


# AUTO_ENTRYPOINT_FOR_MOEX_AGENT
if __name__ == "__main__":
    main()

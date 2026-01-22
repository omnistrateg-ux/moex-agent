"""
MOEX Agent Storage Layer

SQLite database operations with WAL mode for performance.
Provides context manager for graceful connection handling.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("moex_agent.storage")


def connect(db_path: Path, optimize: bool = True) -> sqlite3.Connection:
    """
    Connect to SQLite with performance optimizations.

    Optimizations:
    - WAL mode: allows concurrent reads during writes
    - synchronous=NORMAL: 10x faster writes, still safe for WAL
    - cache_size: 64MB in-memory cache
    - mmap_size: memory-mapped I/O for faster reads
    - temp_store=MEMORY: temp tables in RAM

    Args:
        db_path: Path to SQLite database file
        optimize: Whether to apply performance optimizations

    Returns:
        SQLite connection object
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    if optimize:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-65536")  # 64MB (negative = KB)
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")  # 5s wait on lock

    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def database(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.

    Ensures proper cleanup on exit.

    Usage:
        with database(Path("data/moex.sqlite")) as conn:
            # use conn
    """
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
        logger.debug("Database connection closed")


def init_db(conn: sqlite3.Connection, schema_path: Path) -> None:
    """
    Initialize database schema from SQL file.

    Args:
        conn: SQLite connection
        schema_path: Path to schema.sql file
    """
    sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()
    logger.info(f"Database initialized from {schema_path}")


def upsert_many(
    conn: sqlite3.Connection,
    table: str,
    columns: Tuple[str, ...],
    rows: Iterable[Tuple[Any, ...]],
) -> int:
    """
    Insert or replace multiple rows.

    Args:
        conn: SQLite connection
        table: Table name
        columns: Column names tuple
        rows: Iterable of row tuples

    Returns:
        Number of affected rows
    """
    rows_list = list(rows)
    if not rows_list:
        return 0

    cols = ",".join(columns)
    qmarks = ",".join(["?"] * len(columns))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({qmarks})"
    cur = conn.executemany(sql, rows_list)
    conn.commit()
    return cur.rowcount


def get_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Get value from state table."""
    cur = conn.execute("SELECT value FROM state WHERE key=?", (key,))
    row = cur.fetchone()
    return None if row is None else str(row["value"])


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set value in state table."""
    conn.execute("INSERT OR REPLACE INTO state(key,value) VALUES (?,?)", (key, value))
    conn.commit()


def get_max_ts(conn: sqlite3.Connection, table: str = "candles") -> Optional[str]:
    """
    Get maximum timestamp from a table.

    Useful as anchor for weekend/holiday handling.

    Args:
        conn: SQLite connection
        table: Table name (default: candles)

    Returns:
        ISO timestamp string or None if table is empty
    """
    cur = conn.execute(f"SELECT MAX(ts) as max_ts FROM {table}")
    row = cur.fetchone()
    return row["max_ts"] if row and row["max_ts"] else None


def get_window(
    conn: sqlite3.Connection,
    minutes: int,
    anchor_ts: Optional[str] = None,
    interval: int = 1,
) -> pd.DataFrame:
    """
    Get candles window ending at anchor timestamp.

    If anchor_ts is None, uses MAX(ts) from candles table.
    This fixes the weekend problem where datetime('now','-3 days')
    returns empty results.

    Args:
        conn: SQLite connection
        minutes: Window size in minutes
        anchor_ts: End timestamp (ISO format) or None for latest
        interval: Candle interval (default: 1 = 1min)

    Returns:
        DataFrame with columns: secid, ts, open, high, low, close, value, volume
    """
    if anchor_ts is None:
        anchor_ts = get_max_ts(conn)

    if anchor_ts is None:
        # Empty database
        return pd.DataFrame(columns=["secid", "ts", "open", "high", "low", "close", "value", "volume"])

    # Use SQLite datetime functions for consistent format handling
    # DB stores timestamps as 'YYYY-MM-DD HH:MM:SS' (space-separated)
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = ?
      AND ts >= datetime(?, ?)
      AND ts <= ?
    ORDER BY secid, ts
    """
    # SQLite datetime modifier format: '-N minutes'
    modifier = f"-{minutes} minutes"
    df = pd.read_sql_query(q, conn, params=(interval, anchor_ts, modifier, anchor_ts))
    return df


def get_recent_candles(
    conn: sqlite3.Connection,
    days: int = 3,
    interval: int = 1,
) -> pd.DataFrame:
    """
    Get candles for the last N days (anchored to MAX(ts)).

    Args:
        conn: SQLite connection
        days: Number of days to fetch
        interval: Candle interval

    Returns:
        DataFrame with candles
    """
    return get_window(conn, minutes=days * 24 * 60, interval=interval)


def save_alert(
    conn: sqlite3.Connection,
    secid: str,
    horizon: str,
    p: float,
    signal_type: str,
    entry: Optional[float] = None,
    take: Optional[float] = None,
    stop: Optional[float] = None,
    ttl_minutes: Optional[int] = None,
    anomaly_score: Optional[float] = None,
    payload_json: Optional[str] = None,
) -> int:
    """
    Save alert to database.

    Args:
        conn: SQLite connection
        secid: Ticker symbol
        horizon: Signal horizon (e.g., '5m', '1h')
        p: Prediction probability
        signal_type: 'price-exit' or 'time-exit'
        entry: Entry price
        take: Take profit price
        stop: Stop loss price
        ttl_minutes: Time to live in minutes
        anomaly_score: Anomaly detection score
        payload_json: Full payload as JSON string

    Returns:
        Alert ID
    """
    created_ts = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO alerts (created_ts, secid, horizon, p, signal_type, entry, take, stop, ttl_minutes, anomaly_score, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (created_ts, secid, horizon, p, signal_type, entry, take, stop, ttl_minutes, anomaly_score, payload_json),
    )
    conn.commit()
    return cur.lastrowid or 0


def mark_alert_sent(conn: sqlite3.Connection, alert_id: int) -> None:
    """Mark an alert as sent (to Telegram)."""
    conn.execute("UPDATE alerts SET sent = 1 WHERE id = ?", (alert_id,))
    conn.commit()


def get_alerts(
    conn: sqlite3.Connection,
    limit: int = 100,
    sent_only: bool = False,
) -> List[sqlite3.Row]:
    """
    Get recent alerts from database.

    Args:
        conn: SQLite connection
        limit: Maximum number of alerts
        sent_only: If True, only return sent alerts

    Returns:
        List of alert rows
    """
    where = "WHERE sent = 1" if sent_only else ""
    q = f"""
    SELECT * FROM alerts
    {where}
    ORDER BY created_ts DESC
    LIMIT ?
    """
    cur = conn.execute(q, (limit,))
    return cur.fetchall()

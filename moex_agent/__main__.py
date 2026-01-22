"""
MOEX Agent CLI

Unified command-line interface for all operations.

Usage:
    python -m moex_agent init-db              # Initialize database
    python -m moex_agent bootstrap --days 180 # Load historical data
    python -m moex_agent train                # Train ML models
    python -m moex_agent live                 # Run live signal loop
    python -m moex_agent web --port 8000      # Start web dashboard
    python -m moex_agent telegram-test "msg"  # Test Telegram integration
    python -m moex_agent status               # Show system status
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moex_agent")


def cmd_init_db(args: argparse.Namespace) -> int:
    """Initialize database schema."""
    from .config_schema import load_config
    from .storage import connect, init_db

    config = load_config(args.config)
    conn = connect(config.sqlite_path)

    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return 1

    init_db(conn, schema_path)
    conn.close()

    logger.info(f"Database initialized: {config.sqlite_path}")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Load historical candle data."""
    from .config_schema import load_config
    from .moex_iss import fetch_candles
    from .storage import connect, upsert_many

    config = load_config(args.config)
    conn = connect(config.sqlite_path)

    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=args.days)).isoformat()
    till_date = today.isoformat()

    logger.info(f"Loading {args.days} days of data for {len(config.tickers)} tickers")
    logger.info(f"Period: {from_date} to {till_date}")

    total_candles = 0
    for i, secid in enumerate(config.tickers, 1):
        try:
            candles = fetch_candles(
                config.engine,
                config.market,
                config.board,
                secid,
                interval=1,
                from_date=from_date,
                till_date=till_date,
            )
            rows = [
                (secid, config.board, 1, c.ts, c.open, c.high, c.low, c.close, c.value, c.volume)
                for c in candles
            ]
            upsert_many(
                conn,
                table="candles",
                columns=("secid", "board", "interval", "ts", "open", "high", "low", "close", "value", "volume"),
                rows=rows,
            )
            total_candles += len(candles)
            logger.info(f"[{i}/{len(config.tickers)}] {secid}: {len(candles)} candles")
        except Exception as e:
            logger.warning(f"[{i}/{len(config.tickers)}] {secid}: ERROR - {e}")

    conn.close()
    logger.info(f"Bootstrap complete: {total_candles} candles loaded")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train ML models."""
    from .train import main as train_main

    logger.info("Starting model training...")
    train_main()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run live signal generation loop."""
    from .config_schema import load_config
    from .engine import PipelineEngine
    from .moex_iss import close_session
    from .qwen import analyze_signal, QwenAnalysis
    from .storage import connect, save_alert, mark_alert_sent
    from .telegram import send_signal_alert

    # Graceful shutdown handling
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Shutdown signal received, finishing current cycle...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load config and create engine
    config = load_config(args.config)
    conn = connect(config.sqlite_path)
    engine = PipelineEngine(config)
    engine.load_models()

    # State
    cooldown_map = defaultdict(lambda: datetime(1970, 1, 1, tzinfo=timezone.utc))
    cycle_count = 0
    alerts_sent = 0

    logger.info("Live loop started")
    logger.info(f"Tickers: {len(config.tickers)} | Poll: {config.poll_seconds}s | P threshold: {config.p_threshold}")

    try:
        while not shutdown_requested:
            cycle_count += 1

            try:
                # Run pipeline cycle
                result = engine.run_cycle(conn, cooldown_map=cooldown_map)

                if result.errors:
                    for err in result.errors:
                        logger.warning(f"Cycle error: {err}")

                if not result.signals:
                    if cycle_count % 12 == 0:  # Log every ~1 min
                        logger.info(f"Cycle {cycle_count}: no signals (anomalies: {result.anomalies_count})")

                    if args.once:
                        logger.info("ONCE mode: exiting after first cycle")
                        break

                    time.sleep(config.poll_seconds)
                    continue

                # Process each signal
                for sig in result.signals:
                    # Optional: Qwen LLM analysis
                    analysis: Optional[QwenAnalysis] = None
                    if config.qwen.enabled:
                        try:
                            from .qwen import analyze_signal

                            analysis = analyze_signal(
                                ollama_url=config.qwen.ollama_url,
                                model=config.qwen.model,
                                payload=sig.to_dict(),
                                max_tokens=config.qwen.max_tokens,
                                temperature=config.qwen.temperature,
                            )
                            if analysis.skip:
                                logger.debug(f"Signal {sig.secid} skipped by Qwen: {analysis.skip_reason}")
                                continue
                        except Exception as e:
                            logger.warning(f"Qwen analysis failed for {sig.secid}: {e}")

                    # Save alert to database
                    alert_id = save_alert(
                        conn,
                        secid=sig.secid,
                        horizon=sig.horizon,
                        p=sig.probability,
                        signal_type=sig.signal_type,
                        entry=sig.entry,
                        take=sig.take,
                        stop=sig.stop,
                        ttl_minutes=sig.ttl_minutes,
                        anomaly_score=sig.anomaly_score,
                        payload_json=str(sig.to_dict()),
                    )

                    # Send Telegram notification
                    if config.telegram.enabled:
                        recommendation = analysis.recommendation if analysis else "BUY"
                        if recommendation in config.telegram.send_recommendations:
                            direction = sig.direction.value if hasattr(sig.direction, 'value') else sig.direction
                            sent = send_signal_alert(
                                bot_token=config.telegram.bot_token or "",
                                chat_id=config.telegram.chat_id or "",
                                ticker=sig.secid,
                                direction=direction,
                                horizon=sig.horizon,
                                p=sig.probability,
                                score=sig.anomaly_score,
                                recommendation=recommendation,
                                risk_level=analysis.risk_level if analysis else "MEDIUM",
                                reasoning=analysis.reasoning if analysis else "",
                                entry=sig.entry,
                                take=sig.take,
                                stop=sig.stop,
                                volume_spike=sig.volume_spike,
                                risk_note=analysis.risk_note if analysis else "",
                            )
                            if sent:
                                mark_alert_sent(conn, alert_id)
                                alerts_sent += 1
                                logger.info(f"Telegram sent: {sig.secid} {direction} {recommendation}")
                        else:
                            logger.debug(f"Telegram skipped: {recommendation} not in allowed list")

                    # Update cooldown
                    cooldown_map[sig.secid] = datetime.now(timezone.utc)

                    # Log signal
                    direction = sig.direction.value if hasattr(sig.direction, 'value') else sig.direction
                    logger.info(
                        f"SIGNAL: {sig.secid} {direction} {sig.horizon} "
                        f"p={sig.probability:.0%} score={sig.anomaly_score:.1f}"
                    )

                # Heartbeat
                if cycle_count % 60 == 0:
                    logger.info(f"HEARTBEAT: cycle={cycle_count} alerts={alerts_sent}")

                if args.once:
                    logger.info("ONCE mode: exiting after first cycle with signals")
                    break

                time.sleep(config.poll_seconds)

            except Exception as e:
                logger.error(f"Cycle error: {repr(e)}")
                time.sleep(max(5, config.poll_seconds))

    finally:
        # Graceful shutdown
        conn.close()
        close_session()
        logger.info(f"Shutdown complete. Cycles: {cycle_count}, Alerts: {alerts_sent}")

    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """Start FastAPI web dashboard."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        return 1

    logger.info(f"Starting web server on http://0.0.0.0:{args.port}")
    uvicorn.run(
        "moex_agent.webapp:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_telegram_test(args: argparse.Namespace) -> int:
    """Test Telegram integration."""
    from .config_schema import load_config
    from .telegram import send_telegram

    config = load_config(args.config)

    if not config.telegram.enabled:
        logger.error("Telegram is disabled in config")
        return 1

    if not config.telegram.bot_token or not config.telegram.chat_id:
        logger.error("Telegram bot_token or chat_id not configured")
        return 1

    message = args.message or f"Test message from MOEX Agent at {datetime.now().isoformat()}"

    success = send_telegram(
        bot_token=config.telegram.bot_token,
        chat_id=config.telegram.chat_id,
        text=message,
    )

    if success:
        logger.info("Telegram message sent successfully!")
        return 0
    else:
        logger.error("Failed to send Telegram message")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status."""
    from .config_schema import load_config
    from .storage import connect

    config = load_config(args.config)

    print(f"\n{'=' * 50}")
    print("MOEX Agent Status")
    print(f"{'=' * 50}")

    # Config
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Tickers: {len(config.tickers)}")
    print(f"  Poll interval: {config.poll_seconds}s")
    print(f"  P threshold: {config.p_threshold}")

    # Database
    print(f"\nDatabase:")
    print(f"  Path: {config.sqlite_path}")
    if config.sqlite_path.exists():
        conn = connect(config.sqlite_path)
        cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
        candles_count = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM alerts")
        alerts_count = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM candles")
        row = cur.fetchone()
        conn.close()
        print(f"  Candles: {candles_count:,}")
        print(f"  Alerts: {alerts_count:,}")
        print(f"  Date range: {row['min_ts']} to {row['max_ts']}")
    else:
        print("  Status: NOT INITIALIZED")

    # Models
    print(f"\nModels:")
    models_dir = Path("./models")
    meta_path = models_dir / "meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        print(f"  Loaded: {list(meta.keys())}")
    else:
        print("  Status: NOT TRAINED")

    # Telegram
    print(f"\nTelegram:")
    print(f"  Enabled: {config.telegram.enabled}")
    if config.telegram.enabled:
        print(f"  Bot: {'configured' if config.telegram.bot_token else 'NOT CONFIGURED'}")
        print(f"  Chat ID: {'configured' if config.telegram.chat_id else 'NOT CONFIGURED'}")

    # Qwen
    print(f"\nQwen LLM:")
    print(f"  Enabled: {config.qwen.enabled}")
    if config.qwen.enabled:
        print(f"  URL: {config.qwen.ollama_url}")
        print(f"  Model: {config.qwen.model}")

    print(f"\n{'=' * 50}\n")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="moex_agent",
        description="MOEX Trading Signal Agent",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-db
    sub = subparsers.add_parser("init-db", help="Initialize database schema")

    # bootstrap
    sub = subparsers.add_parser("bootstrap", help="Load historical candle data")
    sub.add_argument("--days", type=int, default=180, help="Days of history to load (default: 180)")

    # train
    sub = subparsers.add_parser("train", help="Train ML models")

    # live
    sub = subparsers.add_parser("live", help="Run live signal generation loop")
    sub.add_argument("--once", action="store_true", help="Run one cycle and exit")

    # web
    sub = subparsers.add_parser("web", help="Start FastAPI web dashboard")
    sub.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    sub.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # telegram-test
    sub = subparsers.add_parser("telegram-test", help="Test Telegram integration")
    sub.add_argument("message", nargs="?", help="Message to send")

    # status
    sub = subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Dispatch command
    if args.command == "init-db":
        return cmd_init_db(args)
    elif args.command == "bootstrap":
        return cmd_bootstrap(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "live":
        return cmd_live(args)
    elif args.command == "web":
        return cmd_web(args)
    elif args.command == "telegram-test":
        return cmd_telegram_test(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

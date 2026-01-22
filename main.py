"""
MOEX Agent — AI-Powered Trading Signal Generator
=================================================

Web dashboard + Paper trading for Moscow Exchange.

Features:
- ML models (Walk-Forward validated, 56% WR, PF>2.3)
- Real-time signal generation
- Margin risk management (Kill-Switch, Dynamic Leverage)
- Telegram notifications
- Web dashboard

Run on Replit:
    1. Set Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    2. Click Run

Author: MOEX Agent Team
"""
import os
import sys
import threading
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moex_agent")


def ensure_directories():
    """Create necessary directories."""
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


def init_database():
    """Initialize database if empty or missing."""
    from moex_agent.config_schema import load_config
    from moex_agent.storage import connect

    config = load_config()
    conn = connect(config.sqlite_path)

    # Check if we have data
    cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
    count = cur.fetchone()["cnt"]

    if count < 10000:
        logger.info(f"Database has only {count} candles, bootstrapping...")
        try:
            from moex_agent.bootstrap import bootstrap_recent
            bootstrap_recent(conn, config, days=7)
            logger.info("Bootstrap complete!")
        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}")
            logger.info("Will fetch data on first cycle")
    else:
        logger.info(f"Database ready: {count:,} candles")

    conn.close()


def run_trading_background():
    """Run margin paper trading in background thread."""
    time.sleep(10)  # Wait for web server to start

    try:
        from moex_agent.margin_paper_trading import MarginPaperTrader

        logger.info("Starting margin paper trading...")
        trader = MarginPaperTrader(
            initial_capital=200_000,
            max_leverage=3.0,
            max_positions=3,
            resume=True,
        )
        trader.run(duration_hours=168)  # 1 week
    except Exception as e:
        logger.error(f"Trading error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    logger.info("=" * 50)
    logger.info("MOEX Agent Starting...")
    logger.info("=" * 50)

    # Setup
    ensure_directories()

    # Check environment
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.environ.get("TELEGRAM_CHAT_ID")

    if telegram_token and telegram_chat:
        logger.info("Telegram: configured via environment")
    else:
        logger.warning("Telegram: not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")

    # Initialize database
    init_database()

    # Check models
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.joblib"))
    if model_files:
        logger.info(f"Models found: {[f.stem for f in model_files]}")
    else:
        logger.warning("No models found in models/ directory")

    # Start trading in background
    trading_thread = threading.Thread(
        target=run_trading_background,
        daemon=True,
        name="TradingThread",
    )
    trading_thread.start()
    logger.info("Trading thread started")

    # Start web server
    import uvicorn
    from moex_agent.webapp import app

    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting web server on port {port}")
    logger.info(f"Dashboard: http://0.0.0.0:{port}/")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # Reduce uvicorn noise
    )


if __name__ == "__main__":
    main()

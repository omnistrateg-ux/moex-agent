"""
MOEX Agent CLI

Unified command-line interface for all operations.

Usage:
    python -m moex_agent init-db              # Initialize database
    python -m moex_agent bootstrap           # Initialize local runtime baseline
    python -m moex_agent train                # Train ML models
    python -m moex_agent live                 # Run live signal loop
    python -m moex_agent web --port 8000      # Start web dashboard
    python -m moex_agent telegram-test "msg"  # Test Telegram integration
    python -m moex_agent status               # Show system status
    python -m moex_agent repair               # Repair runtime inconsistencies
"""
from __future__ import annotations

import argparse
import importlib
import json
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

SCHEMA_VERSION = "1.0"
SNAPSHOT_VERSION = "1.0"
BASELINES_FILE = Path("baselines/baselines.json")


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
    """Safely initialize minimal local runtime resources."""
    import sqlite3

    from .config_schema import load_config

    sections: dict[str, list[tuple[str, str]]] = {
        "Storage": [],
        "Models": [],
        "Runtime": [],
    }
    has_errors = False

    def add(section: str, level: str, message: str) -> None:
        nonlocal has_errors
        sections[section].append((level, message))
        if level == "ERROR":
            has_errors = True

    # Storage bootstrap
    try:
        cfg = load_config(args.config)
        sqlite_path = Path(cfg.sqlite_path)
    except Exception as exc:
        add("Storage", "ERROR", f"failed to load config: {exc}")
        sqlite_path = Path("data/moex_agent.sqlite")
        add("Storage", "SKIPPED", f"fallback sqlite path selected: {sqlite_path}")

    try:
        sqlite_parent = sqlite_path.parent
        if sqlite_parent.exists():
            add("Storage", "EXISTS", f"sqlite parent directory: {sqlite_parent}")
        else:
            sqlite_parent.mkdir(parents=True, exist_ok=True)
            add("Storage", "CREATED", f"sqlite parent directory: {sqlite_parent}")

        db_existed = sqlite_path.exists()
        conn = sqlite3.connect(str(sqlite_path))
        if db_existed:
            add("Storage", "EXISTS", f"sqlite database file: {sqlite_path}")
        else:
            add("Storage", "CREATED", f"sqlite database file: {sqlite_path}")

        table_sql = {
            "candles": """
                CREATE TABLE IF NOT EXISTS candles (
                  secid TEXT NOT NULL,
                  board TEXT NOT NULL,
                  interval INTEGER NOT NULL,
                  ts TEXT NOT NULL,
                  open REAL,
                  high REAL,
                  low REAL,
                  close REAL,
                  value REAL,
                  volume REAL,
                  PRIMARY KEY (secid, board, interval, ts)
                )
            """,
            "quotes": """
                CREATE TABLE IF NOT EXISTS quotes (
                  secid TEXT NOT NULL,
                  board TEXT NOT NULL,
                  ts TEXT NOT NULL,
                  last REAL,
                  bid REAL,
                  ask REAL,
                  numtrades REAL,
                  voltoday REAL,
                  valtoday REAL,
                  PRIMARY KEY (secid, board, ts)
                )
            """,
            "alerts": """
                CREATE TABLE IF NOT EXISTS alerts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_ts TEXT NOT NULL,
                  secid TEXT NOT NULL,
                  horizon TEXT NOT NULL,
                  p REAL NOT NULL,
                  signal_type TEXT NOT NULL,
                  entry REAL,
                  take REAL,
                  stop REAL,
                  ttl_minutes INTEGER,
                  anomaly_score REAL,
                  payload_json TEXT,
                  sent INTEGER DEFAULT 0
                )
            """,
            "state": """
                CREATE TABLE IF NOT EXISTS state (
                  key TEXT PRIMARY KEY,
                  value TEXT
                )
            """,
        }

        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {str(row[0]) for row in cur.fetchall()}

        for table in ["candles", "quotes", "alerts", "state"]:
            conn.execute(table_sql[table])
            if table in existing_tables:
                add("Storage", "EXISTS", f"table: {table}")
            else:
                add("Storage", "CREATED", f"table: {table}")

        conn.commit()
        conn.close()
    except Exception as exc:
        add("Storage", "ERROR", f"storage bootstrap failed: {exc}")

    # Models bootstrap
    models_dir = Path(args.models_dir)
    try:
        if models_dir.exists():
            add("Models", "EXISTS", f"models directory: {models_dir}")
        else:
            models_dir.mkdir(parents=True, exist_ok=True)
            add("Models", "CREATED", f"models directory: {models_dir}")

        meta_path = models_dir / "meta.json"
        if meta_path.exists():
            add("Models", "EXISTS", f"meta.json: {meta_path}")
        else:
            payload = {
                "schema_version": "1.0",
                "horizons": {},
            }
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            add("Models", "CREATED", f"meta.json: {meta_path}")
    except Exception as exc:
        add("Models", "ERROR", f"models bootstrap failed: {exc}")

    # Runtime optional bootstrap
    if args.create_system:
        if not args.output_dir:
            add("Runtime", "ERROR", "--output-dir is required when --create-system is set")
        else:
            runtime_args = argparse.Namespace(preset=args.preset, output_dir=args.output_dir)
            try:
                rc = cmd_create_system(runtime_args)
                if rc == 0:
                    add("Runtime", "CREATED", f"system registry and bundles: {args.output_dir}")
                else:
                    add("Runtime", "ERROR", f"system generation failed with code {rc}")
            except Exception as exc:
                add("Runtime", "ERROR", f"system generation failed: {exc}")
    else:
        add("Runtime", "SKIPPED", "system generation not requested")

    # Report
    print("Storage")
    for level, message in sections["Storage"]:
        print(f"{level}: {message}")

    print() 
    print("Models")
    for level, message in sections["Models"]:
        print(f"{level}: {message}")

    print()
    print("Runtime")
    for level, message in sections["Runtime"]:
        print(f"{level}: {message}")

    print()
    print("Overall")
    if has_errors:
        print("ERROR: bootstrap failed")
        return 1

    created_count = sum(1 for sec in sections.values() for lvl, _ in sec if lvl == "CREATED")
    if created_count > 0:
        print(f"CREATED: bootstrap completed ({created_count} resource(s) created)")
    else:
        print("EXISTS: bootstrap completed (nothing to create)")
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


def cmd_create_agent(args: argparse.Namespace) -> int:
    """Scaffold a new analyst agent prompt + config."""
    from .agent_factory import AGENT_TYPES, AgentBlueprint, create_agent_files

    goals = [g.strip() for g in args.goal if g.strip()] if args.goal else None
    blueprint = AgentBlueprint(
        name=args.name,
        role=args.role,
        agent_type=args.agent_type,
        timeframe=args.timeframe,
        risk_profile=args.risk_profile,
        goals=goals,
        input_data=args.input_data,
        output_data=args.output_data,
        algorithms=args.algorithm,
        risk_limits=args.risk_limit,
        files=args.file,
    )
    prompt_path, config_path = create_agent_files(blueprint, output_dir=args.output_dir)

    logger.info(f"Agent created: {blueprint.name}")
    logger.info(f"Prompt: {prompt_path}")
    logger.info(f"Config: {config_path}")
    return 0




def cmd_create_agent_bundle(args: argparse.Namespace) -> int:
    """Scaffold a full executable agent bundle directory."""
    from .agent_factory import AgentBlueprint, create_agent_bundle

    goals = [g.strip() for g in args.goal if g.strip()] if args.goal else None
    blueprint = AgentBlueprint(
        name=args.name,
        role=args.role,
        agent_type=args.agent_type,
        timeframe=args.timeframe,
        risk_profile=args.risk_profile,
        goals=goals,
        input_data=args.input_data,
        output_data=args.output_data,
        algorithms=args.algorithm,
        risk_limits=args.risk_limit,
        files=args.file,
    )
    bundle_dir = create_agent_bundle(blueprint, output_dir=args.output_dir)
    logger.info(f"Agent bundle created: {bundle_dir}")
    return 0

def cmd_agents_check(args: argparse.Namespace) -> int:
    """Validate generated agent configs."""
    from .agent_registry import RegistryValidationError, load_registry

    try:
        specs = load_registry(args.agents_dir)
    except RegistryValidationError as exc:
        logger.error(f"Agent registry validation failed: {exc}")
        return 1

    logger.info(f"Agent registry is valid: {len(specs)} config(s) in {args.agents_dir}")
    return 0


def cmd_agents_dry_run(args: argparse.Namespace) -> int:
    """Print deterministic dry-run orchestration plan from agent configs."""
    from .agent_registry import RegistryValidationError, build_dry_run_plan, load_registry

    try:
        specs = load_registry(args.agents_dir)
    except RegistryValidationError as exc:
        logger.error(f"Cannot build dry-run plan: {exc}")
        return 1

    print(build_dry_run_plan(specs))
    return 0


def cmd_validate_agent_spec(args: argparse.Namespace) -> int:
    """Validate a single agent specification JSON file."""
    import json

    from .agent_registry import validate_agent_spec

    config_path = Path(args.config)
    if not config_path.exists():
        print("VALIDATION RESULT")
        print()
        print(f"Agent: {config_path.stem}")
        print("Status: INVALID")
        print()
        print("Errors:")
        print(f"- file does not exist: {config_path}")
        return 1

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("VALIDATION RESULT")
        print()
        print(f"Agent: {config_path.stem}")
        print("Status: INVALID")
        print()
        print("Errors:")
        print(f"- invalid json: {exc}")
        return 1

    errors = validate_agent_spec(config_path)
    warnings = []
    if "schema_version" not in payload:
        warnings.append("missing schema_version")

    agent_name = payload.get("slug") or payload.get("name") or config_path.stem

    print("VALIDATION RESULT")
    print()
    print(f"Agent: {agent_name}")

    if errors:
        print("Status: INVALID")
        print()
        print("Errors:")
        for err in errors:
            print(f"- {err}")
        if warnings:
            print()
            print("Warnings:")
            for warning in warnings:
                print(f"- {warning}")
        return 1

    print("Status: VALID")
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


def cmd_create_system(args: argparse.Namespace) -> int:
    """Generate a complete MOEX agent system from a preset."""
    import json

    from .agent_factory import AgentBlueprint, create_agent_bundle
    from .agent_registry import build_dry_run_plan, load_registry, validate_agent_spec

    presets = {
        "core_moex_v1": [
            ("data", "Data Collector", "Сбор рыночных данных MOEX"),
            ("feature", "Feature Builder", "Инженерия признаков"),
            ("ml", "Signal Model", "Прогноз long/short вероятностей"),
            ("risk", "Risk Guard", "Контроль риска и kill-switch"),
            ("strategy", "Strategy Planner", "Построение торгового плана"),
            ("portfolio", "Portfolio Allocator", "Распределение капитала"),
            ("execution", "Execution Router", "Исполнение ордеров"),
            ("monitoring", "System Monitor", "Мониторинг и алерты"),
        ]
    }

    if args.preset not in presets:
        logger.error(f"Unknown preset: {args.preset}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    registry_agents = []
    validation_errors = []

    for agent_type, name, role in presets[args.preset]:
        blueprint = AgentBlueprint(
            name=name,
            role=role,
            agent_type=agent_type,
            timeframe="5m",
        )
        bundle_dir = create_agent_bundle(blueprint, output_dir=output_dir)
        config_path = bundle_dir / "agent_config.json"
        created.append(bundle_dir)

        errors = validate_agent_spec(config_path)
        if errors:
            validation_errors.append(f"{config_path}: " + "; ".join(errors))

        registry_agents.append(
            {
                "slug": blueprint.slug,
                "agent_type": agent_type,
                "config_path": str(config_path.relative_to(output_dir)),
            }
        )

    registry = {
        "schema_version": "1.0",
        "preset": args.preset,
        "generated_at": "static",
        "agents": registry_agents,
    }
    registry_path = output_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        specs = load_registry(output_dir)
        dry_run = build_dry_run_plan(specs)
        dry_run_status = "READY" if "RESULT: READY" in dry_run else "INCOMPLETE"
    except Exception as exc:
        dry_run = f"ERROR: {exc}"
        dry_run_status = "INCOMPLETE"
        validation_errors.append(str(exc))

    print("CREATE SYSTEM RESULT")
    print()
    print("Created bundles:")
    for path in created:
        print(f"- {path}")
    print()
    print(f"Registry: {registry_path}")
    print()
    if validation_errors:
        print("Validation: FAILED")
        for err in validation_errors:
            print(f"- {err}")
    else:
        print(f"Validation: OK ({len(created)}/{len(created)} specs)")
    print()
    print(f"Dry-run status: {dry_run_status}")

    return 1 if validation_errors else 0


def cmd_show_system(args: argparse.Namespace) -> int:
    """Show loaded system registry summary for orchestration readiness."""
    from .orchestrator_loader import SystemRegistryError, load_system_registry

    try:
        registry = load_system_registry(Path(args.registry))
    except SystemRegistryError as exc:
        logger.error(f"Failed to load system registry: {exc}")
        return 1

    print("SYSTEM REGISTRY")
    print()
    print(f"Preset: {registry.preset}")
    print(f"Total agents: {len(registry.agents)}")
    print(f"Status: {registry.status}")
    if registry.missing_types:
        print(f"Missing types: {', '.join(registry.missing_types)}")
    else:
        print("Missing types: none")

    print()
    print("Agents by type:")
    for agent_type, agents in registry.agents_by_type.items():
        if not agents:
            continue
        print(f"- {agent_type} ({len(agents)}):")
        for agent in agents:
            print(f"  • {agent.name} ({agent.slug}) @ {agent.config_path}")
    return 0



def _validate_snapshot_payload(payload: object) -> list[str]:
    """Validate system snapshot manifest payload and return errors."""
    if not isinstance(payload, dict):
        return ["snapshot root must be an object"]

    errors: list[str] = []
    required = ["schema_version", "snapshot_version", "preset", "generated_at", "agents"]
    missing = [f for f in required if f not in payload]
    if missing:
        errors.append("missing top-level fields: " + ", ".join(missing))

    agents = payload.get("agents")
    if not isinstance(agents, list):
        errors.append("field 'agents' must be a list")
        return errors

    if not agents:
        errors.append("field 'agents' must be a non-empty list")
        return errors

    seen_slugs: set[str] = set()
    for idx, entry in enumerate(agents):
        prefix = f"agents[{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue

        required_agent = ["name", "slug", "agent_type", "role", "timeframe", "config"]
        missing_agent = [f for f in required_agent if f not in entry]
        if missing_agent:
            errors.append(f"{prefix} missing fields: {', '.join(missing_agent)}")
            continue

        slug = str(entry.get("slug", "")).strip()
        if not slug:
            errors.append(f"{prefix} has empty slug")
        elif slug in seen_slugs:
            errors.append(f"duplicate slug: {slug}")
        else:
            seen_slugs.add(slug)

        cfg = entry.get("config")
        if not isinstance(cfg, dict):
            errors.append(f"{prefix}.config must be an object")
            continue

        if cfg.get("slug") != entry.get("slug"):
            errors.append(f"{prefix} config.slug mismatch")
        if cfg.get("agent_type") != entry.get("agent_type"):
            errors.append(f"{prefix} config.agent_type mismatch")
        if cfg.get("name") != entry.get("name"):
            errors.append(f"{prefix} config.name mismatch")

    return errors

def migrate_snapshot_payload(payload: dict) -> dict:
    """Migrate snapshot payload to latest canonical supported format."""
    errors = _validate_snapshot_payload(payload)
    if errors:
        raise ValueError("invalid snapshot: " + "; ".join(errors))

    in_version = str(payload.get("snapshot_version"))
    if in_version != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot_version: {in_version}")

    canonical_agents = []
    for entry in sorted(payload["agents"], key=lambda x: str(x["slug"])):
        normalized = dict(entry)
        normalized["name"] = str(entry["name"])
        normalized["slug"] = str(entry["slug"])
        normalized["agent_type"] = str(entry["agent_type"])
        normalized["role"] = str(entry["role"])
        normalized["timeframe"] = str(entry["timeframe"])

        cfg = dict(entry["config"])
        cfg["name"] = normalized["name"]
        cfg["slug"] = normalized["slug"]
        cfg["agent_type"] = normalized["agent_type"]
        cfg.setdefault("role", normalized["role"])
        cfg.setdefault("timeframe", normalized["timeframe"])
        cfg.setdefault("runtime_mode", "stub")
        deps = cfg.get("dependencies")
        if not isinstance(deps, dict):
            deps = {}
        req = deps.get("requires")
        prv = deps.get("provides")
        deps["requires"] = sorted(str(x) for x in req) if isinstance(req, list) else []
        deps["provides"] = sorted(str(x) for x in prv) if isinstance(prv, list) else []
        cfg["dependencies"] = deps
        normalized["config"] = cfg

        canonical_agents.append(normalized)

    canonical = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "preset": str(payload["preset"]),
        "generated_at": str(payload["generated_at"]),
        "agents": canonical_agents,
    }
    return canonical


def cmd_export_system(args: argparse.Namespace) -> int:
    """Export generated system into deterministic snapshot json."""
    from .orchestrator_loader import SystemRegistryError, load_system_registry

    messages: list[tuple[str, str]] = []

    def add(level: str, text: str) -> None:
        messages.append((level, text))

    try:
        registry = load_system_registry(Path(args.registry))
    except SystemRegistryError as exc:
        print("Export")
        print(f"ERROR: failed to load registry: {exc}")
        print()
        print("Overall")
        print("ERROR: export failed")
        return 1

    agents = []
    for agent in sorted(registry.agents, key=lambda x: x.slug):
        try:
            config_payload = json.loads(agent.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("Export")
            print(f"ERROR: failed to read {agent.config_path}: {exc}")
            print()
            print("Overall")
            print("ERROR: export failed")
            return 1

        agents.append(
            {
                "name": agent.name,
                "slug": agent.slug,
                "agent_type": agent.agent_type,
                "role": agent.role,
                "timeframe": agent.timeframe,
                "config": config_payload,
            }
        )

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "preset": registry.preset,
        "generated_at": registry.generated_at,
        "agents": agents,
    }
    snapshot = migrate_snapshot_payload(snapshot)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existed = out_path.exists()
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    add("EXISTS" if existed else "CREATED", f"snapshot: {out_path}")

    print("Export")
    for level, text in messages:
        print(f"{level}: {text}")
    print()
    print("Overall")
    print("CREATED: export completed")
    return 0


def cmd_validate_snapshot(args: argparse.Namespace) -> int:
    """Validate snapshot manifest independently from import flow."""
    snapshot_path = Path(args.snapshot)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Snapshot")
        print(f"ERROR: failed to read snapshot: {exc}")
        print()
        print("Overall")
        print("ERROR: snapshot invalid")
        return 1

    errors = _validate_snapshot_payload(payload)

    print("Snapshot")
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        print()
        print("Overall")
        print("ERROR: snapshot invalid")
        return 1

    print("OK: snapshot manifest is valid")
    print()
    print("Overall")
    print("OK: snapshot valid")
    return 0


def _snapshot_agent_compare_fields(entry: dict) -> dict:
    """Extract deterministic comparable fields from canonical snapshot agent entry."""
    cfg = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    deps = cfg.get("dependencies") if isinstance(cfg.get("dependencies"), dict) else {}
    requires = deps.get("requires") if isinstance(deps.get("requires"), list) else []
    provides = deps.get("provides") if isinstance(deps.get("provides"), list) else []

    runtime_adapter = cfg.get("runtime_adapter")
    if isinstance(runtime_adapter, dict):
        runtime_adapter_cmp = {
            "module": runtime_adapter.get("module"),
            "callable": runtime_adapter.get("callable"),
        }
    else:
        runtime_adapter_cmp = None

    return {
        "name": str(entry.get("name", "")),
        "agent_type": str(entry.get("agent_type", "")),
        "role": str(entry.get("role", "")),
        "timeframe": str(entry.get("timeframe", "")),
        "dependencies.requires": sorted(str(x) for x in requires),
        "dependencies.provides": sorted(str(x) for x in provides),
        "runtime_mode": cfg.get("runtime_mode"),
        "runtime_adapter": runtime_adapter_cmp,
    }


def _diff_canonical_snapshots(left: dict, right: dict) -> tuple[list[str], list[str], list[tuple[str, list[str]]]]:
    """Compute deterministic diff triples for two canonical snapshots."""
    left_by_slug = {str(a["slug"]): a for a in left["agents"]}
    right_by_slug = {str(a["slug"]): a for a in right["agents"]}

    added = sorted(slug for slug in right_by_slug if slug not in left_by_slug)
    removed = sorted(slug for slug in left_by_slug if slug not in right_by_slug)

    field_order = [
        "name",
        "agent_type",
        "role",
        "timeframe",
        "dependencies.requires",
        "dependencies.provides",
        "runtime_mode",
        "runtime_adapter",
    ]

    changed: list[tuple[str, list[str]]] = []
    for slug in sorted(set(left_by_slug.keys()) & set(right_by_slug.keys())):
        lcmp = _snapshot_agent_compare_fields(left_by_slug[slug])
        rcmp = _snapshot_agent_compare_fields(right_by_slug[slug])
        diffs = [field for field in field_order if lcmp[field] != rcmp[field]]
        if diffs:
            changed.append((slug, diffs))

    return added, removed, changed


def _print_snapshot_diff_report(added: list[str], removed: list[str], changed: list[tuple[str, list[str]]]) -> None:
    """Print snapshot diff report with stable deterministic ordering."""
    print("Added agents")
    if added:
        for slug in added:
            print(f"ADDED: {slug}")
    else:
        print("OK: none")

    print()
    print("Removed agents")
    if removed:
        for slug in removed:
            print(f"REMOVED: {slug}")
    else:
        print("OK: none")

    print()
    print("Changed agents")
    if changed:
        for slug, diffs in changed:
            print(f"CHANGED: {slug}")
            for field in diffs:
                print(f"CHANGED: {slug}.{field}")
    else:
        print("OK: none")

    print()
    print("Overall")
    if not added and not removed and not changed:
        print("OK: snapshots are identical")
    else:
        print("OK: snapshots differ")


def _resolve_registry_input(path_value: str) -> Path:
    """Resolve user input to registry.json path (file or containing directory)."""
    p = Path(path_value)
    if p.is_dir():
        return p / "registry.json"
    return p


def _canonical_snapshot_from_registry(registry_path: Path) -> dict:
    """Load system registry and convert to canonical snapshot-like payload."""
    from .orchestrator_loader import SystemRegistryError, load_system_registry

    try:
        system = load_system_registry(registry_path)
    except SystemRegistryError as exc:
        raise ValueError(str(exc)) from exc

    agents: list[dict] = []
    for agent in sorted(system.agents, key=lambda x: x.slug):
        try:
            cfg = json.loads(agent.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to read {agent.config_path}: {exc}") from exc

        agents.append(
            {
                "name": agent.name,
                "slug": agent.slug,
                "agent_type": agent.agent_type,
                "role": agent.role,
                "timeframe": agent.timeframe,
                "config": cfg,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "preset": system.preset,
        "generated_at": system.generated_at,
        "agents": agents,
    }
    return migrate_snapshot_payload(payload)


def cmd_diff_snapshot(args: argparse.Namespace) -> int:
    """Compare two snapshots and report deterministic structural differences."""
    try:
        left_raw = json.loads(Path(args.left).read_text(encoding="utf-8"))
    except Exception as exc:
        print("Overall")
        print(f"ERROR: failed to read left snapshot: {exc}")
        return 1

    try:
        right_raw = json.loads(Path(args.right).read_text(encoding="utf-8"))
    except Exception as exc:
        print("Overall")
        print(f"ERROR: failed to read right snapshot: {exc}")
        return 1

    try:
        left = migrate_snapshot_payload(left_raw)
    except ValueError as exc:
        print("Overall")
        print(f"ERROR: invalid left snapshot: {exc}")
        return 1

    try:
        right = migrate_snapshot_payload(right_raw)
    except ValueError as exc:
        print("Overall")
        print(f"ERROR: invalid right snapshot: {exc}")
        return 1

    added, removed, changed = _diff_canonical_snapshots(left, right)
    _print_snapshot_diff_report(added, removed, changed)
    return 0


def cmd_diff_system(args: argparse.Namespace) -> int:
    """Compare two systems (registry paths or system directories) deterministically."""
    left_registry = _resolve_registry_input(args.left)
    right_registry = _resolve_registry_input(args.right)

    try:
        left = _canonical_snapshot_from_registry(left_registry)
    except ValueError as exc:
        print("Overall")
        print(f"ERROR: invalid left system: {exc}")
        return 1

    try:
        right = _canonical_snapshot_from_registry(right_registry)
    except ValueError as exc:
        print("Overall")
        print(f"ERROR: invalid right system: {exc}")
        return 1

    added, removed, changed = _diff_canonical_snapshots(left, right)
    _print_snapshot_diff_report(added, removed, changed)
    return 0

def cmd_migrate_snapshot(args: argparse.Namespace) -> int:
    """Migrate snapshot to latest canonical deterministic format."""
    snapshot_path = Path(args.snapshot)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Migration")
        print(f"ERROR: failed to read snapshot: {exc}")
        print()
        print("Overall")
        print("ERROR: migration failed")
        return 1

    try:
        migrated = migrate_snapshot_payload(payload)
    except ValueError as exc:
        print("Migration")
        print(f"ERROR: {exc}")
        print()
        print("Overall")
        print("ERROR: migration failed")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Migration")
    print(f"CREATED: migrated snapshot: {out_path}")
    print("OK: migrated to snapshot_version 1.0")
    print()
    print("Overall")
    print("OK: migration completed")
    return 0


def cmd_import_system(args: argparse.Namespace) -> int:
    """Import deterministic system snapshot into bundle layout + registry."""
    from .orchestrator_loader import SystemRegistryError, load_system_registry

    default_prompt = "# AGENT PROMPT\n\nPlaceholder prompt scaffold generated by import-system.\n"
    default_readme = "# Agent Bundle\n\nPlaceholder README scaffold generated by import-system.\n"

    print_messages: list[tuple[str, str]] = []
    had_error = False

    def add(level: str, text: str) -> None:
        nonlocal had_error
        print_messages.append((level, text))
        if level == "ERROR":
            had_error = True

    snapshot_path = Path(args.snapshot)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Import")
        print(f"ERROR: failed to read snapshot: {exc}")
        print()
        print("Overall")
        print("ERROR: import failed")
        return 1

    try:
        payload = migrate_snapshot_payload(payload)
    except ValueError as exc:
        print("Import")
        print(f"ERROR: {exc}")
        print()
        print("Overall")
        print("ERROR: import failed")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_agents = []
    for entry in sorted(payload["agents"], key=lambda x: str(x.get("slug", ""))):
        missing_agent_fields = [field for field in ["name", "slug", "agent_type", "role", "timeframe", "config"] if field not in entry]
        if missing_agent_fields:
            add("ERROR", f"agent entry missing fields: {', '.join(missing_agent_fields)}")
            continue

        slug = str(entry.get("slug", "")).strip()
        if not slug:
            add("ERROR", "agent entry has empty slug")
            continue

        bundle_dir = output_dir / slug
        bundle_dir.mkdir(parents=True, exist_ok=True)

        config_path = bundle_dir / "agent_config.json"
        config_payload = entry.get("config")
        if not isinstance(config_payload, dict):
            add("ERROR", f"agent '{slug}' has non-object config")
            continue

        # normalize key identity with snapshot descriptor
        config_payload = dict(config_payload)
        config_payload["name"] = entry["name"]
        config_payload["slug"] = slug
        config_payload["agent_type"] = entry["agent_type"]
        config_payload["role"] = entry["role"]
        config_payload["timeframe"] = entry["timeframe"]

        existed_cfg = config_path.exists()
        config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        add("EXISTS" if existed_cfg else "CREATED", f"agent config: {config_path}")

        prompt_path = bundle_dir / "AGENT_PROMPT.md"
        if prompt_path.exists():
            add("EXISTS", f"prompt: {prompt_path}")
        else:
            prompt_path.write_text(default_prompt, encoding="utf-8")
            add("CREATED", f"prompt: {prompt_path}")

        readme_path = bundle_dir / "README.md"
        if readme_path.exists():
            add("EXISTS", f"readme: {readme_path}")
        else:
            readme_path.write_text(default_readme, encoding="utf-8")
            add("CREATED", f"readme: {readme_path}")

        src_dir = bundle_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        src_path = src_dir / f"{slug}.py"
        if src_path.exists():
            add("EXISTS", f"src: {src_path}")
        else:
            src_path.write_text(
                '"""Generated placeholder runtime module."""\n\n'
                "def run(payload):\n"
                "    return payload\n",
                encoding="utf-8",
            )
            add("CREATED", f"src: {src_path}")

        tests_dir = bundle_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        test_path = tests_dir / f"test_{slug}.py"
        if test_path.exists():
            add("EXISTS", f"test: {test_path}")
        else:
            test_path.write_text(
                "def test_placeholder():\n"
                "    assert True\n",
                encoding="utf-8",
            )
            add("CREATED", f"test: {test_path}")

        registry_agents.append(
            {
                "slug": slug,
                "agent_type": str(entry["agent_type"]),
                "config_path": f"{slug}/agent_config.json",
            }
        )

    registry_payload = {
        "schema_version": "1.0",
        "preset": payload["preset"],
        "generated_at": payload["generated_at"],
        "agents": sorted(registry_agents, key=lambda x: x["slug"]),
    }
    registry_path = output_dir / "registry.json"
    existed_registry = registry_path.exists()
    registry_path.write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    add("EXISTS" if existed_registry else "CREATED", f"registry: {registry_path}")

    try:
        load_system_registry(registry_path)
        add("EXISTS", "imported registry validation: READY")
    except SystemRegistryError as exc:
        add("ERROR", f"imported registry validation failed: {exc}")

    print("Import")
    for level, text in print_messages:
        print(f"{level}: {text}")
    print()
    print("Overall")
    if had_error:
        print("ERROR: import failed")
        return 1
    print("CREATED: import completed")
    return 0

def _next_paper_run_id(runs_dir: Path) -> str:
    """Allocate deterministic sequential paper run id."""
    max_idx = 0
    for child in runs_dir.iterdir() if runs_dir.exists() else []:
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("run_"):
            continue
        suffix = name[4:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return f"run_{max_idx + 1:06d}"


def _build_run_metrics(run_id: str, artifacts: dict) -> dict:
    """Build deterministic run metrics payload from available artifacts."""
    signals = artifacts.get("signals") if isinstance(artifacts.get("signals"), dict) else {}
    risk_state = artifacts.get("risk_state") if isinstance(artifacts.get("risk_state"), dict) else {}
    allocation = artifacts.get("allocation") if isinstance(artifacts.get("allocation"), dict) else {}
    alerts = artifacts.get("alerts")
    execution_report = artifacts.get("execution_report") if isinstance(artifacts.get("execution_report"), dict) else {}

    p_long = float(signals.get("probability_long", 0.0)) if isinstance(signals.get("probability_long"), (int, float)) else 0.0
    p_short = float(signals.get("probability_short", 0.0)) if isinstance(signals.get("probability_short"), (int, float)) else 0.0
    conf = float(signals.get("confidence", 0.0)) if isinstance(signals.get("confidence"), (int, float)) else 0.0
    signals_count = 1 if signals else 0

    if isinstance(alerts, dict):
        messages = alerts.get("messages")
        alerts_count = len(messages) if isinstance(messages, list) else (1 if alerts else 0)
    elif isinstance(alerts, list):
        alerts_count = len(alerts)
    else:
        alerts_count = 0

    allocation_total = 0.0
    for key in ("cash", "risk_budget"):
        v = allocation.get(key)
        if isinstance(v, (int, float)):
            allocation_total += float(v)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "signals_count": signals_count,
        "long_probability_avg": round(p_long if signals_count else 0.0, 6),
        "short_probability_avg": round(p_short if signals_count else 0.0, 6),
        "confidence_avg": round(conf if signals_count else 0.0, 6),
        "risk_kill_switch": bool(risk_state.get("kill_switch", False)),
        "allocation_total": round(allocation_total, 6),
        "alerts_count": int(alerts_count),
        "execution_status": str(execution_report.get("status", "UNKNOWN")),
    }


def _persist_paper_run_artifacts(
    runs_dir: Path,
    run_id: str,
    artifacts: dict,
    *,
    system,
    registry_path: Path,
    executed_agents: list[str],
) -> tuple[Path, list[str], dict]:
    """Persist deterministic paper-run manifest and produced artifacts."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    ordered_artifacts = [
        "market_data",
        "features",
        "signals",
        "risk_state",
        "trade_plan",
        "allocation",
        "execution_report",
        "alerts",
    ]
    artifact_files: list[str] = []
    for token in ordered_artifacts:
        if token not in artifacts:
            continue
        out_name = f"{token}.json"
        (run_dir / out_name).write_text(
            json.dumps(artifacts[token], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_files.append(out_name)

    status = "ok" if "execution_report" in artifacts else "warning"
    manifest = {
        "run_id": run_id,
        "status": status,
        "preset": system.preset,
        "registry_path": str(registry_path),
        "runtime_mode": "real",
        "executed_agents": list(executed_agents),
        "artifact_files": artifact_files,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = _build_run_metrics(run_id, artifacts)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir, artifact_files, metrics


def cmd_orchestrate(args: argparse.Namespace) -> int:
    """Build/simulate/paper-run deterministic orchestration for generated system."""
    from .orchestrator_loader import SystemRegistryError, build_execution_plan, load_system_registry
    from .runtime_contracts import AgentRuntimeError, run_execution_plan, run_execution_plan_with_artifacts
    from .semantic_contracts import SemanticContractError

    selected_modes = int(bool(args.dry_run)) + int(bool(args.simulate)) + int(bool(args.paper))
    if selected_modes != 1:
        logger.error("Specify exactly one mode: --dry-run, --simulate, or --paper")
        return 1

    try:
        system = load_system_registry(Path(args.registry))
    except SystemRegistryError as exc:
        print("ORCHESTRATION")
        print()
        print("STATUS: INCOMPLETE")
        print(f"Issues: {exc}")
        return 1

    if args.dry_run:
        try:
            plan = build_execution_plan(system)
        except SystemRegistryError as exc:
            print("ORCHESTRATION")
            print()
            print("STATUS: INCOMPLETE")
            print(f"Issues: {exc}")
            return 1

        print("ORCHESTRATION")
        print()
        print("Execution order:")
        for idx, agent in enumerate(plan, 1):
            print(f"{idx}. {agent.slug} [{agent.agent_type}]")
        print()
        print("STATUS: READY")
        return 0

    if args.simulate:
        try:
            plan = build_execution_plan(system)
            results = run_execution_plan(system, runtime_mode_override=args.runtime_mode, with_sample_data=args.with_sample_data)
        except (SystemRegistryError, AgentRuntimeError, SemanticContractError) as exc:
            print("ORCHESTRATION")
            print()
            print("STATUS: ERROR")
            print(f"Reason: {exc}")
            return 1

        print("ORCHESTRATION")
        print()
        print("Simulated execution:")
        for idx, (agent, result) in enumerate(zip(plan, results), 1):
            provides = sorted((result.get("provides") or {}).keys())
            print(f"{idx}. {agent.slug} [{agent.agent_type}] -> {', '.join(provides) if provides else 'no provides'}")
        print()
        print("STATUS: READY")
        return 0

    # paper mode
    runs_dir = Path(args.runs_dir)
    registry_path = Path(args.registry)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = _next_paper_run_id(runs_dir)

    try:
        results, artifact_store, executed_agents = run_execution_plan_with_artifacts(
            system,
            runtime_mode_override="real",
            with_sample_data=args.with_sample_data,
        )
        _ = results
        run_dir, artifact_files, _metrics = _persist_paper_run_artifacts(
            runs_dir,
            run_id,
            artifact_store,
            system=system,
            registry_path=registry_path,
            executed_agents=executed_agents,
        )
    except (SystemRegistryError, AgentRuntimeError, SemanticContractError, OSError, ValueError) as exc:
        print("Paper Run")
        print(f"ERROR: {exc}")
        print()
        print("Overall")
        print("ERROR: paper run failed")
        return 1

    print("Paper Run")
    print(f"CREATED: run directory: {run_dir}")
    print("OK: runtime mode forced to real")
    if artifact_files:
        print(f"CREATED: persisted artifacts: {', '.join(artifact_files)}")
    else:
        print("WARNING: no artifacts were produced")
    print("OK: execution adapter is paper-safe only")
    print()
    print("Overall")
    print("OK: paper run completed")
    return 0


def cmd_show_run(args: argparse.Namespace) -> int:
    """Show persisted run summary, artifacts, and metrics."""
    run_dir = Path(args.run)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"

    print("Run")
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        print()
        print("Artifacts")
        print("WARNING: not available")
        print()
        print("Metrics")
        print("WARNING: not available")
        print()
        print("Overall")
        print("ERROR: show-run failed")
        return 1

    if not manifest_path.exists():
        print(f"ERROR: missing manifest: {manifest_path}")
        print()
        print("Artifacts")
        print("WARNING: not available")
        print()
        print("Metrics")
        print("WARNING: not available")
        print()
        print("Overall")
        print("ERROR: show-run failed")
        return 1

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: failed to read manifest: {exc}")
        print()
        print("Artifacts")
        print("WARNING: not available")
        print()
        print("Metrics")
        print("WARNING: not available")
        print()
        print("Overall")
        print("ERROR: show-run failed")
        return 1

    print(f"OK: run_id: {manifest.get('run_id', 'unknown')}")
    print(f"OK: status: {manifest.get('status', 'unknown')}")
    print(f"OK: runtime_mode: {manifest.get('runtime_mode', 'unknown')}")
    print()

    print("Artifacts")
    artifact_files = manifest.get("artifact_files") if isinstance(manifest.get("artifact_files"), list) else []
    if artifact_files:
        for name in sorted(str(x) for x in artifact_files):
            if (run_dir / name).exists():
                print(f"OK: {name}")
            else:
                print(f"WARNING: missing listed artifact: {name}")
    else:
        print("WARNING: no artifacts listed")
    print()

    print("Metrics")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: failed to read metrics: {exc}")
            print()
            print("Overall")
            print("ERROR: show-run failed")
            return 1
        for key in [
            "schema_version",
            "run_id",
            "signals_count",
            "long_probability_avg",
            "short_probability_avg",
            "confidence_avg",
            "risk_kill_switch",
            "allocation_total",
            "alerts_count",
            "execution_status",
        ]:
            print(f"OK: {key}: {metrics.get(key)}")
    else:
        print("WARNING: metrics.json not found")
    print()

    print("Overall")
    print("OK: show-run completed")
    return 0


def _validate_replay_manifest(payload: object) -> list[str]:
    """Validate replay manifest structure."""
    if not isinstance(payload, dict):
        return ["manifest root must be an object"]

    required = [
        "run_id",
        "status",
        "preset",
        "registry_path",
        "runtime_mode",
        "executed_agents",
        "artifact_files",
    ]
    errors: list[str] = []
    missing = [k for k in required if k not in payload]
    if missing:
        errors.append("missing manifest fields: " + ", ".join(missing))

    if "executed_agents" in payload and not isinstance(payload.get("executed_agents"), list):
        errors.append("manifest.executed_agents must be a list")
    if "artifact_files" in payload and not isinstance(payload.get("artifact_files"), list):
        errors.append("manifest.artifact_files must be a list")

    for key in ["run_id", "status", "preset", "registry_path", "runtime_mode"]:
        if key in payload and not isinstance(payload.get(key), str):
            errors.append(f"manifest.{key} must be a string")
    return errors


def _collect_replay_validation(run_dir: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], bool, dict, dict[str, object]]:
    """Collect replay validation messages and parsed artifact payloads."""
    from .semantic_contracts import SemanticContractError, validate_semantic_output

    manifest_path = run_dir / "manifest.json"
    replay_msgs: list[tuple[str, str]] = []
    artifact_msgs: list[tuple[str, str]] = []
    has_errors = False
    manifest: dict = {}
    parsed_artifacts: dict[str, object] = {}

    def add_replay(level: str, text: str) -> None:
        nonlocal has_errors
        replay_msgs.append((level, text))
        if level == "ERROR":
            has_errors = True

    def add_art(level: str, text: str) -> None:
        nonlocal has_errors
        artifact_msgs.append((level, text))
        if level == "ERROR":
            has_errors = True

    if not run_dir.exists() or not run_dir.is_dir():
        add_replay("ERROR", f"run directory not found: {run_dir}")
    elif not manifest_path.exists():
        add_replay("ERROR", f"missing manifest: {manifest_path}")

    if not has_errors:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            add_replay("ERROR", f"failed to read manifest: {exc}")

    if not has_errors:
        errors = _validate_replay_manifest(manifest)
        if errors:
            for err in errors:
                add_replay("ERROR", err)
        else:
            add_replay("OK", "manifest structure is valid")

    produced_tokens = [
        "market_data",
        "features",
        "signals",
        "risk_state",
        "trade_plan",
        "allocation",
        "execution_report",
        "alerts",
    ]
    semantic_token_to_agent = {
        "signals": "ml",
        "risk_state": "risk",
        "trade_plan": "strategy",
        "allocation": "portfolio",
        "execution_report": "execution",
        "alerts": "monitoring",
    }

    if not has_errors:
        listed = manifest.get("artifact_files")
        listed_files = sorted(set(str(x) for x in listed)) if isinstance(listed, list) else []
        add_replay("OK", f"manifest references {len(listed_files)} artifact files")

        listed_set = set(listed_files)
        for token in produced_tokens:
            fname = f"{token}.json"
            if fname not in listed_set and not (run_dir / fname).exists():
                add_art("WARNING", f"optional artifact absent and unlisted: {fname}")

        for file_name in listed_files:
            artifact_path = run_dir / file_name
            if not artifact_path.exists() or not artifact_path.is_file():
                add_art("ERROR", f"listed artifact is missing: {file_name}")
                continue

            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception as exc:
                add_art("ERROR", f"invalid JSON in {file_name}: {exc}")
                continue

            parsed_artifacts[file_name] = payload
            token = file_name[:-5] if file_name.endswith('.json') else ""
            if token in semantic_token_to_agent:
                try:
                    validate_semantic_output(semantic_token_to_agent[token], {token: payload})
                    add_art("OK", f"semantic artifact valid: {file_name}")
                except SemanticContractError as exc:
                    add_art("ERROR", f"semantic artifact invalid ({file_name}): {exc}")
            else:
                add_art("OK", f"artifact parseable: {file_name}")

    return replay_msgs, artifact_msgs, has_errors, manifest, parsed_artifacts


def cmd_replay_run(args: argparse.Namespace) -> int:
    """Validate persisted paper run artifacts for replay/audit usage."""
    replay_msgs, artifact_msgs, has_errors, _manifest, _artifacts = _collect_replay_validation(Path(args.run))

    print("Replay")
    for lvl, msg in replay_msgs:
        print(f"{lvl}: {msg}")
    if not replay_msgs:
        print("WARNING: no replay checks executed")
    print()

    print("Artifacts")
    for lvl, msg in artifact_msgs:
        print(f"{lvl}: {msg}")
    if not artifact_msgs:
        print("WARNING: no artifacts to validate")
    print()

    print("Overall")
    if has_errors:
        print("ERROR: replay validation failed")
        return 1
    print("OK: replay validation passed")
    return 0


def _compute_run_diff(
    left_manifest: dict,
    right_manifest: dict,
    left_payloads: dict[str, object],
    right_payloads: dict[str, object],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compute deterministic diff vectors for two validated paper runs."""
    artifact_order = [
        "market_data.json",
        "features.json",
        "signals.json",
        "risk_state.json",
        "trade_plan.json",
        "allocation.json",
        "execution_report.json",
        "alerts.json",
    ]

    left_files = set(str(x) for x in left_manifest.get("artifact_files", []))
    right_files = set(str(x) for x in right_manifest.get("artifact_files", []))

    added = [f for f in artifact_order if f in right_files and f not in left_files]
    removed = [f for f in artifact_order if f in left_files and f not in right_files]
    changed = [f for f in artifact_order if f in left_files and f in right_files and left_payloads.get(f) != right_payloads.get(f)]

    manifest_fields = ["preset", "runtime_mode", "executed_agents"]
    manifest_changes = [k for k in manifest_fields if left_manifest.get(k) != right_manifest.get(k)]
    return added, removed, changed, manifest_changes


def cmd_diff_run(args: argparse.Namespace) -> int:
    """Diff two persisted paper runs deterministically."""
    left_dir = Path(args.left)
    right_dir = Path(args.right)

    left_replay, left_artifacts_msgs, left_err, left_manifest, left_payloads = _collect_replay_validation(left_dir)
    right_replay, right_artifacts_msgs, right_err, right_manifest, right_payloads = _collect_replay_validation(right_dir)

    if left_err or right_err:
        print("Overall")
        if left_err:
            print("ERROR: left run failed replay validation")
            for lvl, msg in left_replay + left_artifacts_msgs:
                if lvl == "ERROR":
                    print(f"ERROR: left: {msg}")
        if right_err:
            print("ERROR: right run failed replay validation")
            for lvl, msg in right_replay + right_artifacts_msgs:
                if lvl == "ERROR":
                    print(f"ERROR: right: {msg}")
        return 1

    added, removed, changed, manifest_changes = _compute_run_diff(
        left_manifest,
        right_manifest,
        left_payloads,
        right_payloads,
    )

    print("Added artifacts")
    if added:
        for name in added:
            print(f"ADDED: {name}")
    else:
        print("OK: none")
    print()

    print("Removed artifacts")
    if removed:
        for name in removed:
            print(f"REMOVED: {name}")
    else:
        print("OK: none")
    print()

    print("Changed artifacts")
    if changed:
        for name in changed:
            print(f"CHANGED: {name}")
    else:
        print("OK: none")
    print()

    print("Manifest changes")
    if manifest_changes:
        for key in manifest_changes:
            print(f"CHANGED: {key}")
    else:
        print("OK: none")
    print()

    print("Overall")
    if not added and not removed and not changed and not manifest_changes:
        print("OK: runs are identical")
    else:
        print("OK: runs differ")
    return 0


def cmd_diff_metrics(args: argparse.Namespace) -> int:
    """Compare metrics.json between two persisted runs deterministically."""
    left_dir = Path(args.left)
    right_dir = Path(args.right)

    left_exists, left_metrics, left_err = _load_run_metrics(left_dir)
    right_exists, right_metrics, right_err = _load_run_metrics(right_dir)

    print("Metric Diff")
    if left_err:
        print(f"ERROR: left metrics invalid: {left_err}")
        print()
        print("Overall")
        print("ERROR: metric diff failed")
        return 1
    if right_err:
        print(f"ERROR: right metrics invalid: {right_err}")
        print()
        print("Overall")
        print("ERROR: metric diff failed")
        return 1
    if not left_exists:
        print(f"ERROR: left metrics.json missing: {left_dir / 'metrics.json'}")
        print()
        print("Overall")
        print("ERROR: metric diff failed")
        return 1
    if not right_exists:
        print(f"ERROR: right metrics.json missing: {right_dir / 'metrics.json'}")
        print()
        print("Overall")
        print("ERROR: metric diff failed")
        return 1

    assert left_metrics is not None and right_metrics is not None
    fields = [
        "signals_count",
        "long_probability_avg",
        "short_probability_avg",
        "confidence_avg",
        "risk_kill_switch",
        "allocation_total",
        "alerts_count",
        "execution_status",
    ]
    changed: list[str] = []
    for field in fields:
        lv = left_metrics.get(field)
        rv = right_metrics.get(field)
        if lv != rv:
            changed.append(f"{field}: {lv} -> {rv}")

    if changed:
        for msg in changed:
            print(f"CHANGED: {msg}")
    else:
        print("OK: no metric changes")
    print()
    print("Overall")
    print("OK: metric diff completed")
    return 0


def _load_baselines_payload(path: Path = BASELINES_FILE) -> dict:
    """Load deterministic baselines payload or return empty scaffold."""
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "baselines": {}}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read baselines file: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("baselines payload root must be object")

    baselines = payload.get("baselines")
    if baselines is None:
        baselines = {}
    if not isinstance(baselines, dict):
        raise ValueError("baselines field must be object")

    normalized = {str(k): str(v) for k, v in baselines.items()}
    return {
        "schema_version": str(payload.get("schema_version", SCHEMA_VERSION)),
        "baselines": {k: normalized[k] for k in sorted(normalized.keys())},
    }


def _write_baselines_payload(payload: dict, path: Path = BASELINES_FILE) -> None:
    """Persist baselines payload deterministically."""
    baselines = payload.get("baselines") if isinstance(payload.get("baselines"), dict) else {}
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "baselines": {str(k): str(v) for k, v in sorted(((str(k), str(v)) for k, v in baselines.items()), key=lambda item: item[0])},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_baseline_dir_from_args(args: argparse.Namespace) -> Path:
    """Resolve baseline path from --baseline or --baseline-name CLI args."""
    baseline = getattr(args, "baseline", None)
    baseline_name = getattr(args, "baseline_name", None)

    if bool(baseline) == bool(baseline_name):
        raise ValueError("specify exactly one of --baseline or --baseline-name")

    if baseline:
        return Path(str(baseline))

    payload = _load_baselines_payload(BASELINES_FILE)
    baselines = payload.get("baselines", {})
    if str(baseline_name) not in baselines:
        raise ValueError(f"unknown baseline name: {baseline_name}")
    return Path(str(baselines[str(baseline_name)]))


def cmd_mark_baseline(args: argparse.Namespace) -> int:
    """Validate run and create/update named baseline mapping."""
    run_dir = Path(args.run)
    name = str(args.name)

    replay_msgs, artifact_msgs, has_errors, _manifest, _artifacts = _collect_replay_validation(run_dir)
    if has_errors:
        print("Baseline")
        print(f"ERROR: run validation failed for '{name}'")
        for lvl, msg in replay_msgs + artifact_msgs:
            if lvl == "ERROR":
                print(f"ERROR: {msg}")
        print()
        print("Overall")
        print("ERROR: baseline not updated")
        return 1

    try:
        payload = _load_baselines_payload(BASELINES_FILE)
    except ValueError as exc:
        print("Baseline")
        print(f"ERROR: {exc}")
        print()
        print("Overall")
        print("ERROR: baseline not updated")
        return 1

    baselines = payload.get("baselines") if isinstance(payload.get("baselines"), dict) else {}
    existed = name in baselines
    baselines[name] = str(run_dir)
    payload["baselines"] = baselines

    try:
        _write_baselines_payload(payload, BASELINES_FILE)
    except Exception as exc:
        print("Baseline")
        print(f"ERROR: failed to write baselines file: {exc}")
        print()
        print("Overall")
        print("ERROR: baseline not updated")
        return 1

    print("Baseline")
    if existed:
        print(f"OK: updated baseline '{name}'")
    else:
        print(f"CREATED: baseline '{name}'")
    print(f"OK: run mapped to {run_dir}")
    print()
    print("Overall")
    print("OK: baseline saved")
    return 0


def cmd_list_baselines(args: argparse.Namespace) -> int:
    """List named baselines in deterministic sorted order."""
    _ = args
    try:
        payload = _load_baselines_payload(BASELINES_FILE)
    except ValueError as exc:
        print("Baselines")
        print(f"ERROR: {exc}")
        print()
        print("Overall")
        print("ERROR: unable to list baselines")
        return 1

    baselines = payload.get("baselines") if isinstance(payload.get("baselines"), dict) else {}
    print("Baselines")
    if not baselines:
        print("OK: none")
    else:
        for name in sorted(baselines.keys()):
            print(f"OK: {name} -> {baselines[name]}")
    print()
    print("Overall")
    print("OK: listed baselines")
    return 0


def _load_run_metrics(run_dir: Path) -> tuple[bool, dict | None, str | None]:
    """Load metrics.json from run dir; return (exists, payload, error)."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return False, None, None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return True, None, f"failed to read metrics.json: {exc}"
    if not isinstance(payload, dict):
        return True, None, "metrics.json root must be object"
    return True, payload, None


def cmd_run_regression(args: argparse.Namespace) -> int:
    """Classify candidate run differences against baseline as warnings vs failures."""
    try:
        baseline_dir = _resolve_baseline_dir_from_args(args)
    except ValueError as exc:
        print("Regression")
        print(f"ERROR: {exc}")
        print()
        print("Warnings")
        print("OK: none")
        print()
        print("Failures")
        print(f"ERROR: {exc}")
        print()
        print("Overall")
        print("ERROR: regression failed")
        return 1

    candidate_dir = Path(args.candidate)

    b_replay, b_art, b_err, b_manifest, b_payloads = _collect_replay_validation(baseline_dir)
    c_replay, c_art, c_err, c_manifest, c_payloads = _collect_replay_validation(candidate_dir)

    failures: list[str] = []
    warnings: list[str] = []

    if b_err:
        failures.append("baseline run failed replay validation")
        for lvl, msg in b_replay + b_art:
            if lvl == "ERROR":
                failures.append(f"baseline: {msg}")
    if c_err:
        failures.append("candidate run failed replay validation")
        for lvl, msg in c_replay + c_art:
            if lvl == "ERROR":
                failures.append(f"candidate: {msg}")

    if not failures:
        added, removed, changed, manifest_changes = _compute_run_diff(
            b_manifest,
            c_manifest,
            b_payloads,
            c_payloads,
        )

        if "runtime_mode" in manifest_changes:
            failures.append("runtime_mode changed")
        if "executed_agents" in manifest_changes:
            failures.append("executed_agents changed")
        for file_name in removed:
            failures.append(f"artifact missing in candidate: {file_name}")
        candidate_files = set(str(x) for x in c_manifest.get("artifact_files", []))
        if "execution_report.json" not in candidate_files:
            failures.append("execution_report.json missing in candidate")

        warning_artifacts = {
            "signals.json",
            "risk_state.json",
            "trade_plan.json",
            "allocation.json",
            "alerts.json",
            "market_data.json",
            "features.json",
        }
        for file_name in changed:
            if file_name in warning_artifacts:
                warnings.append(f"changed artifact: {file_name}")

        b_metrics_exists, b_metrics, b_metrics_err = _load_run_metrics(baseline_dir)
        c_metrics_exists, c_metrics, c_metrics_err = _load_run_metrics(candidate_dir)

        if b_metrics_err:
            failures.append(f"baseline metrics invalid: {b_metrics_err}")
        if c_metrics_err:
            failures.append(f"candidate metrics invalid: {c_metrics_err}")

        if b_metrics_exists and not c_metrics_exists:
            failures.append("candidate metrics.json missing while baseline has metrics.json")

        if b_metrics and c_metrics:
            b_conf = float(b_metrics.get("confidence_avg", 0.0)) if isinstance(b_metrics.get("confidence_avg"), (int, float)) else 0.0
            c_conf = float(c_metrics.get("confidence_avg", 0.0)) if isinstance(c_metrics.get("confidence_avg"), (int, float)) else 0.0
            if c_conf < b_conf:
                warnings.append(f"metric confidence_avg decreased: {b_conf} -> {c_conf}")

            b_alerts = int(b_metrics.get("alerts_count", 0)) if isinstance(b_metrics.get("alerts_count"), int) else 0
            c_alerts = int(c_metrics.get("alerts_count", 0)) if isinstance(c_metrics.get("alerts_count"), int) else 0
            if c_alerts > b_alerts:
                warnings.append(f"metric alerts_count increased: {b_alerts} -> {c_alerts}")

            b_sig = int(b_metrics.get("signals_count", 0)) if isinstance(b_metrics.get("signals_count"), int) else 0
            c_sig = int(c_metrics.get("signals_count", 0)) if isinstance(c_metrics.get("signals_count"), int) else 0
            if c_sig != b_sig:
                warnings.append(f"metric signals_count changed: {b_sig} -> {c_sig}")

            b_kill = bool(b_metrics.get("risk_kill_switch", False))
            c_kill = bool(c_metrics.get("risk_kill_switch", False))
            if (not b_kill) and c_kill:
                failures.append("metric risk_kill_switch changed from false to true")

            b_alloc = float(b_metrics.get("allocation_total", 0.0)) if isinstance(b_metrics.get("allocation_total"), (int, float)) else 0.0
            c_alloc = float(c_metrics.get("allocation_total", 0.0)) if isinstance(c_metrics.get("allocation_total"), (int, float)) else 0.0
            if b_alloc > 0.0 and c_alloc == 0.0:
                failures.append("metric allocation_total changed from >0 to 0")

            c_exec = str(c_metrics.get("execution_status", ""))
            if c_exec != "PAPER_SIMULATED":
                failures.append(f"metric execution_status is not PAPER_SIMULATED: {c_exec}")

    print("Regression")
    if failures:
        print("ERROR: regression gate failed")
    elif warnings:
        print("WARNING: differences detected")
    else:
        print("OK: no regression differences")
    print()

    print("Warnings")
    if warnings:
        for msg in sorted(warnings):
            print(f"WARNING: {msg}")
    else:
        print("OK: none")
    print()

    print("Failures")
    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
    else:
        print("OK: none")
    print()

    print("Overall")
    if failures:
        print("ERROR: regression failed")
        return 1
    if warnings:
        print("OK: regression passed with warnings")
    else:
        print("OK: regression passed")
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Repair common non-fatal runtime inconsistencies detected by doctor."""
    import sqlite3

    from .agent_registry import load_registry, RegistryValidationError
    from .orchestrator_loader import load_system_registry, SystemRegistryError

    sections: dict[str, list[tuple[str, str]]] = {
        "Storage": [],
        "Models": [],
        "Runtime": [],
    }
    has_errors = False

    def add(section: str, level: str, message: str) -> None:
        nonlocal has_errors
        sections[section].append((level, message))
        if level == "ERROR":
            has_errors = True

    # Storage repair
    try:
        from .config_schema import load_config

        cfg = load_config(args.config)
        sqlite_path = Path(cfg.sqlite_path)
    except Exception as exc:
        sqlite_path = Path("data/moex_agent.sqlite")
        add("Storage", "WARNING", f"failed to load config, using fallback sqlite path: {sqlite_path} ({exc})")

    try:
        sqlite_parent = sqlite_path.parent
        if sqlite_parent.exists():
            add("Storage", "EXISTS", f"sqlite parent directory: {sqlite_parent}")
        else:
            sqlite_parent.mkdir(parents=True, exist_ok=True)
            add("Storage", "CREATED", f"sqlite parent directory: {sqlite_parent}")

        db_existed = sqlite_path.exists()
        conn = sqlite3.connect(str(sqlite_path))
        if db_existed:
            add("Storage", "EXISTS", f"sqlite database file: {sqlite_path}")
        else:
            add("Storage", "CREATED", f"sqlite database file: {sqlite_path}")

        table_sql = {
            "candles": """
                CREATE TABLE IF NOT EXISTS candles (
                  secid TEXT NOT NULL,
                  board TEXT NOT NULL,
                  interval INTEGER NOT NULL,
                  ts TEXT NOT NULL,
                  open REAL,
                  high REAL,
                  low REAL,
                  close REAL,
                  value REAL,
                  volume REAL,
                  PRIMARY KEY (secid, board, interval, ts)
                )
            """,
            "quotes": """
                CREATE TABLE IF NOT EXISTS quotes (
                  secid TEXT NOT NULL,
                  board TEXT NOT NULL,
                  ts TEXT NOT NULL,
                  last REAL,
                  bid REAL,
                  ask REAL,
                  numtrades REAL,
                  voltoday REAL,
                  valtoday REAL,
                  PRIMARY KEY (secid, board, ts)
                )
            """,
            "alerts": """
                CREATE TABLE IF NOT EXISTS alerts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_ts TEXT NOT NULL,
                  secid TEXT NOT NULL,
                  horizon TEXT NOT NULL,
                  p REAL NOT NULL,
                  signal_type TEXT NOT NULL,
                  entry REAL,
                  take REAL,
                  stop REAL,
                  ttl_minutes INTEGER,
                  anomaly_score REAL,
                  payload_json TEXT,
                  sent INTEGER DEFAULT 0
                )
            """,
            "state": """
                CREATE TABLE IF NOT EXISTS state (
                  key TEXT PRIMARY KEY,
                  value TEXT
                )
            """,
        }
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {str(row[0]) for row in cur.fetchall()}

        for table in ["candles", "quotes", "alerts", "state"]:
            conn.execute(table_sql[table])
            if table in existing_tables:
                add("Storage", "EXISTS", f"table: {table}")
            else:
                add("Storage", "REPAIRED", f"created missing table: {table}")

        conn.commit()
        conn.close()
    except Exception as exc:
        add("Storage", "ERROR", f"storage repair failed: {exc}")

    # Models repair
    models_dir = Path(args.models_dir)
    default_meta = {"schema_version": "1.0", "horizons": {}}

    try:
        if models_dir.exists():
            add("Models", "EXISTS", f"models directory: {models_dir}")
        else:
            models_dir.mkdir(parents=True, exist_ok=True)
            add("Models", "CREATED", f"models directory: {models_dir}")

        meta_path = models_dir / "meta.json"
        meta_payload: dict = {}
        if not meta_path.exists():
            meta_path.write_text(json.dumps(default_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            meta_payload = {"schema_version": "1.0", "horizons": {}}
            add("Models", "CREATED", f"meta.json: {meta_path}")
        else:
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("meta.json top-level must be object")
                meta_payload = loaded
                add("Models", "EXISTS", f"meta.json: {meta_path}")
            except Exception:
                meta_path.write_text(json.dumps(default_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                meta_payload = {"schema_version": "1.0", "horizons": {}}
                add("Models", "REPAIRED", f"replaced invalid meta.json: {meta_path}")

        # normalize deterministic structure
        if not isinstance(meta_payload.get("horizons"), dict):
            meta_payload["horizons"] = {}

        model_files = sorted(models_dir.glob("model_time_*.joblib"))
        file_horizons = sorted(path.stem.replace("model_time_", "", 1) for path in model_files)

        changed = False
        for horizon in file_horizons:
            if horizon not in meta_payload["horizons"]:
                meta_payload["horizons"][horizon] = {
                    "path": f"{models_dir.as_posix()}/model_time_{horizon}.joblib"
                }
                changed = True

        if changed:
            ordered_horizons = {k: meta_payload["horizons"][k] for k in sorted(meta_payload["horizons"].keys())}
            meta_payload["horizons"] = ordered_horizons
            meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            add("Models", "REPAIRED", f"synced orphan model files into meta.json: {', '.join(file_horizons)}")
        else:
            add("Models", "EXISTS", "meta.json model horizons are already synchronized")

        missing_refs: list[str] = []
        for horizon, info in sorted(meta_payload.get("horizons", {}).items()):
            if not isinstance(info, dict) or not isinstance(info.get("path"), str):
                missing_refs.append(f"{horizon}(missing path)")
                continue
            if not Path(info["path"]).exists():
                missing_refs.append(f"{horizon}({info['path']})")
        if missing_refs:
            add("Models", "WARNING", "meta.json references missing model files: " + ", ".join(missing_refs))
    except Exception as exc:
        add("Models", "ERROR", f"models repair failed: {exc}")

    # Runtime repair
    registry_path = Path(args.registry)
    bundles_root = Path(args.output_dir) if args.output_dir else registry_path.parent

    def _rebuild_registry(root: Path, target_registry: Path) -> tuple[bool, str]:
        specs = load_registry(root)
        agents = []
        for spec in sorted(specs, key=lambda x: x.slug):
            rel = spec.source_file.resolve().relative_to(root.resolve())
            agents.append(
                {
                    "slug": spec.slug,
                    "agent_type": spec.agent_type,
                    "config_path": rel.as_posix(),
                }
            )

        payload = {
            "schema_version": "1.0",
            "preset": "repaired",
            "generated_at": "static",
            "agents": agents,
        }
        target_registry.parent.mkdir(parents=True, exist_ok=True)
        target_registry.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Validate that generated registry is loadable
        load_system_registry(target_registry)
        return True, f"registry rebuilt at {target_registry} from {root}"

    try:
        if registry_path.exists():
            try:
                load_system_registry(registry_path)
                add("Runtime", "EXISTS", f"registry is valid: {registry_path}")
            except SystemRegistryError as exc:
                add("Runtime", "WARNING", f"registry is broken: {exc}")
                try:
                    ok, msg = _rebuild_registry(bundles_root, registry_path)
                    if ok:
                        add("Runtime", "REPAIRED", msg)
                except (RegistryValidationError, SystemRegistryError, ValueError) as rebuild_exc:
                    add("Runtime", "WARNING", f"registry rebuild not possible: {rebuild_exc}")
        else:
            if not bundles_root.exists():
                add("Runtime", "SKIPPED", f"registry missing and bundles root missing: {bundles_root}")
            else:
                try:
                    ok, msg = _rebuild_registry(bundles_root, registry_path)
                    if ok:
                        add("Runtime", "REPAIRED", msg)
                except (RegistryValidationError, SystemRegistryError, ValueError) as rebuild_exc:
                    add("Runtime", "WARNING", f"registry missing; rebuild not possible: {rebuild_exc}")
    except Exception as exc:
        add("Runtime", "ERROR", f"runtime repair failed: {exc}")

    print("Storage")
    for level, message in sections["Storage"]:
        print(f"{level}: {message}")

    print()
    print("Models")
    for level, message in sections["Models"]:
        print(f"{level}: {message}")

    print()
    print("Runtime")
    for level, message in sections["Runtime"]:
        print(f"{level}: {message}")

    print()
    print("Overall")
    if has_errors:
        print("ERROR: repair failed")
        return 1

    repaired_count = sum(1 for sec in sections.values() for lvl, _ in sec if lvl == "REPAIRED")
    created_count = sum(1 for sec in sections.values() for lvl, _ in sec if lvl == "CREATED")
    if repaired_count > 0:
        print(f"REPAIRED: repair completed ({repaired_count} fix(es), {created_count} create(s))")
    elif created_count > 0:
        print(f"CREATED: repair completed ({created_count} resource(s) created)")
    else:
        print("EXISTS: repair completed (nothing to change)")

    return 0



def _expected_provide_token(agent_type: str) -> str:
    mapping = {
        "data": "market_data",
        "feature": "features",
        "ml": "signals",
        "risk": "risk_state",
        "strategy": "trade_plan",
        "portfolio": "allocation",
        "execution": "execution_report",
        "monitoring": "alerts",
    }
    return mapping.get((agent_type or "").strip().lower(), "")


def _compatibility_from_canonical_snapshot(canonical: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]], bool]:
    """Return (artifact_msgs, runtime_msgs, contract_msgs, incompatible)."""
    artifact_msgs: list[tuple[str, str]] = []
    runtime_msgs: list[tuple[str, str]] = []
    contract_msgs: list[tuple[str, str]] = []
    incompatible = False

    def add_art(level: str, msg: str) -> None:
        nonlocal incompatible
        artifact_msgs.append((level, msg))
        if level == "ERROR":
            incompatible = True

    def add_run(level: str, msg: str) -> None:
        nonlocal incompatible
        runtime_msgs.append((level, msg))
        if level == "ERROR":
            incompatible = True

    def add_con(level: str, msg: str) -> None:
        nonlocal incompatible
        contract_msgs.append((level, msg))
        if level == "ERROR":
            incompatible = True

    # Artifact format checks
    if str(canonical.get("schema_version")) == SCHEMA_VERSION:
        add_art("OK", f"schema_version supported: {SCHEMA_VERSION}")
    else:
        add_art("ERROR", f"unsupported schema_version: {canonical.get('schema_version')}")

    if str(canonical.get("snapshot_version")) == SNAPSHOT_VERSION:
        add_art("OK", f"snapshot_version supported: {SNAPSHOT_VERSION}")
    else:
        add_art("ERROR", f"unsupported snapshot_version: {canonical.get('snapshot_version')}")

    agents = canonical.get("agents") if isinstance(canonical.get("agents"), list) else []
    by_type: dict[str, list[dict]] = {}
    for a in agents:
        by_type.setdefault(str(a.get("agent_type", "")), []).append(a)

    required_types = ["data", "feature", "ml", "risk", "strategy", "portfolio", "execution", "monitoring"]
    missing_types = [t for t in required_types if not by_type.get(t)]
    if missing_types:
        add_run("ERROR", "missing required agent types: " + ", ".join(missing_types))
    else:
        add_run("OK", "all required agent types are present")

    # execution-plan-like dependency check
    providers: dict[str, set[str]] = {}
    for a in agents:
        cfg = a.get("config") if isinstance(a.get("config"), dict) else {}
        deps = cfg.get("dependencies") if isinstance(cfg.get("dependencies"), dict) else {}
        provides = deps.get("provides") if isinstance(deps.get("provides"), list) else []
        for token in provides:
            providers.setdefault(str(token), set()).add(str(a.get("slug", "")))

    unresolved: list[str] = []
    for a in sorted(agents, key=lambda x: str(x.get("slug", ""))):
        cfg = a.get("config") if isinstance(a.get("config"), dict) else {}
        deps = cfg.get("dependencies") if isinstance(cfg.get("dependencies"), dict) else {}
        requires = deps.get("requires") if isinstance(deps.get("requires"), list) else []
        for token in requires:
            if not providers.get(str(token)):
                unresolved.append(f"{a.get('slug')} requires '{token}' but no provider exists")

    if unresolved:
        add_run("ERROR", "execution plan incompatibility: " + "; ".join(unresolved))
    else:
        add_run("OK", "execution plan can be built from dependency tokens")

    # runtime + contract checks per agent
    supported_runtime = {"stub", "real"}
    for a in sorted(agents, key=lambda x: str(x.get("slug", ""))):
        slug = str(a.get("slug", ""))
        agent_type = str(a.get("agent_type", ""))
        cfg = a.get("config") if isinstance(a.get("config"), dict) else {}

        runtime_mode = str(cfg.get("runtime_mode", "stub"))
        if runtime_mode not in supported_runtime:
            add_run("ERROR", f"{slug}: unsupported runtime_mode '{runtime_mode}'")
        elif runtime_mode == "real":
            adapter = cfg.get("runtime_adapter")
            if not isinstance(adapter, dict) or not adapter.get("module") or not adapter.get("callable"):
                add_run("ERROR", f"{slug}: runtime_adapter required for runtime_mode=real")
            else:
                add_run("OK", f"{slug}: real runtime adapter configured")
                try:
                    mod = importlib.import_module(str(adapter.get("module")))
                    diag_fn = getattr(mod, "get_real_adapter_diagnostics", None)
                    if callable(diag_fn):
                        diag = diag_fn(cfg)
                        if isinstance(diag, dict):
                            if not bool(diag.get("available", False)):
                                details = diag.get("details") or []
                                details_text = "; ".join(str(x) for x in details) if isinstance(details, list) else str(details)
                                add_run("ERROR", f"{slug}: adapter dependencies unavailable: {details_text}")
                            else:
                                add_run("OK", f"{slug}: adapter dependencies available")
                        elif isinstance(diag, list):
                            errors = [str(msg) for lvl, msg in diag if str(lvl).upper() == "ERROR"]
                            warnings = [str(msg) for lvl, msg in diag if str(lvl).upper() == "WARNING"]
                            if errors:
                                add_run("ERROR", f"{slug}: adapter dependencies unavailable: {'; '.join(errors)}")
                            elif warnings:
                                add_run("WARNING", f"{slug}: adapter diagnostics warnings: {'; '.join(warnings)}")
                            else:
                                add_run("OK", f"{slug}: adapter dependencies available")
                        else:
                            add_run("WARNING", f"{slug}: adapter diagnostics returned unsupported payload")
                    else:
                        add_run("WARNING", f"{slug}: adapter diagnostics helper is not implemented")
                except Exception as exc:
                    add_run("ERROR", f"{slug}: failed to run adapter diagnostics: {exc}")

        deps = cfg.get("dependencies") if isinstance(cfg.get("dependencies"), dict) else None
        if not isinstance(deps, dict):
            add_con("ERROR", f"{slug}: dependencies must be an object")
            continue
        req = deps.get("requires")
        prv = deps.get("provides")
        if not isinstance(req, list) or not isinstance(prv, list):
            add_con("ERROR", f"{slug}: dependencies.requires/provides must be lists")
            continue

        expected = _expected_provide_token(agent_type)
        if not expected:
            add_con("ERROR", f"{slug}: unknown agent_type '{agent_type}'")
            continue
        if expected not in [str(x) for x in prv]:
            add_con("ERROR", f"{slug}: semantic contract mismatch, expected provides token '{expected}'")
        else:
            add_con("OK", f"{slug}: semantic contract token '{expected}' present")

        cfg_agent_type = str(cfg.get("agent_type", ""))
        if cfg_agent_type and cfg_agent_type != agent_type:
            add_con("ERROR", f"{slug}: config.agent_type contradicts entry.agent_type")

    if not runtime_msgs:
        add_run("WARNING", "no runtime checks executed")
    if not contract_msgs:
        add_con("WARNING", "no contract checks executed")

    return artifact_msgs, runtime_msgs, contract_msgs, incompatible


def cmd_check_compatibility(args: argparse.Namespace) -> int:
    """Check system/snapshot compatibility with current runtime expectations."""
    source_kind = "snapshot" if args.snapshot else "registry"

    try:
        if args.snapshot:
            payload = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            canonical = migrate_snapshot_payload(payload)
        else:
            registry_path = _resolve_registry_input(args.registry)
            canonical = _canonical_snapshot_from_registry(registry_path)
    except Exception as exc:
        print("Artifact")
        print(f"ERROR: failed to load {source_kind}: {exc}")
        print()
        print("Runtime")
        print("ERROR: compatibility checks skipped")
        print()
        print("Contracts")
        print("ERROR: compatibility checks skipped")
        print()
        print("Overall")
        print("ERROR: incompatible")
        return 1

    artifact_msgs, runtime_msgs, contract_msgs, incompatible = _compatibility_from_canonical_snapshot(canonical)

    print("Artifact")
    for lvl, msg in artifact_msgs:
        print(f"{lvl}: {msg}")

    print()
    print("Runtime")
    for lvl, msg in runtime_msgs:
        print(f"{lvl}: {msg}")

    print()
    print("Contracts")
    for lvl, msg in contract_msgs:
        print(f"{lvl}: {msg}")

    print()
    print("Overall")
    if incompatible:
        print("ERROR: incompatible")
        return 1
    warn_only = any(lvl == "WARNING" for lvl, _ in (artifact_msgs + runtime_msgs + contract_msgs))
    if warn_only:
        print("WARNING: compatible with warnings")
    else:
        print("OK: compatible")
    return 0

def cmd_doctor(args: argparse.Namespace) -> int:
    """Run deterministic startup diagnostics for runtime readiness."""
    import sqlite3

    from .config_schema import load_config
    from .orchestrator_loader import SystemRegistryError, build_execution_plan, load_system_registry

    severity_order = {"OK": 0, "WARNING": 1, "ERROR": 2}
    overall = 0

    sections: dict[str, list[tuple[str, str]]] = {
        "Storage": [],
        "Models": [],
        "Runtime": [],
    }

    def add(section: str, level: str, message: str) -> None:
        nonlocal overall
        sections[section].append((level, message))
        overall = max(overall, severity_order[level])

    # Storage checks
    try:
        cfg = load_config(args.config)
        sqlite_path = Path(cfg.sqlite_path)
        add("Storage", "OK", f"sqlite path configured: {sqlite_path}")
    except Exception as exc:
        sqlite_path = Path("data/moex_agent.sqlite")
        add("Storage", "ERROR", f"failed to load config: {exc}")
        add("Storage", "WARNING", f"fallback sqlite path: {sqlite_path}")

    if sqlite_path.exists():
        add("Storage", "OK", f"database file exists: {sqlite_path}")
    else:
        add("Storage", "WARNING", f"database file missing: {sqlite_path}")

    required_tables = ["candles", "quotes", "alerts", "state"]
    present_tables: set[str] = set()
    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        present_tables = {str(row[0]) for row in cur.fetchall()}
        conn.close()
        add("Storage", "OK", f"table discovery completed ({len(present_tables)} table(s))")
    except Exception as exc:
        add("Storage", "ERROR", f"unable to inspect sqlite tables: {exc}")

    missing_tables = sorted(t for t in required_tables if t not in present_tables)
    if missing_tables:
        add("Storage", "WARNING", f"missing required tables: {', '.join(missing_tables)}")
    else:
        add("Storage", "OK", "all required tables are present")

    # Models checks
    models_dir = Path(args.models_dir)
    if models_dir.exists() and models_dir.is_dir():
        add("Models", "OK", f"models directory exists: {models_dir}")
    else:
        add("Models", "ERROR", f"models directory missing: {models_dir}")

    meta_path = models_dir / "meta.json"
    meta_payload: dict[str, dict] = {}
    meta_valid = False
    if meta_path.exists():
        add("Models", "OK", f"meta.json exists: {meta_path}")
        try:
            raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw_meta, dict):
                meta_payload = raw_meta
                meta_valid = True
                add("Models", "OK", "meta.json is valid JSON")
            else:
                add("Models", "ERROR", "meta.json must contain an object at top-level")
        except json.JSONDecodeError as exc:
            add("Models", "ERROR", f"meta.json invalid JSON: {exc.msg}")
    else:
        add("Models", "WARNING", f"meta.json missing: {meta_path}")

    model_files = sorted(models_dir.glob("model_time_*.joblib"))
    horizons_from_files = sorted(path.stem.replace("model_time_", "", 1) for path in model_files)
    add("Models", "OK", f"discovered model files: {len(model_files)}")

    if meta_valid:
        meta_horizons = sorted(meta_payload.keys())
        orphans = sorted(h for h in horizons_from_files if h not in meta_payload)
        if orphans:
            add("Models", "WARNING", f"model files missing in meta.json: {', '.join(orphans)}")
        else:
            add("Models", "OK", "all discovered model files are listed in meta.json")

        missing_model_paths: list[str] = []
        for horizon in meta_horizons:
            info = meta_payload.get(horizon)
            model_ref = info.get("path") if isinstance(info, dict) else None
            if not model_ref:
                missing_model_paths.append(f"{horizon}(missing path)")
                continue
            if not Path(model_ref).exists():
                missing_model_paths.append(f"{horizon}({model_ref})")

        if missing_model_paths:
            add("Models", "WARNING", "meta.json references missing models: " + ", ".join(missing_model_paths))
        else:
            add("Models", "OK", "all meta.json model references exist")

    # Runtime checks
    registry_path = Path(args.registry)
    if registry_path.exists():
        add("Runtime", "OK", f"registry file exists: {registry_path}")
        try:
            system = load_system_registry(registry_path)
            try:
                plan = build_execution_plan(system)
                add("Runtime", "OK", f"execution plan READY ({len(plan)} agent(s))")
            except SystemRegistryError as exc:
                add("Runtime", "WARNING", f"execution plan INCOMPLETE: {exc}")
        except SystemRegistryError as exc:
            add("Runtime", "WARNING", f"registry INCOMPLETE: {exc}")
    else:
        add("Runtime", "WARNING", f"registry not found: {registry_path}")
        add("Runtime", "WARNING", "execution plan INCOMPLETE")

    print("Storage")
    for level, message in sections["Storage"]:
        print(f"{level}: {message}")

    print()
    print("Models")
    for level, message in sections["Models"]:
        print(f"{level}: {message}")

    print()
    print("Runtime")
    for level, message in sections["Runtime"]:
        print(f"{level}: {message}")

    overall_label = "OK" if overall == 0 else "WARNING" if overall == 1 else "ERROR"
    print()
    print("Overall")
    print(f"{overall_label}: runtime readiness {overall_label}")

    return 0 if overall < 2 else 1

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
    sub = subparsers.add_parser("bootstrap", help="Safely initialize local runtime baseline")
    sub.add_argument("--models-dir", default="models", help="Path to models directory (default: models)")
    sub.add_argument("--create-system", action="store_true", help="Also create system registry and bundles")
    sub.add_argument("--preset", default="core_moex_v1", choices=["core_moex_v1"], help="System preset for --create-system")
    sub.add_argument("--output-dir", help="Output directory for --create-system")

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

    from .agent_factory import AGENT_TYPES

    # create-agent
    sub = subparsers.add_parser("create-agent", help="Scaffold a new MOEX analyst agent")
    sub.add_argument("--name", required=True, help="Human-friendly agent name")
    sub.add_argument("--role", required=True, help="Role description of the sub-agent")
    sub.add_argument(
        "--agent-type",
        required=True,
        choices=list(AGENT_TYPES),
        help="Specialized MOEX agent type",
    )
    sub.add_argument("--timeframe", default="5m", help="Primary timeframe, e.g. 5m/1h/1d")
    sub.add_argument("--risk-profile", default="balanced", help="Risk profile label")
    sub.add_argument("--goal", action="append", default=[], help="Goal (repeatable)")
    sub.add_argument("--input-data", action="append", default=[], help="Input data item (repeatable)")
    sub.add_argument("--output-data", action="append", default=[], help="Output data item (repeatable)")
    sub.add_argument("--algorithm", action="append", default=[], help="Algorithm item (repeatable)")
    sub.add_argument("--risk-limit", action="append", default=[], help="Risk limit item (repeatable)")
    sub.add_argument("--file", action="append", default=[], help="File to create item (repeatable)")
    sub.add_argument("--output-dir", default="agents", help="Directory for generated files")

    # create-agent-bundle
    sub = subparsers.add_parser("create-agent-bundle", help="Scaffold agent prompt + JSON bundle")
    sub.add_argument("--name", required=True, help="Human-friendly agent name")
    sub.add_argument("--role", required=True, help="Role description of the sub-agent")
    sub.add_argument(
        "--agent-type",
        required=True,
        choices=list(AGENT_TYPES),
        help="Specialized MOEX agent type",
    )
    sub.add_argument("--timeframe", default="5m", help="Primary timeframe, e.g. 5m/1h/1d")
    sub.add_argument("--risk-profile", default="balanced", help="Risk profile label")
    sub.add_argument("--goal", action="append", default=[], help="Goal (repeatable)")
    sub.add_argument("--input-data", action="append", default=[], help="Input data item (repeatable)")
    sub.add_argument("--output-data", action="append", default=[], help="Output data item (repeatable)")
    sub.add_argument("--algorithm", action="append", default=[], help="Algorithm item (repeatable)")
    sub.add_argument("--risk-limit", action="append", default=[], help="Risk limit item (repeatable)")
    sub.add_argument("--file", action="append", default=[], help="File to create item (repeatable)")
    sub.add_argument("--output-dir", default="agents", help="Directory for generated files")

    # agents-check
    sub = subparsers.add_parser("agents-check", help="Validate generated agent configs")
    sub.add_argument("--agents-dir", default="agents", help="Directory with generated agent JSON configs")

    # agents-dry-run
    sub = subparsers.add_parser("agents-dry-run", help="Print deterministic orchestration dry-run plan")
    sub.add_argument("--agents-dir", default="agents", help="Directory with generated agent JSON configs")

    # validate-agent-spec
    sub = subparsers.add_parser("validate-agent-spec", help="Validate a single agent JSON specification")
    sub.add_argument("--config", required=True, help="Path to agent JSON config")

    # create-system
    sub = subparsers.add_parser("create-system", help="Generate complete MOEX agent system from preset")
    sub.add_argument("--preset", required=True, choices=["core_moex_v1"], help="System preset name")
    sub.add_argument("--output-dir", required=True, help="Output directory for generated system")

    # show-system
    sub = subparsers.add_parser("show-system", help="Load and show generated system registry summary")
    sub.add_argument("--registry", required=True, help="Path to registry.json")

    # export-system
    sub = subparsers.add_parser("export-system", help="Export generated system into deterministic snapshot")
    sub.add_argument("--registry", required=True, help="Path to registry.json")
    sub.add_argument("--out", required=True, help="Path to snapshot json")

    # import-system
    sub = subparsers.add_parser("import-system", help="Import deterministic snapshot into system bundles")
    sub.add_argument("--snapshot", required=True, help="Path to snapshot json")
    sub.add_argument("--output-dir", required=True, help="Output directory for imported system")

    # validate-snapshot
    sub = subparsers.add_parser("validate-snapshot", help="Validate system snapshot manifest")
    sub.add_argument("--snapshot", required=True, help="Path to snapshot json")

    # migrate-snapshot
    sub = subparsers.add_parser("migrate-snapshot", help="Migrate snapshot to latest canonical format")
    sub.add_argument("--snapshot", required=True, help="Input snapshot json")
    sub.add_argument("--out", required=True, help="Output migrated snapshot json")

    # diff-snapshot
    sub = subparsers.add_parser("diff-snapshot", help="Compare two canonical snapshots")
    sub.add_argument("--left", required=True, help="Path to left snapshot json")
    sub.add_argument("--right", required=True, help="Path to right snapshot json")

    # diff-system
    sub = subparsers.add_parser("diff-system", help="Compare two generated systems (dirs or registries)")
    sub.add_argument("--left", required=True, help="Left system directory or registry.json")
    sub.add_argument("--right", required=True, help="Right system directory or registry.json")

    # check-compatibility
    sub = subparsers.add_parser("check-compatibility", help="Check system or snapshot compatibility with runtime")
    group = sub.add_mutually_exclusive_group(required=True)
    group.add_argument("--registry", help="System directory or registry.json")
    group.add_argument("--snapshot", help="Snapshot json path")

    # doctor
    sub = subparsers.add_parser("doctor", help="Run startup diagnostics for runtime readiness")
    sub.add_argument("--models-dir", default="models", help="Path to models directory (default: models)")
    sub.add_argument("--registry", default="registry.json", help="Path to system registry.json (default: registry.json)")

    # repair
    sub = subparsers.add_parser("repair", help="Repair common runtime inconsistencies detected by doctor")
    sub.add_argument("--models-dir", default="models", help="Path to models directory (default: models)")
    sub.add_argument("--registry", default="registry.json", help="Path to system registry.json (default: registry.json)")
    sub.add_argument("--output-dir", help="Bundles root to use for registry rebuild (default: registry parent)")

    # orchestrate
    sub = subparsers.add_parser("orchestrate", help="Build dependency-aware execution plan")
    mode_group = sub.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true", help="Build execution plan only")
    mode_group.add_argument("--simulate", action="store_true", help="Execute runtime stubs/adapters with contracts")
    mode_group.add_argument("--paper", action="store_true", help="Execute real-mode adapters and persist paper-run artifacts")
    sub.add_argument("--registry", required=True, help="Path to registry.json")
    sub.add_argument("--runtime-mode", choices=["stub", "real"], help="Override runtime mode for --simulate only")
    sub.add_argument("--with-sample-data", action="store_true", help="Seed runtime artifacts with deterministic sample market scenario")
    sub.add_argument("--runs-dir", default="runs", help="Runs directory for --paper artifacts (default: runs)")

    # show-run
    sub = subparsers.add_parser("show-run", help="Show persisted paper run summary and metrics")
    sub.add_argument("--run", required=True, help="Path to paper run directory")

    # replay-run
    sub = subparsers.add_parser("replay-run", help="Validate persisted paper run for replay/audit")
    sub.add_argument("--run", required=True, help="Path to paper run directory")

    # diff-run
    sub = subparsers.add_parser("diff-run", help="Compare two persisted paper runs")
    sub.add_argument("--left", required=True, help="Path to left paper run directory")
    sub.add_argument("--right", required=True, help="Path to right paper run directory")

    # diff-metrics
    sub = subparsers.add_parser("diff-metrics", help="Compare metrics.json between two persisted runs")
    sub.add_argument("--left", required=True, help="Path to left paper run directory")
    sub.add_argument("--right", required=True, help="Path to right paper run directory")

    # run-regression
    sub = subparsers.add_parser("run-regression", help="Classify candidate run differences vs baseline")
    baseline_group = sub.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument("--baseline", help="Path to baseline paper run directory")
    baseline_group.add_argument("--baseline-name", help="Named baseline from baselines/baselines.json")
    sub.add_argument("--candidate", required=True, help="Path to candidate paper run directory")

    # mark-baseline
    sub = subparsers.add_parser("mark-baseline", help="Create or update named baseline mapping")
    sub.add_argument("--run", required=True, help="Path to paper run directory")
    sub.add_argument("--name", required=True, help="Baseline name")

    # list-baselines
    sub = subparsers.add_parser("list-baselines", help="List named baseline mappings")

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
    elif args.command == "create-agent":
        return cmd_create_agent(args)
    elif args.command == "create-agent-bundle":
        return cmd_create_agent_bundle(args)
    elif args.command == "agents-check":
        return cmd_agents_check(args)
    elif args.command == "agents-dry-run":
        return cmd_agents_dry_run(args)
    elif args.command == "validate-agent-spec":
        return cmd_validate_agent_spec(args)
    elif args.command == "create-system":
        return cmd_create_system(args)
    elif args.command == "show-system":
        return cmd_show_system(args)
    elif args.command == "export-system":
        return cmd_export_system(args)
    elif args.command == "import-system":
        return cmd_import_system(args)
    elif args.command == "validate-snapshot":
        return cmd_validate_snapshot(args)
    elif args.command == "migrate-snapshot":
        return cmd_migrate_snapshot(args)
    elif args.command == "diff-snapshot":
        return cmd_diff_snapshot(args)
    elif args.command == "diff-system":
        return cmd_diff_system(args)
    elif args.command == "check-compatibility":
        return cmd_check_compatibility(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "repair":
        return cmd_repair(args)
    elif args.command == "orchestrate":
        return cmd_orchestrate(args)
    elif args.command == "show-run":
        return cmd_show_run(args)
    elif args.command == "replay-run":
        return cmd_replay_run(args)
    elif args.command == "diff-run":
        return cmd_diff_run(args)
    elif args.command == "diff-metrics":
        return cmd_diff_metrics(args)
    elif args.command == "run-regression":
        return cmd_run_regression(args)
    elif args.command == "mark-baseline":
        return cmd_mark_baseline(args)
    elif args.command == "list-baselines":
        return cmd_list_baselines(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

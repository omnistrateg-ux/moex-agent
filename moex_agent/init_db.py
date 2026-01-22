from __future__ import annotations

from pathlib import Path

from .config import load_config
from .storage import connect, init_db


def main() -> None:
    cfg = load_config()
    conn = connect(cfg.sqlite_path)
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    init_db(conn, schema_path)
    print(f"DB ready: {cfg.sqlite_path}")


if __name__ == "__main__":
    main()

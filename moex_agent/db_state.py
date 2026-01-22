"""
PostgreSQL State Storage — Хранение состояния торговли в БД.

Заменяет файловое хранение (paper_state.json) на PostgreSQL.
Обеспечивает персистентность между деплоями.

Использование:
    from moex_agent.db_state import StateStorage

    storage = StateStorage()
    state = storage.load_state()
    storage.save_state(state)
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Попытка импорта psycopg2 (PostgreSQL)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    logger.warning("psycopg2 not installed. PostgreSQL storage disabled.")


# Дефолтное состояние
DEFAULT_STATE = {
    "initial_capital": 100000,
    "cash": 100000,
    "equity": 100000,
    "margin_used": 0.0,
    "positions": {},
    "closed_trades": [],
    "daily_pnl": 0,
    "weekly_pnl": 0,
    "consecutive_losses": 0,
    "kill_switch_active": False,
    "kill_switch_reason": "",
    "start_time": None,
    "last_update": None,
}


class StateStorage:
    """
    Хранилище состояния торговли.

    Приоритет:
    1. PostgreSQL (если DATABASE_URL установлен)
    2. SQLite (если sqlite_path указан)
    3. Файл JSON (fallback)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        json_path: str = "data/margin_paper_state.json",
    ):
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.sqlite_path = sqlite_path
        self.json_path = json_path

        self._conn = None
        self._storage_type = self._detect_storage_type()

        logger.info(f"StateStorage initialized: {self._storage_type}")

        # Инициализация таблицы если PostgreSQL
        if self._storage_type == "postgresql":
            self._init_postgres_table()

    def _detect_storage_type(self) -> str:
        """Определить тип хранилища."""
        if self.database_url and HAS_POSTGRES:
            return "postgresql"
        elif self.sqlite_path:
            return "sqlite"
        else:
            return "json"

    def _get_postgres_conn(self):
        """Получить соединение с PostgreSQL."""
        if not self._conn or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
        return self._conn

    def _init_postgres_table(self):
        """Создать таблицу если не существует."""
        if not HAS_POSTGRES:
            return

        try:
            conn = self._get_postgres_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_state (
                        id SERIAL PRIMARY KEY,
                        key VARCHAR(50) UNIQUE NOT NULL,
                        state JSONB NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Создаём индекс
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trading_state_key
                    ON trading_state(key)
                """)

            conn.commit()
            logger.info("PostgreSQL table 'trading_state' ready")

        except Exception as e:
            logger.error(f"Failed to init PostgreSQL table: {e}")

    def load_state(self, key: str = "main") -> Dict[str, Any]:
        """Загрузить состояние."""
        if self._storage_type == "postgresql":
            return self._load_from_postgres(key)
        elif self._storage_type == "sqlite":
            return self._load_from_sqlite(key)
        else:
            return self._load_from_json()

    def save_state(self, state: Dict[str, Any], key: str = "main") -> bool:
        """Сохранить состояние."""
        state["last_update"] = datetime.now().isoformat()

        if self._storage_type == "postgresql":
            return self._save_to_postgres(state, key)
        elif self._storage_type == "sqlite":
            return self._save_to_sqlite(state, key)
        else:
            return self._save_to_json(state)

    # === PostgreSQL ===

    def _load_from_postgres(self, key: str) -> Dict[str, Any]:
        """Загрузить из PostgreSQL."""
        try:
            conn = self._get_postgres_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT state FROM trading_state WHERE key = %s",
                    (key,)
                )
                row = cur.fetchone()

                if row:
                    state = row["state"]
                    logger.info(f"Loaded state from PostgreSQL: {len(state.get('closed_trades', []))} trades")
                    return state
                else:
                    logger.info("No state in PostgreSQL, using default")
                    return DEFAULT_STATE.copy()

        except Exception as e:
            logger.error(f"Failed to load from PostgreSQL: {e}")
            return DEFAULT_STATE.copy()

    def _save_to_postgres(self, state: Dict[str, Any], key: str) -> bool:
        """Сохранить в PostgreSQL."""
        try:
            conn = self._get_postgres_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_state (key, state, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (key)
                    DO UPDATE SET state = %s, updated_at = CURRENT_TIMESTAMP
                """, (key, json.dumps(state), json.dumps(state)))

            conn.commit()
            logger.debug(f"Saved state to PostgreSQL: equity={state.get('equity', 0)}")
            return True

        except Exception as e:
            logger.error(f"Failed to save to PostgreSQL: {e}")
            return False

    # === SQLite ===

    def _load_from_sqlite(self, key: str) -> Dict[str, Any]:
        """Загрузить из SQLite."""
        import sqlite3

        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row

            # Создаём таблицу если нет
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_state (
                    key TEXT PRIMARY KEY,
                    state TEXT,
                    updated_at TEXT
                )
            """)

            cur = conn.execute(
                "SELECT state FROM trading_state WHERE key = ?",
                (key,)
            )
            row = cur.fetchone()
            conn.close()

            if row:
                return json.loads(row["state"])
            return DEFAULT_STATE.copy()

        except Exception as e:
            logger.error(f"Failed to load from SQLite: {e}")
            return DEFAULT_STATE.copy()

    def _save_to_sqlite(self, state: Dict[str, Any], key: str) -> bool:
        """Сохранить в SQLite."""
        import sqlite3

        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.execute("""
                INSERT OR REPLACE INTO trading_state (key, state, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(state), datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")
            return False

    # === JSON File ===

    def _load_from_json(self) -> Dict[str, Any]:
        """Загрузить из JSON файла."""
        try:
            from pathlib import Path
            path = Path(self.json_path)

            if path.exists():
                with open(path) as f:
                    return json.load(f)
            return DEFAULT_STATE.copy()

        except Exception as e:
            logger.error(f"Failed to load from JSON: {e}")
            return DEFAULT_STATE.copy()

    def _save_to_json(self, state: Dict[str, Any]) -> bool:
        """Сохранить в JSON файл."""
        try:
            from pathlib import Path
            path = Path(self.json_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            return True

        except Exception as e:
            logger.error(f"Failed to save to JSON: {e}")
            return False

    def close(self):
        """Закрыть соединения."""
        if self._conn and not self._conn.closed:
            self._conn.close()


# Синглтон
_storage: Optional[StateStorage] = None

def get_storage() -> StateStorage:
    """Получить инстанс хранилища."""
    global _storage
    if _storage is None:
        _storage = StateStorage()
    return _storage


def load_trading_state(key: str = "main") -> Dict[str, Any]:
    """Загрузить состояние торговли."""
    return get_storage().load_state(key)


def save_trading_state(state: Dict[str, Any], key: str = "main") -> bool:
    """Сохранить состояние торговли."""
    return get_storage().save_state(state, key)


# Тестирование
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    storage = StateStorage()
    print(f"Storage type: {storage._storage_type}")

    # Загрузка
    state = storage.load_state()
    print(f"Loaded: equity={state.get('equity')}, trades={len(state.get('closed_trades', []))}")

    # Тестовое сохранение
    state["test_field"] = "test_value"
    success = storage.save_state(state)
    print(f"Save: {'OK' if success else 'FAILED'}")

    # Проверка
    state2 = storage.load_state()
    print(f"Verify: test_field={state2.get('test_field')}")

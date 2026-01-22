"""
MOEX Agent Configuration Schema (Pydantic V2)

Provides validated configuration with clear error messages.
All config keys have sensible defaults except 'universe.tickers' which is required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class HorizonConfig(BaseModel):
    """Single horizon configuration."""
    name: str
    minutes: int = Field(ge=1)


class StorageConfig(BaseModel):
    """Database storage configuration."""
    sqlite_path: Path = Field(default=Path("data/moex_agent.sqlite"))

    @field_validator("sqlite_path", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v


class UniverseConfig(BaseModel):
    """Market universe configuration."""
    engine: str = "stock"
    market: str = "shares"
    board: str = "TQBR"
    tickers: List[str] = Field(min_length=1)

    @field_validator("tickers", mode="before")
    @classmethod
    def filter_tickers(cls, v: Any) -> List[str]:
        """Filter out any non-string or empty values."""
        if isinstance(v, list):
            return [t for t in v if isinstance(t, str) and t.strip()]
        return v


class PriceExitConfig(BaseModel):
    """Price-based exit configuration."""
    enabled: bool = True
    take_atr: float = Field(default=0.8, ge=0)
    stop_atr: float = Field(default=0.6, ge=0)


class SignalsConfig(BaseModel):
    """Signal generation configuration."""
    horizons: List[HorizonConfig] = Field(
        default=[
            HorizonConfig(name="5m", minutes=5),
            HorizonConfig(name="10m", minutes=10),
            HorizonConfig(name="30m", minutes=30),
            HorizonConfig(name="1h", minutes=60),
            HorizonConfig(name="1d", minutes=1440),
            HorizonConfig(name="1w", minutes=10080),
        ]
    )
    p_threshold: float = Field(default=0.35, ge=0, le=1)
    cooldown_minutes: int = Field(default=30, ge=1)
    top_n_anomalies: int = Field(default=10, ge=1)
    price_exit: PriceExitConfig = Field(default_factory=PriceExitConfig)


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_spread_bps: float = Field(default=200, ge=0)
    min_turnover_rub_5m: float = Field(default=1_000_000, ge=0)


class QwenConfig(BaseModel):
    """Qwen LLM configuration."""
    enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    max_tokens: int = Field(default=500, ge=1)
    temperature: float = Field(default=0.3, ge=0, le=2)


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""
    enabled: bool = False
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    send_recommendations: List[str] = Field(
        default=["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]
    )

    @model_validator(mode="after")
    def validate_enabled(self) -> "TelegramConfig":
        import os
        # Load from environment if not set in config
        if not self.bot_token:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.chat_id:
            self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        if self.enabled:
            if not self.bot_token:
                raise ValueError(
                    "telegram.bot_token is required when telegram.enabled=true\n"
                    "Set TELEGRAM_BOT_TOKEN env var or in config.yaml"
                )
            if not self.chat_id:
                raise ValueError(
                    "telegram.chat_id is required when telegram.enabled=true\n"
                    "Set TELEGRAM_CHAT_ID env var or in config.yaml"
                )
        return self


class AppConfig(BaseModel):
    """
    Main application configuration.

    Example config.yaml:

        app:
          poll_seconds: 5
          cooldown_minutes: 30

        storage:
          sqlite_path: "data/moex_agent.sqlite"

        universe:
          tickers:
            - SBER
            - GAZP

        signals:
          p_threshold: 0.35
          horizons:
            - { name: "5m", minutes: 5 }
            - { name: "30m", minutes: 30 }
    """
    # App-level settings
    poll_seconds: int = Field(default=5, ge=1)
    cooldown_minutes: int = Field(default=30, ge=1)
    top_n_anomalies: int = Field(default=10, ge=1)
    max_workers: int = Field(default=20, ge=1, le=100)

    # Sub-configs
    storage: StorageConfig = Field(default_factory=StorageConfig)
    universe: UniverseConfig
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    qwen: QwenConfig = Field(default_factory=QwenConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)

    # Computed properties for backward compatibility
    @property
    def sqlite_path(self) -> Path:
        return self.storage.sqlite_path

    @property
    def tickers(self) -> List[str]:
        return self.universe.tickers

    @property
    def engine(self) -> str:
        return self.universe.engine

    @property
    def market(self) -> str:
        return self.universe.market

    @property
    def board(self) -> str:
        return self.universe.board

    @property
    def horizons(self) -> List[HorizonConfig]:
        return self.signals.horizons

    @property
    def p_threshold(self) -> float:
        return self.signals.p_threshold

    @property
    def price_exit(self) -> Dict[str, Any]:
        return self.signals.price_exit.model_dump()

    @property
    def telegram_enabled(self) -> bool:
        return self.telegram.enabled

    @property
    def telegram_bot_token(self) -> str:
        return self.telegram.bot_token or ""

    @property
    def telegram_chat_id(self) -> str:
        return self.telegram.chat_id or ""

    @property
    def telegram_send_recommendations(self) -> List[str]:
        return self.telegram.send_recommendations

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "AppConfig":
        """
        Load configuration from YAML file with helpful error messages.

        Args:
            path: Path to config.yaml

        Returns:
            Validated AppConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid with clear error message
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n\n"
                "Create config.yaml with at least:\n\n"
                "universe:\n"
                "  tickers:\n"
                "    - SBER\n"
                "    - GAZP\n"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

        # Handle flat 'app' section
        app_section = raw.pop("app", {})

        # Merge app settings into root
        for key in ["poll_seconds", "cooldown_minutes", "top_n_anomalies", "max_workers"]:
            if key in app_section and key not in raw:
                raw[key] = app_section[key]

        try:
            return cls.model_validate(raw)
        except Exception as e:
            # Provide helpful error message
            error_msg = str(e)

            if "universe" in error_msg.lower() and "tickers" in error_msg.lower():
                raise ValueError(
                    "Missing required config: universe.tickers\n\n"
                    "Add to config.yaml:\n\n"
                    "universe:\n"
                    "  tickers:\n"
                    "    - SBER\n"
                    "    - GAZP\n"
                ) from e

            raise ValueError(f"Config validation error: {e}") from e


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """
    Load and validate configuration.

    This is the main entry point for loading config.
    Provides backward compatibility with the old Config class.

    Args:
        path: Path to config.yaml

    Returns:
        Validated AppConfig instance
    """
    return AppConfig.from_yaml(path)


# For backward compatibility
Config = AppConfig


if __name__ == "__main__":
    # Test config loading
    try:
        cfg = load_config()
        print(f"Config loaded successfully!")
        print(f"  Tickers: {len(cfg.tickers)}")
        print(f"  Poll: {cfg.poll_seconds}s")
        print(f"  P threshold: {cfg.p_threshold}")
        print(f"  Telegram: {'enabled' if cfg.telegram_enabled else 'disabled'}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

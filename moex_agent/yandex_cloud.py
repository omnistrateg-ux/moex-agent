"""
Yandex Cloud Integration Module

Provides integration with:
- YandexGPT (Foundation Models API)
- Yandex Object Storage (S3-compatible)
- Yandex Managed PostgreSQL
- Yandex Message Queue
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("moex_agent.yandex_cloud")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class YandexCloudConfig:
    """Yandex Cloud configuration."""
    folder_id: str = ""
    api_key: str = ""
    iam_token: str = ""  # Alternative to API key

    # YandexGPT settings
    gpt_model: str = "yandexgpt"  # yandexgpt, yandexgpt-lite
    gpt_version: str = "latest"  # latest, rc, deprecated
    temperature: float = 0.3
    max_tokens: int = 2000

    # Object Storage
    s3_endpoint: str = "https://storage.yandexcloud.net"
    s3_bucket: str = "moex-agent-storage"
    s3_access_key: str = ""
    s3_secret_key: str = ""

    # PostgreSQL
    pg_host: str = ""
    pg_port: int = 6432
    pg_database: str = "moex_agent"
    pg_user: str = ""
    pg_password: str = ""

    @classmethod
    def from_env(cls) -> "YandexCloudConfig":
        """Load configuration from environment variables."""
        return cls(
            folder_id=os.getenv("YANDEX_CLOUD_FOLDER_ID", ""),
            api_key=os.getenv("YANDEX_API_KEY", ""),
            iam_token=os.getenv("YANDEX_IAM_TOKEN", ""),
            gpt_model=os.getenv("YANDEXGPT_MODEL", "yandexgpt"),
            gpt_version=os.getenv("YANDEXGPT_VERSION", "latest"),
            temperature=float(os.getenv("YANDEXGPT_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("YANDEXGPT_MAX_TOKENS", "2000")),
            s3_endpoint=os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net"),
            s3_bucket=os.getenv("S3_BUCKET", "moex-agent-storage"),
            s3_access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            s3_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            pg_host=os.getenv("POSTGRES_HOST", ""),
            pg_port=int(os.getenv("POSTGRES_PORT", "6432")),
            pg_database=os.getenv("POSTGRES_DB", "moex_agent"),
            pg_user=os.getenv("POSTGRES_USER", ""),
            pg_password=os.getenv("POSTGRES_PASSWORD", ""),
        )


# =============================================================================
# YandexGPT Client
# =============================================================================

class YandexGPTClient:
    """Client for YandexGPT Foundation Models API."""

    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    ASYNC_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"

    def __init__(self, config: Optional[YandexCloudConfig] = None):
        self.config = config or YandexCloudConfig.from_env()
        self._client = httpx.Client(timeout=60.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        headers = {"Content-Type": "application/json"}

        if self.config.api_key:
            headers["Authorization"] = f"Api-Key {self.config.api_key}"
        elif self.config.iam_token:
            headers["Authorization"] = f"Bearer {self.config.iam_token}"
        else:
            raise ValueError("No API key or IAM token configured")

        return headers

    def _get_model_uri(self, model: Optional[str] = None) -> str:
        """Get model URI."""
        model_name = model or self.config.gpt_model
        return f"gpt://{self.config.folder_id}/{model_name}/{self.config.gpt_version}"

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send completion request to YandexGPT.

        Args:
            user_prompt: User message
            system_prompt: System message (role definition)
            model: Model name (yandexgpt, yandexgpt-lite)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Dict with response or error
        """
        if not self.config.folder_id:
            logger.warning("YandexGPT: folder_id not configured, returning stub")
            return self._stub_response(user_prompt)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": user_prompt})

        payload = {
            "modelUri": self._get_model_uri(model),
            "completionOptions": {
                "stream": False,
                "temperature": temperature or self.config.temperature,
                "maxTokens": str(max_tokens or self.config.max_tokens),
            },
            "messages": messages,
        }

        try:
            response = self._client.post(
                self.API_URL,
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Extract text from response
            if "result" in result and "alternatives" in result["result"]:
                text = result["result"]["alternatives"][0]["message"]["text"]
                return {
                    "success": True,
                    "text": text,
                    "model": self._get_model_uri(model),
                    "usage": result["result"].get("usage", {}),
                }

            return {"success": False, "error": "Unexpected response format", "raw": result}

        except httpx.HTTPStatusError as e:
            logger.error(f"YandexGPT HTTP error: {e.response.status_code} - {e.response.text}")
            return {"success": False, "error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"YandexGPT error: {e}")
            return {"success": False, "error": str(e)}

    def analyze_trade(
        self,
        state_json: Dict[str, Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        Analyze trade signal using YandexGPT.

        Args:
            state_json: Current market state and signal
            system_prompt: Analyst system prompt

        Returns:
            Parsed analysis result
        """
        user_prompt = f"""Проанализируй торговый сигнал и верни JSON с решением.

ТЕКУЩЕЕ СОСТОЯНИЕ:
```json
{json.dumps(state_json, ensure_ascii=False, indent=2)}
```

Верни ТОЛЬКО JSON без дополнительного текста."""

        response = self.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model="yandexgpt",  # Use Pro model for trading analysis
            temperature=0.2,  # Lower temperature for more deterministic output
        )

        if not response.get("success"):
            return {
                "provider": "yandexgpt",
                "decision": "NO_OP",
                "error": response.get("error"),
                "verdict": "error",
            }

        # Try to parse JSON from response
        text = response.get("text", "")
        try:
            # Find JSON in response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    "provider": "yandexgpt",
                    "decision": "NO_OP",
                    "error": "No JSON found in response",
                    "raw_text": text,
                    "verdict": "error",
                }
        except json.JSONDecodeError as e:
            return {
                "provider": "yandexgpt",
                "decision": "NO_OP",
                "error": f"JSON parse error: {e}",
                "raw_text": text,
                "verdict": "error",
            }

    def _stub_response(self, prompt: str) -> Dict[str, Any]:
        """Return stub response when not configured."""
        return {
            "success": True,
            "text": json.dumps({
                "provider": "yandexgpt",
                "decision": "NO_OP",
                "verdict": "stub",
                "verdict_reason": "YandexGPT not configured (missing YANDEX_CLOUD_FOLDER_ID)",
            }),
            "model": "stub",
            "usage": {},
        }

    def close(self):
        """Close HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# YandexGPT Analyst (Multi-Role)
# =============================================================================

class YandexGPTAnalyst:
    """
    YandexGPT-based trading analyst with multiple roles.

    Supports:
    - Main Analyst (decision making)
    - Risk Validator
    - Market Context
    - News Scanner
    - Devil's Advocate
    """

    ROLES = {
        "main": {
            "name": "Main Analyst",
            "model": "yandexgpt",
            "description": "Главный аналитик — принятие решений",
        },
        "risk": {
            "name": "Risk Validator",
            "model": "yandexgpt",
            "description": "Риск-менеджер — проверка лимитов",
        },
        "context": {
            "name": "Market Context",
            "model": "yandexgpt",
            "description": "Аналитик контекста — режим рынка",
        },
        "news": {
            "name": "News Scanner",
            "model": "yandexgpt-lite",
            "description": "Сканер новостей — быстрая проверка",
        },
        "advocate": {
            "name": "Devil's Advocate",
            "model": "yandexgpt",
            "description": "Адвокат дьявола — поиск рисков",
        },
    }

    def __init__(self, config: Optional[YandexCloudConfig] = None):
        self.client = YandexGPTClient(config)
        self._prompts: Dict[str, str] = {}

    def load_prompts(self, prompts_dir: str = "."):
        """Load analyst prompts from files."""
        prompt_file = os.path.join(prompts_dir, "YANDEXGPT_ANALYST_PROMPT.md")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                self._prompts["main"] = f.read()
                logger.info("Loaded YandexGPT analyst prompt")

    def analyze(
        self,
        state_json: Dict[str, Any],
        role: str = "main",
    ) -> Dict[str, Any]:
        """
        Analyze trade using specified role.

        Args:
            state_json: Current market state
            role: Analyst role (main, risk, context, news, advocate)

        Returns:
            Analysis result
        """
        if role not in self.ROLES:
            raise ValueError(f"Unknown role: {role}. Available: {list(self.ROLES.keys())}")

        role_config = self.ROLES[role]
        system_prompt = self._prompts.get(role, self._get_default_prompt(role))

        result = self.client.analyze_trade(
            state_json=state_json,
            system_prompt=system_prompt,
        )

        result["role"] = role
        result["role_name"] = role_config["name"]
        return result

    def analyze_all_roles(
        self,
        state_json: Dict[str, Any],
        roles: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze trade using multiple roles.

        Args:
            state_json: Current market state
            roles: List of roles to use (default: all)

        Returns:
            Dict of role -> analysis result
        """
        roles = roles or list(self.ROLES.keys())
        results = {}

        for role in roles:
            try:
                results[role] = self.analyze(state_json, role)
            except Exception as e:
                logger.error(f"Error in {role} analysis: {e}")
                results[role] = {
                    "provider": "yandexgpt",
                    "role": role,
                    "decision": "NO_OP",
                    "error": str(e),
                    "verdict": "error",
                }

        return results

    def _get_default_prompt(self, role: str) -> str:
        """Get default prompt for role."""
        role_config = self.ROLES.get(role, {})
        return f"""Ты — {role_config.get('name', 'Аналитик')} в системе MOEX Trading Agent.
Твоя задача: {role_config.get('description', 'анализ торговых сигналов')}.

Проанализируй входные данные и верни JSON с результатом анализа.
Включи поля: provider, decision, ticker, tier, confidence, reasoning_bullets, verdict, verdict_reason.

decision: LONG | SHORT | NO_TRADE | NO_OP
verdict: support | caution | reject
"""

    def close(self):
        """Close client."""
        self.client.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_yandexgpt_client() -> YandexGPTClient:
    """Create YandexGPT client from environment."""
    return YandexGPTClient(YandexCloudConfig.from_env())


def create_yandexgpt_analyst(prompts_dir: str = ".") -> YandexGPTAnalyst:
    """Create YandexGPT analyst from environment."""
    analyst = YandexGPTAnalyst(YandexCloudConfig.from_env())
    analyst.load_prompts(prompts_dir)
    return analyst


# =============================================================================
# Utility Functions
# =============================================================================

def test_yandexgpt_connection() -> bool:
    """Test YandexGPT API connection."""
    try:
        with create_yandexgpt_client() as client:
            response = client.complete(
                user_prompt="Скажи 'OK' одним словом.",
                system_prompt="Ты тестовый ассистент.",
                max_tokens=10,
            )
            return response.get("success", False)
    except Exception as e:
        logger.error(f"YandexGPT connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    logging.basicConfig(level=logging.INFO)

    print("Testing YandexGPT connection...")
    if test_yandexgpt_connection():
        print("✅ YandexGPT connection OK")
    else:
        print("❌ YandexGPT connection failed")
        print("   Check YANDEX_CLOUD_FOLDER_ID and YANDEX_API_KEY environment variables")

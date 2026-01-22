"""
YandexGPT Integration — LLM для анализа торговых сигналов.

Использование:
1. Получить API ключ: https://console.cloud.yandex.ru/
2. Добавить в Secrets: YANDEX_API_KEY, YANDEX_FOLDER_ID
3. Включить в config.yaml: yandex_gpt.enabled: true
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class YandexGPT:
    """Клиент для YandexGPT API."""

    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        model: str = "yandexgpt-lite",  # или "yandexgpt" для полной версии
    ):
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        self.model = model

        if not self.api_key or not self.folder_id:
            logger.warning("YandexGPT: API key or folder ID not configured")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"YandexGPT initialized: model={model}")

    def _get_model_uri(self) -> str:
        """Получить URI модели."""
        return f"gpt://{self.folder_id}/{self.model}"

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """Отправить запрос к YandexGPT."""
        if not self.enabled:
            return None

        messages = []

        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})

        messages.append({"role": "user", "text": prompt})

        payload = {
            "modelUri": self._get_model_uri(),
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": messages,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]

        except Exception as e:
            logger.error(f"YandexGPT error: {e}")
            return None

    def analyze_signal(self, signal: Dict[str, Any]) -> Optional[str]:
        """Анализ торгового сигнала с помощью LLM."""
        system_prompt = """Ты — профессиональный трейдер и аналитик Московской биржи.
Анализируй торговые сигналы кратко и по делу. Отвечай на русском языке.
Формат ответа: 2-3 предложения с оценкой сигнала и рекомендацией."""

        prompt = f"""Проанализируй торговый сигнал:

Тикер: {signal.get('ticker', 'N/A')}
Направление: {signal.get('direction', 'N/A')}
Горизонт: {signal.get('horizon', 'N/A')}
Вероятность ML: {signal.get('probability', 0):.1%}
Режим рынка: {signal.get('regime', 'N/A')}
RSI: {signal.get('rsi', 'N/A')}
MACD: {signal.get('macd', 'N/A')}

Дай краткую оценку: стоит ли открывать позицию?"""

        return self.complete(prompt, system_prompt=system_prompt)

    def explain_trade(self, trade: Dict[str, Any]) -> Optional[str]:
        """Объяснение закрытой сделки."""
        system_prompt = """Ты — профессиональный трейдер. Объясни результат сделки
кратко и понятно. Отвечай на русском."""

        pnl = trade.get('pnl', 0)
        result = "прибыль" if pnl > 0 else "убыток"

        prompt = f"""Объясни результат сделки:

Тикер: {trade.get('ticker', 'N/A')}
Направление: {trade.get('direction', 'N/A')}
Вход: {trade.get('entry_price', 0):.2f}
Выход: {trade.get('exit_price', 0):.2f}
P&L: {pnl:+.2f} ₽ ({result})
Причина выхода: {trade.get('exit_reason', 'N/A')}
Режим рынка: {trade.get('regime', 'N/A')}

Почему сделка закрылась с таким результатом?"""

        return self.complete(prompt, system_prompt=system_prompt)

    def market_summary(self, tickers_data: list) -> Optional[str]:
        """Краткий обзор рынка."""
        system_prompt = """Ты — аналитик Московской биржи. Дай краткий обзор рынка
на основе данных. 3-4 предложения максимум."""

        # Форматируем данные по тикерам
        summary_lines = []
        for t in tickers_data[:10]:  # Топ 10
            change = t.get('change_pct', 0)
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            summary_lines.append(f"{t['ticker']}: {change:+.1f}% {arrow}")

        prompt = f"""Данные по акциям:

{chr(10).join(summary_lines)}

Дай краткий обзор: что происходит на рынке?"""

        return self.complete(prompt, system_prompt=system_prompt)


# Пример использования
if __name__ == "__main__":
    gpt = YandexGPT()

    if gpt.enabled:
        # Тест анализа сигнала
        test_signal = {
            "ticker": "SBER",
            "direction": "LONG",
            "horizon": "5m",
            "probability": 0.62,
            "regime": "BULL",
            "rsi": 45,
            "macd": 0.5,
        }

        analysis = gpt.analyze_signal(test_signal)
        print(f"Анализ: {analysis}")
    else:
        print("YandexGPT не настроен. Добавьте YANDEX_API_KEY и YANDEX_FOLDER_ID")

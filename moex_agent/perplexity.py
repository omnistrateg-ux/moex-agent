"""
Perplexity AI Integration — Новости и аналитика для торговых решений.

Perplexity отвечает за:
- Поиск и анализ новостей по тикерам
- Оценка news_risk (low/med/high)
- Факт-чекинг событий
- Контекст рынка

Использование:
1. Получить API ключ: https://www.perplexity.ai/settings/api
2. Добавить в Secrets: PERPLEXITY_API_KEY
3. Вызывать через PerplexityAnalyst.analyze()

API Docs: https://docs.perplexity.ai/
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class PerplexityAnalyst:
    """
    Perplexity AI аналитик для новостей и факт-чекинга.

    Роль в оркестраторе: FACT CHECK & NEWS RESEARCH
    - Поиск актуальных новостей по тикеру
    - Оценка влияния новостей на цену
    - Определение news_risk уровня
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    # Модели Perplexity
    MODELS = {
        "sonar": "llama-3.1-sonar-small-128k-online",      # Быстрый, с поиском
        "sonar-large": "llama-3.1-sonar-large-128k-online", # Мощный, с поиском
        "sonar-huge": "llama-3.1-sonar-huge-128k-online",   # Самый мощный
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar",
    ):
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self.model = self.MODELS.get(model, self.MODELS["sonar"])

        if not self.api_key:
            logger.warning("Perplexity: API key not configured (PERPLEXITY_API_KEY)")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Perplexity initialized: model={model}")

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """Вызов Perplexity API."""
        if not self.enabled or not requests:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return None

    def search_news(self, ticker: str, company_name: Optional[str] = None) -> Optional[Dict]:
        """
        Поиск новостей по тикеру.

        Returns:
            {
                "news": [...],
                "news_risk": "low|med|high",
                "summary": "...",
                "key_events": [...]
            }
        """
        if not self.enabled:
            return self._stub_response("search_news")

        # Маппинг тикеров на названия компаний
        TICKER_NAMES = {
            "SBER": "Сбербанк",
            "GAZP": "Газпром",
            "LKOH": "Лукойл",
            "GMKN": "Норникель",
            "NVTK": "Новатэк",
            "ROSN": "Роснефть",
            "YNDX": "Яндекс",
            "MTSS": "МТС",
            "MGNT": "Магнит",
            "VTBR": "ВТБ",
            "MOEX": "Московская биржа",
            "ALRS": "Алроса",
            "CHMF": "Северсталь",
            "NLMK": "НЛМК",
            "PLZL": "Полюс золото",
            "TATN": "Татнефть",
            "SNGS": "Сургутнефтегаз",
            "AFLT": "Аэрофлот",
        }

        company = company_name or TICKER_NAMES.get(ticker, ticker)

        messages = [
            {
                "role": "system",
                "content": """Ты — финансовый аналитик. Найди актуальные новости и оцени их влияние на акции.

Отвечай СТРОГО в JSON формате:
{
  "news": [{"title": "...", "date": "...", "impact": "positive|negative|neutral"}],
  "news_risk": "low|med|high",
  "summary": "краткое резюме за 1-2 предложения",
  "key_events": ["событие 1", "событие 2"],
  "recommendation": "..."
}"""
            },
            {
                "role": "user",
                "content": f"""Найди последние новости по компании {company} (тикер {ticker} на Московской бирже).

Период: последние 24-48 часов.

Оцени:
1. Какие новости вышли?
2. Как они влияют на акции (positive/negative/neutral)?
3. Уровень риска для торговли (low/med/high)?
4. Есть ли предстоящие события (отчётность, дивиденды, сделки)?"""
            }
        ]

        response = self._call_api(messages)

        if not response:
            return self._stub_response("search_news")

        try:
            # Пытаемся распарсить JSON из ответа
            # Perplexity может вернуть текст с JSON внутри
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response

            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            # Если не JSON, возвращаем как текст
            return {
                "news": [],
                "news_risk": "unknown",
                "summary": response[:500],
                "key_events": [],
                "raw_response": response,
            }

    def analyze_for_trade(
        self,
        ticker: str,
        side: str,
        timeframe: str,
        current_price: float,
        ml_signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Полный анализ для торгового решения.

        Формат ответа соответствует схеме оркестратора.
        """
        if not self.enabled:
            return self._stub_response("analyze")

        messages = [
            {
                "role": "system",
                "content": """Ты — Perplexity Research Analyst в торговой системе MOEX.

Твоя роль: FACT CHECK & NEWS RESEARCH
- Проверить фактологию новостей
- Оценить news_risk
- Найти подтверждения или опровержения торговой идеи

Отвечай СТРОГО в JSON по схеме:
{
  "provider": "perplexity",
  "decision": "LONG|SHORT|NO_TRADE|NO_OP",
  "ticker": "...",
  "timeframe": "...",
  "tier": "A|B|C|NONE",
  "market_regime": "trend|range|event|low_liq|unclear",
  "confidence": 0-100,
  "expected_R": null,
  "news_risk": "low|med|high|unknown",
  "news_summary": "...",
  "key_facts": ["...", "..."],
  "upcoming_events": ["...", "..."],
  "invalidations": ["...", "..."],
  "reasoning_bullets": ["...", "..."]
}"""
            },
            {
                "role": "user",
                "content": f"""Проанализируй торговую идею:

Тикер: {ticker}
Направление: {side}
Таймфрейм: {timeframe}
Текущая цена: {current_price}

ML сигнал:
- Вероятность: {ml_signal.get('probability', 'N/A')}
- Режим рынка: {ml_signal.get('regime', 'N/A')}

Задачи:
1. Найди последние новости по {ticker}
2. Оцени news_risk для сделки {side}
3. Есть ли события которые могут сломать сетап?
4. Подтверждаешь ли ты направление {side}?"""
            }
        ]

        response = self._call_api(messages, temperature=0.1)

        if not response:
            return self._stub_response("analyze")

        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response

            result = json.loads(json_str.strip())
            result["provider"] = "perplexity"
            return result

        except json.JSONDecodeError:
            return {
                "provider": "perplexity",
                "decision": "NO_OP",
                "ticker": ticker,
                "timeframe": timeframe,
                "tier": "NONE",
                "news_risk": "unknown",
                "confidence": 0,
                "reasoning_bullets": [
                    "Failed to parse response",
                    response[:200] if response else "No response"
                ],
            }

    def check_market_events(self) -> Dict[str, Any]:
        """
        Проверка общих рыночных событий.

        - Заседания ЦБ
        - Макроэкономические данные
        - Геополитика
        """
        if not self.enabled:
            return self._stub_response("market_events")

        messages = [
            {
                "role": "system",
                "content": """Найди важные события для российского фондового рынка на сегодня и ближайшие дни.

Отвечай в JSON:
{
  "events_today": [{"time": "...", "event": "...", "impact": "high|med|low"}],
  "events_upcoming": [...],
  "market_sentiment": "bullish|bearish|neutral|uncertain",
  "risk_level": "low|med|high",
  "key_factors": ["...", "..."]
}"""
            },
            {
                "role": "user",
                "content": f"""Дата: {datetime.now().strftime('%Y-%m-%d')}

Найди:
1. Заседания ЦБ РФ, ФРС
2. Выход макроэкономических данных
3. Корпоративные события (отчётности, дивиденды)
4. Геополитические факторы

Как это влияет на торговлю сегодня?"""
            }
        ]

        response = self._call_api(messages)

        if not response:
            return self._stub_response("market_events")

        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response
            return json.loads(json_str.strip())
        except:
            return {
                "events_today": [],
                "events_upcoming": [],
                "market_sentiment": "uncertain",
                "risk_level": "unknown",
                "raw_response": response,
            }

    def _stub_response(self, method: str) -> Dict[str, Any]:
        """Заглушка когда API недоступен."""
        return {
            "provider": "perplexity",
            "decision": "NO_OP",
            "news_risk": "unknown",
            "confidence": 0,
            "reasoning_bullets": [
                f"Perplexity API not configured",
                f"Method: {method}",
                "Add PERPLEXITY_API_KEY to secrets"
            ],
            "stub": True,
        }


# Синглтон для использования в системе
_analyst: Optional[PerplexityAnalyst] = None

def get_perplexity_analyst() -> PerplexityAnalyst:
    """Получить инстанс Perplexity аналитика."""
    global _analyst
    if _analyst is None:
        _analyst = PerplexityAnalyst()
    return _analyst


def analyze_ticker_news(ticker: str) -> Dict[str, Any]:
    """Быстрый анализ новостей по тикеру."""
    analyst = get_perplexity_analyst()
    return analyst.search_news(ticker)


def get_market_context() -> Dict[str, Any]:
    """Получить контекст рынка."""
    analyst = get_perplexity_analyst()
    return analyst.check_market_events()


# Тестирование
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyst = PerplexityAnalyst()

    if analyst.enabled:
        print("Testing Perplexity API...")

        # Тест поиска новостей
        news = analyst.search_news("SBER")
        print(f"\nNews for SBER:")
        print(json.dumps(news, indent=2, ensure_ascii=False))

        # Тест анализа для сделки
        analysis = analyst.analyze_for_trade(
            ticker="SBER",
            side="LONG",
            timeframe="5m",
            current_price=250.50,
            ml_signal={"probability": 0.62, "regime": "BULL"}
        )
        print(f"\nTrade analysis:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        print("Perplexity not configured. Add PERPLEXITY_API_KEY to environment.")
        print("\nStub response:")
        print(json.dumps(analyst._stub_response("test"), indent=2))

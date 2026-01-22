from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("moex_agent.qwen")

# ─────────────────────────────────────────────────────────────
# Конфигурация тикеров по ликвидности
# ─────────────────────────────────────────────────────────────
# Tier 1 — Blue chips (высшая ликвидность, можно агрессивнее)
LIQUID_TICKERS = {
    "SBER", "SBERP", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "SIBN",
    "T", "YDEX", "VTBR", "PLZL"
}

# Tier 2 — Средняя ликвидность (стандартный подход)
MEDIUM_TICKERS = {
    "TATN", "TATNP", "SNGS", "SNGSP", "OZON", "X5", "MTSS", "CHMF",
    "NLMK", "MAGN", "ALRS", "MOEX", "AFLT", "POSI", "IRAO", "HYDR",
    "FEES", "RUAL", "MGNT", "SMLT", "SFIN", "SOFL", "SPBE", "VKCO",
    "PHOR", "TRNFP", "FLOT"
}

# Tier 3 — Низкая ликвидность (консервативно, без шортов)
# Все остальные: PIKK, CBOM, HEAD, LENT, OKEY, ENPG, RSTI, ...

# ─────────────────────────────────────────────────────────────
# System prompt для анализа торговых сигналов
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ты аналитик торговых сигналов для российского рынка акций (MOEX).

## Твоя задача
Проанализировать входящий сигнал и вернуть структурированный JSON с оценкой.

## Входные данные
- ticker: тикер акции
- direction: LONG (покупка) или SHORT (продажа)
- horizon: горизонт сделки (5m, 10m, 30m, 1h, 1d, 1w)
- p: вероятность успеха от ML-модели (0.0 - 1.0)
- signal_type: "price-exit" или "time-exit"
- entry/take/stop: уровни цен
- anomaly: метрики аномалии
  - score: сила аномалии
  - z_ret_5m: z-score доходности
  - z_vol_5m: z-score объёма
  - volume_spike: всплеск объёма (1.0 = норма, 2.0 = в 2 раза выше)
  - spread_bps: спред в б.п.
- market_context: время торгов, ликвидность тикера

## Правила анализа

### Когда ОТКЛОНИТЬ (skip: true):
1. p < 0.30 — низкая уверенность (модель калибрована, max ≈ 0.60)
2. |z_ret| < 0.5 и |z_vol| < 0.5 — нет аномалии
3. Короткий горизонт + широкий спред — высокие издержки
4. Первые 15 мин торгов + низкий score — утренний шум
5. SHORT + низколиквидный тикер — сложно шортить

### Уровни риска (calibrated model):
- LOW: p > 0.45, |z_ret| > 1.5, spread < 20, volume_spike > 1.5
- MEDIUM: p > 0.38, |z_ret| > 1.0, spread < 35
- HIGH: остальные

## Формат ответа (ТОЛЬКО JSON)
{
  "skip": false,
  "risk_level": "LOW|MEDIUM|HIGH",
  "confidence": 0.85,
  "reasoning": "Обоснование (1-2 предложения)",
  "risk_note": "Предупреждение для Telegram",
  "recommendation": "STRONG_BUY|BUY|WEAK_BUY|STRONG_SELL|SELL|WEAK_SELL|SKIP"
}
"""


@dataclass
class QwenAnalysis:
    """Результат анализа сигнала."""
    skip: bool
    risk_level: str
    confidence: float
    reasoning: str
    risk_note: str
    recommendation: str
    skip_reason: Optional[str] = None
    raw_response: Optional[Dict] = None


def _get_market_context() -> Dict[str, Any]:
    """Определяет контекст рынка (время торгов)."""
    now = datetime.now(timezone.utc)
    moscow_hour = (now.hour + 3) % 24  # UTC+3

    # MOEX торги: 10:00-18:50 MSK
    is_trading = 10 <= moscow_hour < 19

    # Первые 15 минут
    is_opening = moscow_hour == 10 and now.minute < 15

    # Последние 30 минут
    is_closing = moscow_hour == 18 and now.minute >= 20

    return {
        "moscow_hour": moscow_hour,
        "is_trading": is_trading,
        "is_opening": is_opening,
        "is_closing": is_closing,
        "day_of_week": now.weekday(),  # 0=Mon, 4=Fri
    }


def _get_ticker_liquidity(ticker: str) -> str:
    """Возвращает уровень ликвидности тикера."""
    if ticker in LIQUID_TICKERS:
        return "HIGH"
    if ticker in MEDIUM_TICKERS:
        return "MEDIUM"
    return "LOW"


def _call_ollama(
    ollama_url: str,
    model: str,
    payload: Dict[str, Any],
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Вызов Ollama API."""
    url = ollama_url.rstrip("/") + "/api/chat"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]

    r = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "").strip()

    # Извлечь JSON из markdown блока
    if content.startswith("```"):
        lines = content.split("\n")
        json_lines = [l for l in lines if not l.startswith("```")]
        content = "\n".join(json_lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Qwen response: {content[:200]}")
        return {"skip": False, "raw_text": content}


def _rule_based_analysis(payload: Dict[str, Any]) -> QwenAnalysis:
    """
    Правила анализа без LLM.
    Возвращает QwenAnalysis с skip=True если сигнал плохой,
    или с нормальной оценкой если сигнал приемлемый.
    """
    anomaly = payload.get("anomaly", {})
    p = payload.get("p", 0)
    direction = payload.get("direction", "LONG")
    horizon = payload.get("horizon", "")
    ticker = payload.get("ticker", "")

    z_ret = anomaly.get("z_ret_5m", 0)
    z_vol = anomaly.get("z_vol_5m", 0)
    spread = anomaly.get("spread_bps") or 0
    volume_spike = anomaly.get("volume_spike", 1.0)

    market = payload.get("market_context", {})
    is_opening = market.get("is_opening", False)
    liquidity = market.get("ticker_liquidity", "MEDIUM")

    # ─────────────────────────────────────────────────────
    # SKIP RULES
    # ─────────────────────────────────────────────────────

    # Rule 1: Низкая вероятность (calibrated model max ≈ 0.60)
    if p < 0.30:
        return QwenAnalysis(
            skip=True,
            risk_level="HIGH",
            confidence=p,
            reasoning="Вероятность модели ниже порога",
            risk_note="",
            recommendation="SKIP",
            skip_reason=f"p={p:.2f} < 0.30",
        )

    # Rule 2: Нет реальной аномалии
    if abs(z_ret) < 0.5 and abs(z_vol) < 0.5:
        return QwenAnalysis(
            skip=True,
            risk_level="HIGH",
            confidence=0.3,
            reasoning="Движение в пределах нормы",
            risk_note="",
            recommendation="SKIP",
            skip_reason=f"|z_ret|={abs(z_ret):.1f}, |z_vol|={abs(z_vol):.1f} < 0.5",
        )

    # Rule 3: Короткий горизонт + широкий спред
    if horizon in ("5m", "10m") and spread > 25:
        return QwenAnalysis(
            skip=True,
            risk_level="HIGH",
            confidence=0.5,
            reasoning="Спред съест прибыль на коротком горизонте",
            risk_note="",
            recommendation="SKIP",
            skip_reason=f"spread={spread:.0f}bps для H={horizon}",
        )

    # Rule 4: Открытие торгов + слабый сигнал
    if is_opening and abs(z_ret) < 1.2:
        return QwenAnalysis(
            skip=True,
            risk_level="HIGH",
            confidence=0.4,
            reasoning="Утренняя волатильность, слабый сигнал",
            risk_note="",
            recommendation="SKIP",
            skip_reason="opening + weak signal",
        )

    # Rule 5: SHORT + низкая ликвидность
    if direction == "SHORT" and liquidity == "LOW":
        return QwenAnalysis(
            skip=True,
            risk_level="HIGH",
            confidence=0.4,
            reasoning="Шорт низколиквидной бумаги рискован",
            risk_note="",
            recommendation="SKIP",
            skip_reason="SHORT + low liquidity",
        )

    # ─────────────────────────────────────────────────────
    # SCORING (если не SKIP)
    # ─────────────────────────────────────────────────────

    abs_z_ret = abs(z_ret)

    # Определяем risk_level и recommendation (calibrated: max p ≈ 0.60)
    if p > 0.45 and abs_z_ret > 1.5 and spread < 20 and volume_spike > 1.3:
        risk_level = "LOW"
        rec_prefix = "STRONG_"
    elif p > 0.38 and abs_z_ret > 1.0 and spread < 35:
        risk_level = "MEDIUM"
        rec_prefix = ""
    else:
        risk_level = "HIGH"
        rec_prefix = "WEAK_"

    # Recommendation based on direction
    if direction == "SHORT":
        recommendation = f"{rec_prefix}SELL" if rec_prefix else "SELL"
    else:
        recommendation = f"{rec_prefix}BUY" if rec_prefix else "BUY"

    # Risk note
    risk_notes = []
    if spread > 30:
        risk_notes.append(f"⚠️ Широкий спред ({spread:.0f} bps)")
    if volume_spike < 1.0:
        risk_notes.append("⚠️ Объём ниже среднего")
    if is_opening:
        risk_notes.append("⚠️ Открытие торгов")
    if direction == "SHORT":
        risk_notes.append("📉 SHORT позиция")

    # Reasoning
    vol_note = f"vol_spike={volume_spike:.1f}x" if volume_spike > 1.2 else ""
    reasoning = f"z_ret={z_ret:.1f}, z_vol={z_vol:.1f}, spread={spread:.0f}bps {vol_note}".strip()

    return QwenAnalysis(
        skip=False,
        risk_level=risk_level,
        confidence=p,
        reasoning=reasoning,
        risk_note=" | ".join(risk_notes) if risk_notes else "",
        recommendation=recommendation,
    )


def analyze_signal(
    ollama_url: str,
    model: str,
    payload: Dict[str, Any],
    max_tokens: int = 500,
    temperature: float = 0.3,
    use_rules_only: bool = False,
) -> QwenAnalysis:
    """
    Анализирует торговый сигнал.

    Добавляет market_context к payload перед анализом.
    Сначала проверяет правилами, затем (опционально) через LLM.
    """
    # Добавляем контекст рынка
    market_context = _get_market_context()
    market_context["ticker_liquidity"] = _get_ticker_liquidity(payload.get("ticker", ""))
    payload["market_context"] = market_context

    # Rule-based анализ
    rule_result = _rule_based_analysis(payload)

    # Если правила сказали SKIP — сразу возвращаем
    if rule_result.skip:
        logger.debug(f"Signal rejected by rules: {rule_result.skip_reason}")
        return rule_result

    # Если только правила — возвращаем rule_result
    if use_rules_only:
        return rule_result

    # Вызов LLM для дополнительного анализа
    try:
        response = _call_ollama(ollama_url, model, payload, max_tokens, temperature)
    except Exception as e:
        logger.warning(f"Ollama failed, using rules: {e}")
        return rule_result

    # Парсинг LLM ответа
    skip = response.get("skip", False)

    return QwenAnalysis(
        skip=skip,
        risk_level=response.get("risk_level", rule_result.risk_level),
        confidence=float(response.get("confidence", payload.get("p", 0.5))),
        reasoning=response.get("reasoning", rule_result.reasoning),
        risk_note=response.get("risk_note", rule_result.risk_note),
        recommendation=response.get("recommendation", rule_result.recommendation),
        skip_reason=response.get("skip_reason") if skip else None,
        raw_response=response,
    )


def format_telegram_message(
    ticker: str,
    horizon: str,
    p: float,
    analysis: QwenAnalysis,
    direction: str = "LONG",
    entry: Optional[float] = None,
    take: Optional[float] = None,
    stop: Optional[float] = None,
    anomaly_score: float = 0,
    volume_spike: float = 1.0,
) -> str:
    """Форматирует сообщение для Telegram."""

    # Эмодзи по рекомендации
    emoji_map = {
        "STRONG_BUY": "🟢🟢",
        "BUY": "🟢",
        "WEAK_BUY": "🟡",
        "STRONG_SELL": "🔴🔴",
        "SELL": "🔴",
        "WEAK_SELL": "🟠",
        "SKIP": "⚫",
    }
    emoji = emoji_map.get(analysis.recommendation, "⚪")

    # Direction emoji
    dir_emoji = "📈" if direction == "LONG" else "📉"

    # Risk
    risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🔴"}.get(analysis.risk_level, "")

    lines = [
        f"{emoji} **{ticker}** {dir_emoji} {direction} | {horizon}",
        f"📊 p={p:.0%} | score={anomaly_score:.1f} | vol={volume_spike:.1f}x",
    ]

    if entry and take and stop:
        lines.append(f"💰 Entry: {entry:.2f} → Take: {take:.2f} | Stop: {stop:.2f}")

    lines.append(f"{risk_emoji} Risk: {analysis.risk_level} | {analysis.recommendation}")

    if analysis.reasoning:
        lines.append(f"💡 {analysis.reasoning}")

    if analysis.risk_note:
        lines.append(analysis.risk_note)

    return "\n".join(lines)


# Legacy API
def explain_with_qwen(
    ollama_url: str,
    model: str,
    payload: Dict[str, Any],
    max_tokens: int = 350,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Legacy функция для совместимости."""
    analysis = analyze_signal(
        ollama_url=ollama_url,
        model=model,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "skip": analysis.skip,
        "skip_reason": analysis.skip_reason,
        "risk_level": analysis.risk_level,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "risk_note": analysis.risk_note,
        "recommendation": analysis.recommendation,
    }

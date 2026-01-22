from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("moex_agent.telegram")


def send_telegram(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: Optional[str] = "Markdown",
    disable_preview: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> bool:
    """
    Отправляет сообщение в Telegram с retry-логикой.

    Args:
        bot_token: Токен бота от @BotFather
        chat_id: ID чата или канала
        text: Текст сообщения
        parse_mode: "Markdown", "HTML" или None
        disable_preview: Отключить превью ссылок
        max_retries: Максимум попыток
        retry_delay: Задержка между попытками (секунды)

    Returns:
        True если успешно, False если все попытки неудачны
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview,
    }

    if parse_mode:
        payload["parse_mode"] = parse_mode

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=30)

            if r.status_code == 429:
                # Rate limit - wait and retry
                retry_after = r.json().get("parameters", {}).get("retry_after", 5)
                logger.warning(f"Telegram rate limit, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            if r.status_code == 400 and parse_mode:
                # Markdown parsing error - retry without parse_mode
                logger.warning("Markdown parse error, retrying as plain text")
                payload.pop("parse_mode", None)
                r = requests.post(url, json=payload, timeout=30)

            r.raise_for_status()
            logger.debug(f"Telegram message sent to {chat_id}")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Telegram timeout (attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram error: {e} (attempt {attempt + 1}/{max_retries})")

        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False


def send_signal_alert(
    bot_token: str,
    chat_id: str,
    ticker: str,
    direction: str,
    horizon: str,
    p: float,
    score: float,
    recommendation: str,
    risk_level: str,
    reasoning: str,
    entry: Optional[float] = None,
    take: Optional[float] = None,
    stop: Optional[float] = None,
    volume_spike: float = 1.0,
    risk_note: str = "",
) -> bool:
    """
    Отправляет форматированный торговый сигнал в Telegram.
    """
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
    emoji = emoji_map.get(recommendation, "⚪")

    # Direction emoji
    dir_emoji = "📈" if direction == "LONG" else "📉"

    # Risk emoji
    risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🔴"}.get(risk_level, "")

    lines = [
        f"{emoji} *{ticker}* {dir_emoji} {direction} | {horizon}",
        f"📊 p={p:.0%} | score={score:.1f} | vol={volume_spike:.1f}x",
    ]

    if entry and take and stop:
        lines.append(f"💰 Entry: {entry:.2f} → Take: {take:.2f} | Stop: {stop:.2f}")

    lines.append(f"{risk_emoji} Risk: {risk_level} | {recommendation}")

    if reasoning:
        lines.append(f"💡 _{reasoning}_")

    if risk_note:
        lines.append(risk_note)

    text = "\n".join(lines)
    return send_telegram(bot_token, chat_id, text)

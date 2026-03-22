"""
Telegram Notifier — Optional push notifications for trade signals and closures.

Setup:
  1. Create a bot via @BotFather on Telegram → get BOT_TOKEN
  2. Get your chat ID: message your bot, then visit:
     https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
  3. Add to your .env file:
       TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       TELEGRAM_CHAT_ID=987654321

If env vars are missing, all notifications are silently skipped.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("telegram_notifier")

_BOT_TOKEN: Optional[str] = None
_CHAT_ID: Optional[str] = None
_enabled: bool = False


def _load_config():
    """Load Telegram credentials from environment (once, lazily)."""
    global _BOT_TOKEN, _CHAT_ID, _enabled
    if _enabled:
        return

    # Try loading .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    _BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    _CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    _enabled = bool(_BOT_TOKEN and _CHAT_ID)

    if _enabled:
        logger.info("Telegram notifications enabled (chat_id=%s)", _CHAT_ID)
    else:
        logger.debug("Telegram not configured — notifications disabled")


def send(message: str) -> bool:
    """
    Send a Telegram message.

    Parameters
    ----------
    message : str
        Markdown-formatted message text.

    Returns
    -------
    bool
        True if sent successfully, False otherwise.
    """
    _load_config()

    if not _enabled:
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": _CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }, timeout=5)
        if resp.ok:
            logger.debug("Telegram sent OK")
            return True
        else:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.warning("Telegram send error: %s", e)
        return False


def send_signal(symbol: str, resolution: str, signal) -> bool:
    """Send a formatted new-signal notification."""
    _load_config()
    if not _enabled:
        return False

    direction_emoji = "📈" if signal.direction.value == "LONG" else "📉"
    conf = getattr(signal, 'confidence', 0)
    conf_stars = "⭐" * max(1, round(conf * 5))

    msg = (
        f"{direction_emoji} *NEW SIGNAL — {symbol} {resolution}*\n"
        f"Direction : `{signal.direction.value}`\n"
        f"Entry     : `{signal.entry_price:.2f}`\n"
        f"Stop Loss : `{signal.stop_loss:.2f}`\n"
        f"Take Profit: `{signal.take_profit:.2f}`\n"
        f"R:R       : `{signal.rr_ratio:.1f}x`\n"
        f"Confidence: {conf_stars} `({conf:.2f})`\n"
        f"Session   : `{signal.session}`\n"
        f"Entry Type: `{signal.entry_type}`\n"
        f"HTF Bias  : `{signal.htf_bias}`"
    )
    return send(msg)


def send_trade_close(symbol: str, resolution: str, outcome: str,
                     pnl: float, balance: float, signal) -> bool:
    """Send a formatted trade-closed notification."""
    _load_config()
    if not _enabled:
        return False

    if outcome == "TAKE PROFIT":
        emoji = "✅"
    elif "STOP" in outcome or "LOSS" in outcome:
        emoji = "❌"
    elif outcome == "BREAK-EVEN":
        emoji = "➡️"
    else:
        emoji = "⏰"

    pnl_sign = "+" if pnl >= 0 else ""
    msg = (
        f"{emoji} *TRADE CLOSED — {symbol} {resolution}*\n"
        f"Outcome   : `{outcome}`\n"
        f"Direction : `{signal.direction.value}`\n"
        f"Entry     : `{signal.entry_price:.2f}`\n"
        f"P&L       : `{pnl_sign}${pnl:.2f}`\n"
        f"Balance   : `${balance:.2f}`"
    )
    return send(msg)


def send_daily_summary(symbol: str, resolution: str, stats: dict) -> bool:
    """Send end-of-day performance summary."""
    _load_config()
    if not _enabled:
        return False

    trades = stats.get("trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pnl = stats.get("daily_pnl", 0)
    balance = stats.get("balance", 0)
    win_rate = (wins / trades * 100) if trades > 0 else 0

    pnl_emoji = "📈" if pnl >= 0 else "📉"
    pnl_sign = "+" if pnl >= 0 else ""

    msg = (
        f"{pnl_emoji} *Daily Summary — {symbol} {resolution}*\n"
        f"Trades : `{trades}` (W:{wins} L:{losses}) — WR: `{win_rate:.0f}%`\n"
        f"P&L    : `{pnl_sign}${pnl:.2f}`\n"
        f"Balance: `${balance:.2f}`"
    )
    return send(msg)

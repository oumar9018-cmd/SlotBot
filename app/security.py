"""Security: webhook secret verification + Mini App initData validation."""
import hashlib
import hmac
from urllib.parse import parse_qsl

from . import config


def verify_webhook_secret(request_headers) -> bool:
    """Telegram sends X-Telegram-Bot-Api-Secret-Token on webhook calls."""
    if not config.WEBHOOK_SECRET:
        return False
    sent = request_headers.get("x-telegram-bot-api-secret-token", "")
    return hmac.compare_digest(sent, config.WEBHOOK_SECRET)


def validate_init_data(init_data: str) -> dict | None:
    """Validate Telegram Mini App initData. Returns parsed user dict or None.

    Algorithm (per Telegram docs): sort all key=<value> pairs except `hash`
    joined by newline -> data_check_string. secret_key = HMAC_SHA256(bot_token,
    key="WebAppData"). Compare HMAC_SHA256(data_check_string, secret_key)
    hex digest with `hash` using compare_digest.
    """
    try:
        pairs = dict(parse_qsl(init_data, keep_blank_values=True))
        received_hash = pairs.pop("hash", None)
        if not received_hash:
            return None
        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
        secret_key = hmac.new(
            b"WebAppData", config.BOT_TOKEN.encode(), hashlib.sha256
        ).digest()
        calc = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(calc, received_hash):
            return None
        import json
        user = json.loads(pairs.get("user", "{}"))
        return user if user.get("id") else None
    except Exception:
        return None

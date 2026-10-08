"""Thin async client for the Telegram Bot API (httpx, no framework)."""
import logging
import os

import httpx

from . import config

log = logging.getLogger("slotbot.telegram")


def _client(timeout: int = 15) -> httpx.AsyncClient:
    # trust_env=False: httpx 0.28 fails parsing IPv6 entries in no_proxy;
    # read the proxy URL explicitly instead. None -> direct connection.
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    # Egress proxy does TLS interception: use its CA bundle when present.
    ca_bundle = (os.environ.get("SSL_CERT_FILE")
                 or os.environ.get("REQUESTS_CA_BUNDLE"))
    verify = ca_bundle or True
    return httpx.AsyncClient(timeout=timeout, trust_env=False,
                             proxy=proxy, verify=verify)


async def api(method: str, **params):
    url = f"{config.TELEGRAM_API}/bot{config.BOT_TOKEN}/{method}"
    # Never log the token
    safe = {k: ("<redacted>" if "token" in k.lower() else v) for k, v in params.items()}
    try:
        async with _client() as client:
            r = await client.post(url, json=params)
            data = r.json()
    except Exception as e:
        log.error("Telegram API %s failed: %s params=%s", method, e, safe)
        return {"ok": False, "error": str(e)}
    if not data.get("ok"):
        log.warning("Telegram API %s error: %s params=%s",
                    method, data.get("description"), safe)
    return data


async def send_message(chat_id: int, text: str, reply_markup: dict | None = None,
                       parse_mode: str | None = None):
    params = {"chat_id": chat_id, "text": text}
    if reply_markup:
        params["reply_markup"] = reply_markup
    if parse_mode:
        params["parse_mode"] = parse_mode
    return await api("sendMessage", **params)


async def answer_callback(callback_query_id: str, text: str | None = None,
                          show_alert: bool = False):
    params = {"callback_query_id": callback_query_id, "show_alert": show_alert}
    if text:
        params["text"] = text
    return await api("answerCallbackQuery", **params)


async def edit_message(chat_id: int, message_id: int, text: str,
                       reply_markup: dict | None = None):
    params = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup is not None:
        params["reply_markup"] = reply_markup
    return await api("editMessageText", **params)


async def send_invoice(chat_id: int, title: str, description: str, payload: str,
                       amount_stars: int):
    """Stars invoice for a digital service. provider_token empty per docs."""
    return await api(
        "sendInvoice",
        chat_id=chat_id,
        title=title,
        description=description,
        payload=payload,
        currency="XTR",
        prices=[{"label": title, "amount": amount_stars}],
        provider_token="",
    )


async def answer_precheckout(pre_checkout_query_id: str, ok: bool,
                             error_message: str | None = None):
    params = {"pre_checkout_query_id": pre_checkout_query_id, "ok": ok}
    if error_message:
        params["error_message"] = error_message
    return await api("answerPreCheckoutQuery", **params)


async def refund_stars(user_id: int, telegram_charge_id: str):
    return await api(
        "refundStarPayment",
        user_id=user_id,
        telegram_payment_charge_id=telegram_charge_id,
    )


async def set_webhook():
    url = f"{config.BASE_URL}/webhook"
    return await api("setWebhook", url=url, secret_token=config.WEBHOOK_SECRET,
                     allowed_updates=["message", "callback_query",
                                      "pre_checkout_query"])


async def delete_webhook():
    return await api("deleteWebhook", drop_pending_updates=False)


async def get_updates(offset: int | None = None, timeout: int = 30):
    params = {"timeout": timeout,
              "allowed_updates": ["message", "callback_query", "pre_checkout_query"]}
    if offset is not None:
        params["offset"] = offset
    url = f"{config.TELEGRAM_API}/bot{config.BOT_TOKEN}/getUpdates"
    try:
        async with _client(timeout + 10) as client:
            r = await client.post(url, json=params)
            return r.json()
    except Exception as e:
        log.error("getUpdates failed: %s", e)
        return {"ok": False, "result": []}


def inline_keyboard(buttons: list[list[dict]]) -> dict:
    return {"inline_keyboard": buttons}


def webapp_button(text: str, url: str) -> dict:
    return {"text": text, "web_app": {"url": url}}

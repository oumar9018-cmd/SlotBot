"""
Strict English Voice Test — Backend (FastAPI)
AI-powered strict English speaking test. Stern examiner persona via OpenAI Realtime API.

Endpoints:
  POST /session/start              — mint ephemeral token, create session record
  POST /session/end                — score transcript, save, return verdict
  POST /payment/stars-invoice      — create Telegram Stars invoice link (750 Stars)
  POST /payment/telegram-webhook   — pre_checkout approval + successful_payment handler
  GET  /affiliate/stats/{code}
  WS   /ws/voice/{session_id}      (realtime_proxy)
"""
import os
import time
import logging
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import firestore_db as db
import scoring
from realtime_proxy import voice_proxy, PERSONA_INSTRUCTIONS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("englishtest")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")  # optional

# Pricing: full 15-min speaking test + report — Telegram Stars (XTR), 1-click native
FULL_TEST_PRICE_STARS = 750

# Time limits (seconds)
FREE_TRIAL_MAX_SEC = 120   # 2 min hard cutoff
PAID_MAX_SEC = 900         # 15 min

app = FastAPI(title="Strict English Voice Test API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Telegram WebApp; tighten in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------

class SessionStartReq(BaseModel):
    user_id: str = Field(..., description="Telegram user id (string)")
    test_type: str = Field(..., pattern="^(academic|business|general)$")
    plan: str = Field(..., pattern="^(free_trial|paid)$")
    affiliate_code: str | None = None
    background_summary: str | None = Field(None, max_length=2000,
        description="Optional applicant background for sharper examiner questions")


class SessionStartResp(BaseModel):
    session_id: str
    ephemeral_token: str
    expires_in_sec: int
    max_duration_sec: int
    ws_url_path: str


class SessionEndReq(BaseModel):
    session_id: str


class StarsInvoiceReq(BaseModel):
    user_id: str
    affiliate_code: str | None = None


# ---------- Helpers ----------

TEST_TYPE_HINTS = {
    "academic": "Academic speaking test — probe study plans, academic background, and future goals.",
    "business": "Business speaking test — probe work experience, professional background, and career plans.",
    "general": "General speaking test — probe personal background and everyday topics in depth.",
}


async def tg_api(method: str, payload: dict) -> dict:
    """Call Telegram Bot API. Raises 502 on failure."""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}",
            json=payload,
        )
    if r.status_code != 200:
        log.error("Telegram API %s failed: %s %s", method, r.status_code, r.text[:300])
        raise HTTPException(502, "Telegram API error")
    data = r.json()
    if not data.get("ok"):
        log.error("Telegram API %s not ok: %s", method, str(data)[:300])
        raise HTTPException(502, "Telegram API error")
    return data


async def mint_realtime_token(test_type: str, background_summary: str | None) -> dict:
    """Create an OpenAI Realtime session, return ephemeral client secret.
    The API key NEVER leaves the backend."""
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not configured")

    instructions = PERSONA_INSTRUCTIONS
    instructions += "\n\nTEST FOCUS: " + TEST_TYPE_HINTS.get(test_type, TEST_TYPE_HINTS["general"])
    if background_summary:
        instructions += (
            "\n\nAPPLICANT BACKGROUND FORM SUMMARY (use for contradiction traps):\n"
            + background_summary.strip()
        )

    payload = {
        "model": OPENAI_REALTIME_MODEL,
        "voice": "onyx",  # deep, stern — fits the examiner persona
        "instructions": instructions,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": {"type": "server_vad"},
        "max_response_output_tokens": 300,  # short, clipped examiner replies
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if r.status_code != 200:
        log.error("Realtime session mint failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(502, "Voice engine unavailable, try again")
    data = r.json()
    secret = data.get("client_secret", {})
    return {"token": secret.get("value", ""), "expires_in": secret.get("expires_in", 60)}


# ---------- Telegram Auth (HMAC-SHA256 verification) ----------

import hmac
import hashlib
import json as json_lib
from urllib.parse import parse_qsl, unquote


def verify_telegram_init_data(init_data: str, bot_token: str, max_age_sec: int = 86400) -> dict:
    """
    Verify Telegram Mini App initData using HMAC-SHA256.
    Returns the parsed user dict on success, raises HTTPException on failure.

    Telegram docs: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
    """
    if not init_data or not bot_token:
        raise HTTPException(401, "Missing initData or bot token")

    try:
        pairs = dict(parse_qsl(init_data, strict_parsing=True))
    except Exception:
        raise HTTPException(401, "Malformed initData")

    recv_hash = pairs.pop("hash", None)
    pairs.pop("signature", None)  # not part of HMAC check (Bot API 7.0+)
    if not recv_hash:
        raise HTTPException(401, "Missing hash in initData")

    # Data-check string: sorted key=value pairs joined by \n
    data_check = "\n".join(f"{k}={pairs[k]}" for k in sorted(pairs))

    # Secret key: HMAC_SHA256(key="WebAppData", msg=bot_token)
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    # Verify: HMAC_SHA256(key=secret, msg=data_check)
    calc_hash = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calc_hash, recv_hash):
        log.warning("Telegram initData HMAC verification FAILED")
        raise HTTPException(401, "Invalid initData signature")

    # Check auth_date freshness (default: 24h)
    try:
        auth_date = int(pairs.get("auth_date", 0))
    except ValueError:
        raise HTTPException(401, "Invalid auth_date")
    if auth_date + max_age_sec < time.time():
        raise HTTPException(401, "initData expired — reopen the Mini App")

    # Parse user JSON
    user_raw = pairs.get("user", "")
    try:
        user = json_lib.loads(unquote(user_raw)) if user_raw else {}
    except Exception:
        raise HTTPException(401, "Invalid user data in initData")

    if not user.get("id"):
        raise HTTPException(401, "No user id in initData")

    log.info("Telegram auth OK for user_id=%s", user["id"])
    return user


class TelegramAuthReq(BaseModel):
    initData: str = Field(..., description="Raw Telegram WebApp initData string")
    affiliate_code: str | None = None


class TelegramAuthResp(BaseModel):
    telegram_id: str
    first_name: str
    username: str
    is_new: bool
    free_trial_used: bool
    paid: bool


# ---------- Routes ----------

@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.post("/api/auth/telegram", response_model=TelegramAuthResp)
async def telegram_auth(req: TelegramAuthReq):
    """
    Step 2-4 of Telegram auto-login:
    1. Verify initData HMAC-SHA256 signature using bot token (proves user is real)
    2. Check Firestore for existing user
    3. Create new user doc if first time ("Free 2-minute test available")
    4. Return verified user profile
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")

    # Step 2-3: Cryptographic verification
    tg_user = verify_telegram_init_data(req.initData, TELEGRAM_BOT_TOKEN)

    telegram_id = str(tg_user["id"])

    # Step 4: Database check (Firestore)
    user = db.get_user(telegram_id)
    is_new = False
    if not user:
        user = db.create_user(
            telegram_id,
            first_name=tg_user.get("first_name", ""),
            username=tg_user.get("username", ""),
            affiliate_code=req.affiliate_code,
        )
        is_new = True
        log.info("New Telegram user created: %s", telegram_id)
    else:
        # Update profile fields (name/username may have changed)
        db.update_user(telegram_id, {
            "first_name": tg_user.get("first_name", ""),
            "username": tg_user.get("username", ""),
        })

    return TelegramAuthResp(
        telegram_id=telegram_id,
        first_name=tg_user.get("first_name", ""),
        username=tg_user.get("username", ""),
        is_new=is_new,
        free_trial_used=bool(user.get("free_trial_used")),
        paid=bool(user.get("paid")),
    )


@app.post("/session/start", response_model=SessionStartResp)
async def session_start(req: SessionStartReq):
    user = db.get_user(req.user_id)
    if not user:
        user = db.create_user(req.user_id, affiliate_code=req.affiliate_code)

    if req.plan == "free_trial":
        if user.get("free_trial_used"):
            raise HTTPException(402, "Free trial already used — upgrade to continue")
        max_dur = FREE_TRIAL_MAX_SEC
    else:
        if not user.get("paid"):
            raise HTTPException(402, "Payment required for full session")
        max_dur = PAID_MAX_SEC

    tok = await mint_realtime_token(req.test_type, req.background_summary)
    if not tok["token"]:
        raise HTTPException(502, "Could not mint voice token")

    session_id = db.create_session(
        user_id=req.user_id,
        test_type=req.test_type,
        plan=req.plan,
        max_duration_sec=max_dur,
        affiliate_code=req.affiliate_code or user.get("affiliate_code"),
    )
    log.info("session %s started for user %s plan=%s", session_id, req.user_id, req.plan)
    return SessionStartResp(
        session_id=session_id,
        ephemeral_token=tok["token"],
        expires_in_sec=tok["expires_in"],
        max_duration_sec=max_dur,
        ws_url_path=f"/ws/voice/{session_id}",
    )


@app.post("/session/end")
async def session_end(req: SessionEndReq):
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess.get("score") is not None:
        # Already scored — idempotent
        return {
            "session_id": req.session_id,
            "score": sess["score"],
            "verdict": sess["verdict"],
            "weaknesses": sess.get("weaknesses", []),
            "feedback": sess.get("feedback", ""),
        }

    transcript = sess.get("transcript", [])
    if not transcript:
        raise HTTPException(409, "No transcript captured — session had no audio")

    result = await scoring.score_transcript(transcript, sess.get("test_type", "general"))
    db.update_session_scored(
        req.session_id,
        score=result["total"],
        verdict=result["verdict"],
        breakdown={k: result[k] for k in ("clarity", "ties", "financial", "consistency", "composure")},
        weaknesses=result.get("weaknesses", []),
        feedback=result.get("feedback", ""),
    )
    if sess.get("plan") == "free_trial":
        db.mark_trial_used(sess["user_id"])
    log.info("session %s scored: %s (%s)", req.session_id, result["total"], result["verdict"])
    return {"session_id": req.session_id, **result}


@app.post("/payment/stars-invoice")
async def stars_invoice(req: StarsInvoiceReq):
    """Create a Telegram Stars invoice link for the full speaking test.
    1-click native purchase inside Telegram — no forms, no KYC."""
    user = db.get_user(req.user_id)
    if not user:
        user = db.create_user(req.user_id, affiliate_code=req.affiliate_code)
    if user.get("paid"):
        raise HTTPException(409, "User already paid")

    payload_token = f"{req.user_id}:full:{int(time.time())}"
    db.record_stars_invoice(payload_token, req.user_id, req.affiliate_code)

    data = await tg_api("createInvoiceLink", {
        "title": "Full Speaking Test + Report",
        "description": "15-minute strict speaking test with detailed 0-100 score report",
        "payload": payload_token,
        # NOTE: provider_token intentionally omitted — required for Telegram Stars (XTR)
        "currency": "XTR",
        "prices": [{"label": "Full Test", "amount": FULL_TEST_PRICE_STARS}],
    })
    return {"invoice_url": data["result"], "payload": payload_token}


@app.post("/payment/telegram-webhook")
async def telegram_webhook(request: Request):
    """Handle Telegram payment updates: pre_checkout_query + successful_payment."""
    if TELEGRAM_WEBHOOK_SECRET:
        got = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if got != TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(401, "Bad webhook secret")

    update = await request.json()

    pcq = update.get("pre_checkout_query")
    if pcq:
        # We trust Telegram's payment plumbing — approve everything.
        await tg_api("answerPreCheckoutQuery", {
            "pre_checkout_query_id": pcq["id"],
            "ok": True,
        })
        return {"ok": True}

    msg = update.get("message") or {}
    sp = msg.get("successful_payment")
    if sp:
        charge_id = sp.get("telegram_payment_charge_id", "")
        if not charge_id:
            return {"ok": True}
        if db.get_payment(charge_id):
            return {"ok": True}  # idempotent — already processed

        payload_token = sp.get("invoice_payload", "") or ""
        inv = db.get_stars_invoice(payload_token) if payload_token else None
        user_id = (inv or {}).get("user_id") or payload_token.split(":")[0]
        affiliate_code = (inv or {}).get("affiliate_code")
        if not user_id:
            log.warning("successful_payment with no resolvable user: %s", charge_id)
            return {"ok": True}

        db.record_payment(
            payment_id=charge_id,
            user_id=user_id,
            amount_stars=int(sp.get("total_amount", 0) or 0),
            status="captured",
            affiliate_code=affiliate_code,
        )
        db.mark_paid(user_id)
        if payload_token:
            db.mark_invoice_used(payload_token)
        log.info("user %s paid %s Stars (charge %s)",
                 user_id, sp.get("total_amount"), charge_id)
    return {"ok": True}


@app.get("/affiliate/stats/{code}")
async def affiliate_stats(code: str):
    stats = db.get_affiliate_stats(code)
    if not stats:
        raise HTTPException(404, "Affiliate not found")
    return stats


@app.websocket("/ws/voice/{session_id}")
async def ws_voice(ws: WebSocket, session_id: str):
    await voice_proxy(ws, session_id)

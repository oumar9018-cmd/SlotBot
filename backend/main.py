"""
Strict English Voice Test — Backend (FastAPI)
AI-powered strict English speaking test. Stern examiner persona via Grok (xAI)
for dialogue, Deepgram for speech-to-text and text-to-speech.

Endpoints:
  POST /api/auth/telegram           — verify Telegram initData (HMAC-SHA256), get/create user
  POST /session/start              — create session record (no token minting; WS is server-driven)
  POST /session/end                — score transcript, save, return verdict
  POST /payment/stars-invoice      — create Telegram Stars invoice link (750 Stars)
  POST /payment/telegram-webhook   — pre_checkout approval + successful_payment handler
  GET  /affiliate/stats/{code}
  WS   /ws/voice/{session_id}      (realtime_proxy: Deepgram STT -> Grok -> Deepgram TTS)
"""
import os
import time
import logging
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import firestore_db as db
import scoring
from realtime_proxy import voice_proxy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("englishtest")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")  # optional

# Pricing: pay-per-use interview credit bundles — Telegram Stars (XTR), 1-click native
FULL_TEST_PRICE_STARS = 750  # legacy single full-test price (kept for backward compat)

BUNDLES = {
    "reality": {
        "id": "reality",
        "name": "Reality Check",
        "stars": 50,
        "interviews": 1,
        "title": "Reality Check — 1 Mock Interview",
        "description": "1 full strict mock interview + brutal feedback report",
    },
    "serious": {
        "id": "serious",
        "name": "Serious Candidate",
        "stars": 200,
        "interviews": 5,
        "title": "Serious Candidate — 5 Mock Interviews",
        "description": "5 full strict mock interviews + feedback reports",
    },
    "panic": {
        "id": "panic",
        "name": "Panic Mode",
        "stars": 500,
        "interviews": 15,
        "title": "Panic Mode — 15 Mock Interviews",
        "description": "15 full strict mock interviews + feedback reports. Interview in 3 days? This is for you.",
    },
}

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


@app.on_event("startup")
async def check_required_env():
    """Log warnings for missing env vars so misconfiguration shows up in Render logs."""
    missing = []
    for name in ("TELEGRAM_BOT_TOKEN", "GROQ_API_KEY", "DEEPGRAM_API_KEY"):
        if not os.environ.get(name):
            missing.append(name)
            log.warning("MISSING env var %s — related features will fail", name)
    if not os.environ.get("FIREBASE_CRED_PATH"):
        log.info("FIREBASE_CRED_PATH not set — using in-memory Firestore fallback")
    if not os.environ.get("TELEGRAM_WEBHOOK_SECRET"):
        log.info("TELEGRAM_WEBHOOK_SECRET not set — webhook secret check disabled")
    if missing:
        log.warning("Backend starting with missing env vars: %s", ", ".join(missing))
    else:
        log.info("All required env vars present")


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
    ephemeral_token: str = ""  # legacy field, unused — voice WS is fully server-driven
    expires_in_sec: int = 0
    max_duration_sec: int
    ws_url_path: str


class SessionEndReq(BaseModel):
    session_id: str


class StarsInvoiceReq(BaseModel):
    user_id: str
    affiliate_code: str | None = None
    bundle_id: str | None = None  # reality | serious | panic — None = legacy 750 flow


# ---------- Helpers ----------


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


@app.get("/test-keys")
async def test_keys():
    """Diagnostic: test Groq and Deepgram API keys. Returns status for each."""
    import httpx
    groq_key = os.environ.get("GROQ_API_KEY", "")
    dg_key = os.environ.get("DEEPGRAM_API_KEY", "")
    result = {}

    # Test Groq
    if not groq_key:
        result["Groq_Status"] = "MISSING_KEY"
    else:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # First: list available models for this key
                mr = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {groq_key}"},
                )
                if mr.status_code == 200:
                    models = [m.get("id", "?") for m in mr.json().get("data", [])]
                    result["Groq_Available_Models"] = sorted(models)
                else:
                    result["Groq_Available_Models"] = f"list failed: {mr.status_code}"

                # Then: test chat completion with configured model
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={"model": "openai/gpt-oss-20b",
                          "messages": [{"role": "user", "content": "Say hi"}],
                          "max_tokens": 5},
                )
            result["Groq_Status"] = "200 OK" if r.status_code == 200 else f"{r.status_code} {r.text[:100]}"
        except Exception as e:
            result["Groq_Status"] = f"ERROR: {str(e)[:100]}"

    # Test Deepgram (TTS endpoint with minimal text)
    if not dg_key:
        result["Deepgram_Status"] = "MISSING_KEY"
    else:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    "https://api.deepgram.com/v1/speak?model=aura-asteria-en",
                    headers={"Authorization": f"Token {dg_key}", "Content-Type": "application/json"},
                    json={"text": "hi"},
                )
            result["Deepgram_Status"] = "200 OK" if r.status_code == 200 else f"{r.status_code} {r.text[:100]}"
        except Exception as e:
            result["Deepgram_Status"] = f"ERROR: {str(e)[:100]}"

    return result


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

    session_id = db.create_session(
        user_id=req.user_id,
        test_type=req.test_type,
        plan=req.plan,
        max_duration_sec=max_dur,
        affiliate_code=req.affiliate_code or user.get("affiliate_code"),
        background_summary=req.background_summary,
    )
    log.info("session %s started for user %s plan=%s", session_id, req.user_id, req.plan)
    return SessionStartResp(
        session_id=session_id,
        ephemeral_token="",
        expires_in_sec=0,
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


class FeedbackAudioReq(BaseModel):
    session_id: str


@app.post("/session/feedback-audio")
async def session_feedback_audio(req: FeedbackAudioReq):
    """Synthesize the session verdict + feedback with the FRIENDLY 'Visa Coach'
    voice (aura-luna-en). Returns raw PCM16 24kHz mono audio bytes.
    The interview uses the strict officer voice; feedback uses the warm coach."""
    from fastapi.responses import Response
    from realtime_proxy import deepgram_tts
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess.get("score") is None:
        raise HTTPException(409, "Session not scored yet")
    verdict = sess.get("verdict", "")
    feedback = sess.get("feedback", "")
    score = sess.get("score", 0)
    script = (
        f"Your score is {score} out of 100. Verdict: {verdict}. "
        f"Here is my honest coaching feedback. {feedback} "
        "This was a practice simulation, not an official assessment."
    )
    audio = await deepgram_tts(script, voice_mode="friendly")
    return Response(content=audio, media_type="audio/pcm")


@app.post("/payment/stars-invoice")
async def stars_invoice(req: StarsInvoiceReq):
    """Create a Telegram Stars invoice link.
    Bundle flow (bundle_id set): pay-per-use interview credit bundles.
    Legacy flow (bundle_id None): 750-Star full test (backward compat)."""
    user = db.get_user(req.user_id)
    if not user:
        user = db.create_user(req.user_id, affiliate_code=req.affiliate_code)

    bundle = BUNDLES.get(req.bundle_id) if req.bundle_id else None
    if req.bundle_id and not bundle:
        raise HTTPException(400, f"Unknown bundle_id: {req.bundle_id}")
    if bundle:
        stars = bundle["stars"]
        title = bundle["title"]
        description = bundle["description"]
        label = bundle["name"]
        payload_token = f"{req.user_id}:{bundle['id']}:{uuid.uuid4().hex[:12]}"
    else:
        if user.get("paid"):
            raise HTTPException(409, "User already paid")
        stars = FULL_TEST_PRICE_STARS
        title = "Full Speaking Test + Report"
        description = "15-minute strict speaking test with detailed 0-100 score report"
        label = "Full Test"
        payload_token = f"{req.user_id}:full:{int(time.time())}"

    db.record_stars_invoice(payload_token, req.user_id, req.affiliate_code,
                            bundle_id=bundle["id"] if bundle else None)

    if not TELEGRAM_BOT_TOKEN:
        log.error("stars-invoice called but TELEGRAM_BOT_TOKEN is missing/empty")
        raise HTTPException(
            500, "Payment not configured: TELEGRAM_BOT_TOKEN missing on server")
    try:
        data = await tg_api("createInvoiceLink", {
            "title": title,
            "description": description,
            "payload": payload_token,
            # NOTE: provider_token intentionally omitted — required for Telegram Stars (XTR)
            "currency": "XTR",
            "prices": [{"label": label, "amount": stars}],
        })
    except HTTPException:
        raise
    except Exception as e:
        log.exception("createInvoiceLink failed unexpectedly")
        raise HTTPException(502, f"Telegram payment error: {str(e)[:200]}")
    return {"invoice_url": data["result"], "payload": payload_token,
            "stars": stars, "bundle_id": bundle["id"] if bundle else None,
            "interviews": bundle["interviews"] if bundle else 0}


WELCOME_TEXT = (
    "🔥 Welcome to VoiceGrill — the world's STRICTEST AI interview coach!\n\n"
    "Grilly the examiner will grill you like a real visa officer: "
    "interrupts rambling, catches contradictions, scores you 0-100 with zero mercy.\n\n"
    "✅ First 3-question mini interview FREE\n"
    "✅ Then 1 full trial interview (3 days)\n\n"
    "Tap below to start getting grilled! 👇"
)

WELCOME_KEYBOARD = {
    "inline_keyboard": [[{
        "text": "🎤 Start Free Test",
        "web_app": {"url": "https://oumar9018-cmd.github.io/SlotBot/"},
    }]]
}


@app.post("/payment/telegram-webhook")
async def telegram_webhook(request: Request):
    """Handle Telegram updates: /start welcome + pre_checkout_query + successful_payment."""
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

    # /start welcome message — must never break payment handling below
    text = (msg.get("text") or "").strip()
    if text.startswith("/start"):
        chat_id = (msg.get("chat") or {}).get("id")
        try:
            await tg_api("sendMessage", {
                "chat_id": chat_id,
                "text": WELCOME_TEXT,
                "reply_markup": WELCOME_KEYBOARD,
            })
            log.info("sent /start welcome to chat %s", chat_id)
        except Exception:
            log.exception("failed to send /start welcome to chat %s", chat_id)
        return {"ok": True}

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
        # Bundle flow: credit interview interviews to the user's account.
        bundle_id = (inv or {}).get("bundle_id")
        bundle = BUNDLES.get(bundle_id) if bundle_id else None
        if bundle:
            db.add_interview_credits(user_id, bundle["interviews"])
            log.info("user %s credited %d interviews (bundle %s)",
                     user_id, bundle["interviews"], bundle_id)
        if payload_token:
            db.mark_invoice_used(payload_token)
        log.info("user %s paid %s Stars (charge %s)",
                 user_id, sp.get("total_amount"), charge_id)
    return {"ok": True}


@app.get("/payment/credits/{user_id}")
async def payment_credits(user_id: str):
    """Return the user's remaining interview credits (single source of truth)."""
    return {
        "user_id": user_id,
        "interview_credits": db.get_interview_credits(user_id),
    }


@app.get("/payment/bundles")
async def payment_bundles():
    """Public bundle catalog for the Mini App store."""
    return {"bundles": list(BUNDLES.values())}


@app.get("/affiliate/stats/{code}")
async def affiliate_stats(code: str):
    stats = db.get_affiliate_stats(code)
    if not stats:
        raise HTTPException(404, "Affiliate not found")
    return stats


@app.websocket("/ws/voice/{session_id}")
async def ws_voice(ws: WebSocket, session_id: str):
    await voice_proxy(ws, session_id)

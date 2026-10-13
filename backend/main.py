"""
MockGrill — Backend (FastAPI)
AI-powered mock interview practice. Stern examiner persona via Groq
for dialogue, Deepgram for speech-to-text and text-to-speech.

Endpoints:
  POST /api/auth/telegram           — verify Telegram initData (HMAC-SHA256), get/create user
  POST /session/start              — create session record; mints a WS session
                                   token (ws_token) ONLY when verified Telegram
                                   initData is supplied (initData.user.id must
                                   equal user_id). Without initData, ws_token
                                   is "" and the WS must be opened with
                                   ?initData= instead (verified per-connect).
  POST /session/end                — score transcript, save, return verdict
  POST /payment/stars-invoice      — create Telegram Stars invoice link (750 Stars)
  POST /payment/telegram-webhook   — pre_checkout approval + successful_payment handler
  GET  /affiliate/stats/{code}
  WS   /ws/voice/{session_id}?token=... | ?initData=...
                                   (realtime_proxy: Deepgram STT -> Groq -> Deepgram TTS)
                                   WS AUTH (Wave 37): the WS is CLOSED without
                                   proof the caller owns the session. Exactly one
                                   of ?token= (server-issued HMAC session token
                                   from /session/start) or ?initData= (raw Telegram
                                   WebApp initData, HMAC-verified, user must equal
                                   the session owner) is required. If ?token= is
                                   present it is authoritative — a bad token is
                                   rejected even if initData is also supplied.
                                   Unverified WS is closed (4401/4403/4404)
                                   BEFORE ws.accept() and before any Deepgram
                                   connect. Unknown session_ids are rejected —
                                   the WS path NEVER creates sessions (the old
                                   temp-session fallback in voice_proxy was a
                                   free-voice hole: any random id got a session).
"""
import asyncio
import base64
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
from realtime_proxy import request_session_flush, voice_proxy

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

app = FastAPI(title="MockGrill API", version="0.2.0")

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
    initData: str | None = Field(None,
        description="Optional raw Telegram WebApp initData. When supplied it is "
        "HMAC-verified BEFORE any credit is consumed, its user id must equal "
        "user_id, and a WS session token (ws_token) is minted for this session. "
        "Without it ws_token is '' and the WS must be opened with ?initData=.")


class SessionStartResp(BaseModel):
    session_id: str
    ephemeral_token: str = ""  # legacy field, unused — voice WS is fully server-driven
    expires_in_sec: int = 0
    max_duration_sec: int
    ws_url_path: str
    ws_token: str = ""  # server-issued WS session token ("" when no initData was verified)


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


# ---------- WS session tokens (Wave 37) ----------
# Stateless, single-session, expiring proof of session ownership for
# /ws/voice/{session_id}. Format (base64url, no padding):
#   base64url("<session_id>.<user_id>.<exp_unix>.<hmac_hex>")
# where hmac = HMAC-SHA256(key, "<session_id>.<user_id>.<exp_unix>").
# Statelessness is deliberate: firestore_db.create_session has no token
# field, so storing a token would need a DB-schema change. An HMAC token
# needs no storage, is bound to ONE session + ONE owner, and expires with
# the session (max_duration + grace). It is NOT a general auth token and
# must never be accepted for any other session or endpoint.
# Signing key: WS_AUTH_SECRET when set, else the Telegram bot token.
# Set WS_AUTH_SECRET in production so WS tokens survive a bot-token rotation
# and so a leaked bot token alone does not mint WS tokens. The key is read
# from the environment on every call (never cached) so rotation takes effect
# without a restart; outstanding tokens signed with the old key then fail
# verification, which is the intended behaviour.

def _ws_signing_key() -> bytes:
    secret = os.environ.get("WS_AUTH_SECRET") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    return secret.encode()


def mint_session_token(session_id: str, user_id: str, ttl_sec: int) -> str:
    """Mint a WS session token for (session_id, user_id), valid ttl_sec seconds.

    Callers MUST only mint after the caller's identity was verified (verified
    initData whose user id equals user_id) — a token minted for an unverified
    user_id would let anyone use a session they created for someone else.
    Returns "" when no signing key is configured (fail-closed: the WS will
    then only accept ?initData=).
    """
    if not session_id or user_id is None:
        return ""
    if not _ws_signing_key():
        log.error("mint_session_token: no WS signing key (WS_AUTH_SECRET / TELEGRAM_BOT_TOKEN)")
        return ""
    exp = int(time.time()) + int(ttl_sec)
    payload = f"{session_id}.{user_id}.{exp}"
    sig = hmac.new(_ws_signing_key(), payload.encode(), hashlib.sha256).hexdigest()
    raw = f"{payload}.{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def verify_session_token(token: str, session_id: str, expected_user_id: str) -> bool:
    """Verify a WS session token. Returns False on ANY failure (never raises).

    Checks, in order: well-formed base64url + 4 dot-separated parts, token's
    session_id equals the URL session_id, token's user_id equals the session
    owner's user_id, token not expired, HMAC matches (constant-time compare).
    """
    if not token or not session_id or expected_user_id is None:
        return False
    if not _ws_signing_key():
        return False
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        tok_sid, tok_uid, exp_s, sig = raw.split(".", 3)
    except Exception:
        return False
    if tok_sid != session_id or str(tok_uid) != str(expected_user_id):
        return False
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < time.time():
        return False
    expected_sig = hmac.new(
        _ws_signing_key(), f"{tok_sid}.{tok_uid}.{exp_s}".encode(),
        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_sig, sig)


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

    # Test Deepgram STT WebSocket (the actual voice pipeline connection)
    if dg_key:
        try:
            import websockets
            from realtime_proxy import DG_STT_URL
            try:
                dg_ws = await websockets.connect(
                    DG_STT_URL,
                    additional_headers={"Authorization": f"Token {dg_key}"},
                    max_size=1024*1024,
                    open_timeout=15,
                )
            except TypeError:
                dg_ws = await websockets.connect(
                    DG_STT_URL,
                    extra_headers={"Authorization": f"Token {dg_key}"},
                    max_size=1024*1024,
                    open_timeout=15,
                )
            await dg_ws.close()
            result["Deepgram_STT_WS"] = "200 OK - nova-3 streaming works"
        except Exception as e:
            result["Deepgram_STT_WS"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

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
    # Wave 37: when the client supplies initData, verify it BEFORE any credit
    # is consumed or any session is created. A bad initData (or one for a
    # different user) must never burn a credit — and a token is minted ONLY
    # from verified initData, so an attacker who starts a session for
    # someone else's user_id gets a session they cannot use on the WS
    # (ws_token stays "" and they do not hold the victim's initData).
    initdata_verified = False
    if req.initData:
        tg_user = verify_telegram_init_data(req.initData, TELEGRAM_BOT_TOKEN)
        if str(tg_user.get("id")) != str(req.user_id):
            raise HTTPException(403, "initData user does not match user_id")
        initdata_verified = True

    user = db.get_user(req.user_id)
    if not user:
        user = db.create_user(req.user_id, affiliate_code=req.affiliate_code)

    consumed_credit = False
    if req.plan == "free_trial":
        if user.get("free_trial_used"):
            raise HTTPException(402, "Free trial already used — upgrade to continue")
        max_dur = FREE_TRIAL_MAX_SEC
    else:
        # Paid plan: 'paid' flag = unlimited access, otherwise consume one
        # interview credit atomically. use_interview_credit is transactional:
        # True = a credit was taken, False = none available.
        if not user.get("paid"):
            if not db.use_interview_credit(req.user_id):
                raise HTTPException(402, "Payment required for full session")
            consumed_credit = True
        max_dur = PAID_MAX_SEC

    try:
        session_id = db.create_session(
            user_id=req.user_id,
            test_type=req.test_type,
            plan=req.plan,
            max_duration_sec=max_dur,
            affiliate_code=req.affiliate_code or user.get("affiliate_code"),
            background_summary=req.background_summary,
        )
    except Exception:
        # Session creation failed after we consumed a credit — refund it so a
        # failed start never burns a credit.
        if consumed_credit:
            try:
                db.add_interview_credits(req.user_id, 1)
            except Exception:
                log.exception("credit refund failed for user %s", req.user_id)
        raise
    log.info("session %s started for user %s plan=%s", session_id, req.user_id, req.plan)
    # Mint the WS token ONLY from verified initData (see above). Token TTL =
    # session length + 10 min grace for reconnects; the WS re-checks expiry.
    ws_token = ""
    ws_url_path = f"/ws/voice/{session_id}"
    if initdata_verified:
        ws_token = mint_session_token(session_id, req.user_id, ttl_sec=max_dur + 600)
        if ws_token:
            ws_url_path = f"/ws/voice/{session_id}?token={ws_token}"
        else:
            # No signing key configured — client must fall back to ?initData=.
            log.error("session %s: initData verified but no WS signing key — no token minted", session_id)
    return SessionStartResp(
        session_id=session_id,
        ephemeral_token="",
        expires_in_sec=0,
        max_duration_sec=max_dur,
        ws_url_path=ws_url_path,
        ws_token=ws_token,
    )


@app.post("/session/end")
async def session_end(req: SessionEndReq):
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    # FLUSH-BEFORE-SCORE (VOICE WAVE 47/50): the transcript lives in the
    # voice_proxy task's memory and only reaches DB on finish / timeup /
    # socket close. A manual finish calls this endpoint while the WS is
    # still open (or just closed), so scoring the DB transcript as-is
    # 409s on "empty" even though the user spoke. Fix in two layers:
    #  1. If the WS is still live in this process, signal it to flush
    #     (request_session_flush -> finisher()).
    #  2. Poll DB up to ~8s for the flushed transcript (covers the
    #     close-then-score race: WS closed, socket-close flush in flight).
    # Break early when the session is fully closed (ended_at set) — the
    # flush already landed, so an empty transcript there is genuinely empty.
    # The canonical frontend path (task 26 resync) sends {"type":"finish"}
    # and waits for the {"type":"flushed"} ack before calling here, so in
    # practice this loop usually breaks on the first check.
    try:
        request_session_flush(req.session_id)
    except Exception:
        log.exception("request_session_flush failed for %s", req.session_id)
    for _ in range(16):  # 16 x 0.5s = up to 8s
        if sess.get("score") is not None or sess.get("denied"):
            break
        if sess.get("transcript"):
            break
        if sess.get("ended_at") is not None:
            break  # socket-close flush already landed; transcript is final
        await asyncio.sleep(0.5)
        sess = db.get_session(req.session_id)
        if not sess:
            raise HTTPException(404, "Session not found")

    # Feature 4: instant cut-off — the officer ended the interview early.
    if sess.get("denied"):
        reason = sess.get("denied_reason", "Over-explaining / Vague Answers")
        log.info("session %s was denied: %s", req.session_id, reason)
        return {
            "session_id": req.session_id,
            "score": 0,
            "total": 0,
            "verdict": "DENIED",
            "denied_reason": reason,
            "weaknesses": [reason],
            "feedback": ("The officer stopped the interview. " + reason + ". "
                         "In a real interview, long stories get cut off — answer directly, "
                         "with facts, in under 20 seconds."),
            "nervousness_score": int(sess.get("nervousness_score", 0) or 0),
            "nervousness_level": "High" if int(sess.get("nervousness_score", 0) or 0) > 60 else "Medium",
            "over_explain_count": int(sess.get("over_explain_count", 0) or 0),
        }

    if sess.get("score") is not None:
        # Already scored — idempotent (must include "total": frontend reads res.total)
        return {
            "session_id": req.session_id,
            "score": sess["score"],
            "total": sess["score"],
            "verdict": sess["verdict"],
            "weaknesses": sess.get("weaknesses", []),
            "feedback": sess.get("feedback", ""),
        }

    # Race guard: only one concurrent /session/end may run the (expensive)
    # Groq scoring. Others wait for the winner or get the cached result.
    claim = db.claim_scoring(req.session_id)
    if claim == "scored":
        sess = db.get_session(req.session_id)
        return {
            "session_id": req.session_id,
            "score": sess["score"],
            "total": sess["score"],
            "verdict": sess["verdict"],
            "weaknesses": sess.get("weaknesses", []),
            "feedback": sess.get("feedback", ""),
        }
    if claim == "in_progress":
        # Another request is scoring right now — poll briefly for its result
        # instead of firing a duplicate Groq call.
        for _ in range(30):
            await asyncio.sleep(1)
            sess = db.get_session(req.session_id)
            if sess and sess.get("score") is not None:
                return {
                    "session_id": req.session_id,
                    "score": sess["score"],
                    "total": sess["score"],
                    "verdict": sess["verdict"],
                    "weaknesses": sess.get("weaknesses", []),
                    "feedback": sess.get("feedback", ""),
                }
        raise HTTPException(409, "Scoring in progress — retry shortly")
    if claim == "missing":
        raise HTTPException(404, "Session not found")

    # Re-fetch: the flush may have landed while we were claiming, and the
    # `sess` above can be up to one poll-loop iteration stale.
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    transcript = sess.get("transcript", [])
    if not transcript:
        db.release_scoring(req.session_id)
        raise HTTPException(409, "No transcript captured — session had no audio")

    # Feature 2: pass the filler-word nervousness score into scoring
    nervousness = int(sess.get("nervousness_score", 0) or 0)
    try:
        result = await scoring.score_transcript(
            transcript, sess.get("test_type", "general"),
            nervousness_score=nervousness)
    except Exception:
        # Scoring crashed — release the claim so a retry isn't stuck behind
        # a stale in_progress flag.
        db.release_scoring(req.session_id)
        raise
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
    log.info("session %s scored: %s (%s) nervousness=%d", req.session_id,
             result["total"], result["verdict"], nervousness)
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
    "🎤 Welcome to MockGrill — AI Mock Interview Practice Bot!\n\n"
    "Practice for:\n"
    "🇺🇸 US Visa Interview (F1, B1/B2, H1B)\n"
    "❌ 214(b) Rejection Practice\n"
    "💼 Job Interview Mock Test\n"
    "🎓 University Admission Interview\n\n"
    "Grilly the strict AI examiner will grill you, interrupt rambling, "
    "probe contradictions, and score you 0-100 with zero mercy.\n\n"
    "🎁 FIRST 2-MINUTE GRILLING FREE!\n\n"
    "Tap below to start! 👇"
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
    # SECURITY: fail-closed — reject all webhooks if no secret configured,
    # rather than accepting forged payments.
    if not TELEGRAM_WEBHOOK_SECRET:
        log.error("TELEGRAM_WEBHOOK_SECRET not set — rejecting webhook")
        raise HTTPException(503, "Webhook not configured")
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

        # Idempotent + race-safe: record_payment returns True only for the
        # first processing of this charge (Firestore transaction). Retries or
        # concurrent deliveries get False and grant nothing.
        is_new = db.record_payment(
            payment_id=charge_id,
            user_id=user_id,
            amount_stars=int(sp.get("total_amount", 0) or 0),
            status="captured",
            affiliate_code=affiliate_code,
        )
        if not is_new:
            log.info("duplicate webhook for charge %s — credits not re-granted", charge_id)
            return {"ok": True}
        # Bundle flow: credit interview credits. Do NOT mark_paid for bundles —
        # paid=True would grant unlimited sessions after credits run out.
        bundle_id = (inv or {}).get("bundle_id")
        bundle = BUNDLES.get(bundle_id) if bundle_id else None
        if bundle:
            db.add_interview_credits(user_id, bundle["interviews"])
        else:
            # Legacy non-bundle payment: mark paid access.
            db.mark_paid(user_id)
            # FIX 2026-10-13 (scan HIGH): bundle is None here — old code
            # referenced bundle["interviews"] → TypeError → webhook crash
            # after mark_paid succeeded. Log without bundle reference.
            log.info("user %s granted legacy paid access (bundle %s)",
                     user_id, bundle_id)
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


class ReferralClaimReq(BaseModel):
    user_id: str
    referrer_id: str


@app.post("/referral/claim")
async def referral_claim(req: ReferralClaimReq):
    """Claim a referral bonus: +1 free interview for the new user AND +1 for
    the referrer. One claim per user ever; self-referral rejected."""
    if not req.user_id or not req.referrer_id:
        raise HTTPException(400, "user_id and referrer_id are required")
    if req.user_id == req.referrer_id:
        raise HTTPException(400, "self_referral")
    try:
        res = db.claim_referral(req.user_id, req.referrer_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, **res}


@app.get("/referral/stats/{user_id}")
async def referral_stats(user_id: str):
    """Referral stats for the invite UI: friends joined + free interviews earned."""
    return db.get_referral_stats(user_id)


@app.get("/affiliate/stats/{code}")
async def affiliate_stats(code: str):
    stats = db.get_affiliate_stats(code)
    if not stats:
        raise HTTPException(404, "Affiliate not found")
    return stats


# ---------- Feature 5: Daily Grilly Push (retention loop) ----------
# Cron (Render cron job or external scheduler) hits this endpoint once a day.
# It messages users who used their free trial but never bought Stars.

DAILY_GRILLY_QUESTIONS = [
    "Why should I approve your visa? Convince me in 20 seconds.",
    "Who is paying for your trip? Give me exact numbers.",
    "Why this university and not a cheaper one?",
    "What will you do after your course ends? Be specific.",
    "Show me proof your sponsor actually has this money.",
    "You have 3 backlogs. Why should I trust your academics?",
    "Why didn't you apply to universities in your own country?",
    "What does your father do? How much does he earn monthly?",
    "If I reject you today, what changes before you reapply?",
    "Name 3 professors in your department. Right now.",
    "How will you survive financially without working illegally?",
    "Your English is shaky. How will you attend lectures?",
    "Why this course? It has nothing to do with your bachelor's.",
    "Who will take care of your family while you're away?",
    "Give me one reason you're not an immigration risk.",
    "What is your exact plan for the first 30 days after landing?",
    "Your bank statement is 2 months old. Explain.",
    "Why should I believe you'll return home after your studies?",
    "What happens if you fail a semester? Who pays then?",
    "Describe your course syllabus. No vague answers.",
    "How many universities did you apply to? Why only this one?",
    "Your sponsor's income doesn't match the fees. Explain.",
    "What ties do you have to your home country? List them.",
    "Why is your passport almost empty? No travel history?",
    "Tell me your daily schedule as a student. Be precise.",
    "What will you do if your visa is rejected today?",
    "How did you hear about this university? Who told you?",
    "Your answers are too rehearsed. Say something real.",
    "What is the biggest weakness in your application?",
    "Convince me you're not using studies as an immigration cover.",
]

DAILY_GRILLY_KEYBOARD = {
    "inline_keyboard": [[{
        "text": "🎤 Practice Properly — Open MockGrill",
        "web_app": {"url": "https://oumar9018-cmd.github.io/SlotBot/"},
    }]]
}


class DailyGrillyReq(BaseModel):
    dry_run: bool = False
    limit: int = 100
    secret: str | None = None


@app.post("/notify/daily-grilly")
async def notify_daily_grilly(req: DailyGrillyReq):
    """Daily retention push — requires CRON_SECRET env var to match req.secret.
    Prevents anyone from triggering mass Telegram messages."""
    import os
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or req.secret != expected:
        raise HTTPException(403, "Invalid or missing cron secret")
    """Daily retention push: message trial users who never bought Stars with a
    brutal practice question. Called by an external cron once per day."""
    from datetime import date
    targets = db.get_daily_push_targets(limit=max(1, min(req.limit, 500)))
    if req.dry_run:
        return {"ok": True, "dry_run": True, "targets": len(targets),
                "sample": targets[:3]}

    # Rotate the question daily — deterministic so a retry same-day is idempotent
    q = DAILY_GRILLY_QUESTIONS[date.today().timetuple().tm_yday % len(DAILY_GRILLY_QUESTIONS)]
    sent, failed = 0, 0
    for t in targets:
        chat_id = t.get("telegram_id")
        name = (t.get("first_name") or "candidate").split()[0]
        text = (
            f"🔥 Grilly is watching, {name}.\n\n"
            f"Today's brutal question:\n\"{q}\"\n\n"
            "Reply with a VOICE NOTE and I'll roast your answer in one line. "
            "Or tap below and practice properly — before the real officer does this to you."
        )
        try:
            await tg_api("sendMessage", {
                "chat_id": chat_id,
                "text": text,
                "reply_markup": DAILY_GRILLY_KEYBOARD,
            })
            sent += 1
        except Exception as e:
            log.warning("daily-grilly failed for %s: %s", chat_id, str(e)[:120])
            failed += 1
    log.info("daily-grilly: sent=%d failed=%d question=%s", sent, failed, q[:40])
    return {"ok": True, "sent": sent, "failed": failed,
            "question": q, "date": date.today().isoformat()}


@app.websocket("/ws/voice/{session_id}")
async def ws_voice(ws: WebSocket, session_id: str):
    # WAVE 37 — WS AUTH GATE. Runs BEFORE ws.accept() and before voice_proxy
    # (hence before any Deepgram/Groq connection). Closes unverified sockets
    # without accepting them; Starlette allows close() before accept().
    # Close codes: 4401 = missing/invalid credential, 4403 = session belongs
    # to another user, 4404 = unknown session / lookup failed.
    params = ws.query_params
    token = (params.get("token") or "").strip()
    init_data = (params.get("initData") or "").strip()
    if not token and not init_data:
        log.warning("ws_voice: rejected session %s — no token or initData", session_id)
        await ws.close(code=4401, reason="missing ws auth: token or initData required")
        return
    try:
        sess = db.get_session(session_id)
    except Exception as e:
        log.error("ws_voice: db.get_session failed for %s: %s", session_id, e)
        await ws.close(code=4404, reason="session lookup failed")
        return
    if not sess:
        log.warning("ws_voice: rejected unknown session %s", session_id)
        await ws.close(code=4404, reason="unknown session")
        return
    owner = str(sess.get("user_id", "") or "")
    if not owner:
        log.warning("ws_voice: rejected session %s — no owner", session_id)
        await ws.close(code=4403, reason="session has no owner")
        return
    # If ?token= is present it is authoritative. A bad token is rejected even
    # when initData is also supplied (fail-closed: no silent downgrade to the
    # weaker/fallback credential, and no oracle for which credential failed
    # beyond the close code).
    if token:
        if not verify_session_token(token, session_id, owner):
            log.warning("ws_voice: rejected session %s — invalid/expired token", session_id)
            await ws.close(code=4401, reason="invalid or expired session token")
            return
        verified_user_id = owner
    else:
        if not TELEGRAM_BOT_TOKEN:
            log.error("ws_voice: initData auth impossible — TELEGRAM_BOT_TOKEN not configured")
            await ws.close(code=4401, reason="telegram auth not configured")
            return
        try:
            tg_user = verify_telegram_init_data(init_data, TELEGRAM_BOT_TOKEN)
        except HTTPException:
            log.warning("ws_voice: rejected session %s — invalid initData", session_id)
            await ws.close(code=4401, reason="invalid initData")
            return
        if str(tg_user.get("id")) != owner:
            log.warning("ws_voice: rejected session %s — initData user != owner", session_id)
            await ws.close(code=4403, reason="session belongs to another user")
            return
        verified_user_id = owner
    # Authenticated: voice_proxy re-checks ownership (defense-in-depth) and
    # only then accepts the socket and connects to Deepgram.
    await voice_proxy(ws, session_id, verified_user_id=verified_user_id)




# ---------- TEMPORARY admin endpoints (added 2026-10-12 ~20:25 IST) ----------
# Purpose: grant 10 test credits to user 1664819949 (@TheCaddi) for the voice
# rebuild v1 test, AFTER the Render service account fix is deployed.
# DELETE THIS WHOLE BLOCK and redeploy once the user confirms credits on device.
# Safety: the grant REFUSES (503) when Firestore is not really connected, so a
# silent in-memory "grant" (the 2026-10-12 ~20:10 IST failure) cannot repeat.

class GrantCreditsReq(BaseModel):
    user_id: str = Field(..., description="Telegram numeric user id")
    credits: int = Field(..., ge=1, le=50, description="Credits to grant (1-50)")






"""
MockGrill — Backend (FastAPI)
AI-powered mock interview practice. Stern examiner persona via Groq
for dialogue, Deepgram for speech-to-text and text-to-speech.

Endpoints:
  POST /api/auth/telegram           — verify Telegram initData (HMAC-SHA256), get/create user
  POST /session/start              — create session record; REQUIRES verified
                                   Telegram initData (initData.user.id must
                                   equal user_id). Mints a WS session token
                                   (ws_token). No unsigned identity: a bare
                                   user_id without initData is rejected.
  POST /session/end                — score transcript, save, return verdict
                                   (marks trial used ONLY for plan=free_trial,
                                   never for free_mini)
  POST /payment/stars-invoice      — create Telegram Stars invoice link
                                   (bundle_id REQUIRED; legacy 750-Star
                                   unlimited flow deleted)
  POST /payment/telegram-webhook   — pre_checkout approval (validates payload /
                                   amount / currency, ok=False on mismatch) +
                                   successful_payment handler (ONE atomic
                                   grant via db.grant_stars_payment; unknown
                                   payload rejected outright, never mark_paid)
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
Rate limiting (stdlib only): per-IP middleware (120/min default, /health
exempt, 429 + Retry-After) + entity limits on /session/start (10 IP / 3
entity per min), /referral/claim (10/3), /test-keys (2/IP).
CORS: allowlist from CORS_ALLOWED_ORIGINS env, credentials OFF.
"""
import asyncio
import base64
import ipaddress
import os
import threading
import time
import logging
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import firestore_db as db
import scoring
from realtime_proxy import request_session_flush, wait_for_flush_completion, voice_proxy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("englishtest")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")  # optional

# Pricing: pay-per-use interview credit bundles — Telegram Stars (XTR), 1-click native
# NOTE: the legacy 750-Star unlimited flow is DELETED. bundle_id is required
# for invoices; the webhook never grants paid=True. Existing paid=True users
# keep unlimited access via the /session/start paid branch (honoured, not granted).

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
FREE_MINI_MAX_SEC = 120    # free_mini: same 2 min cap as the free trial
PAID_MAX_SEC = 900         # 15 min

# CORS allowlist: comma-separated env, credentials OFF (the old "*" +
# allow_credentials=True let any site read credentialed responses).
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "https://oumar9018-cmd.github.io,https://web.telegram.org,https://telegram.org"
    ).split(",") if o.strip()
]

app = FastAPI(title="MockGrill API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Rate limiting (stdlib only: time + threading, no new deps) ----------
# In-memory sliding-window buckets. Single-process state: Render runs one
# worker by default; if the service ever scales to N workers, each worker
# enforces its own budget (fail-safe direction: stricter, not looser).

_RATE_WINDOW_SEC = 60
_rate_buckets: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _trusted_proxy_networks() -> list:
    """Parse TRUSTED_PROXIES env (comma-separated IPs/CIDRs, e.g.
    '10.0.0.0/8,172.16.0.0/12'). Empty = trust NO proxy: XFF is ignored
    and the direct peer IP is used. Set this to the Render / LB proxy
    ranges in production so XFF is honoured only from those peers."""
    raw = os.environ.get("TRUSTED_PROXIES", "") or ""
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            try:
                nets.append(ipaddress.ip_network(part + "/32", strict=False))
            except ValueError:
                log.warning("TRUSTED_PROXIES: ignoring invalid entry %r", part)
    return nets


def _is_trusted_proxy(peer_ip: str) -> bool:
    """True iff the direct peer IP is in the configured trusted-proxy set."""
    if not peer_ip or peer_ip == "unknown":
        return False
    try:
        addr = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False
    for net in _trusted_proxy_networks():
        try:
            if addr in net:
                return True
        except TypeError:
            continue
    return False


def _client_ip(request: Request) -> str:
    """FIX 11 (trusted-proxy XFF): X-Forwarded-For is honoured ONLY when
    the direct peer is a configured trusted proxy (TRUSTED_PROXIES). Any
    other peer — including a spoofed XFF from an untrusted client — falls
    back to the direct peer IP. Rate limiting only — never used for auth."""
    client = getattr(request, "client", None)
    peer = (client.host if client else "unknown") or "unknown"
    if _is_trusted_proxy(peer):
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    return peer


def _app_version() -> tuple[str, str]:
    """FIX 25: version/commit for /health deploy parity. From env
    (VERSION/APP_VERSION, COMMIT/GIT_COMMIT) or a VERSION file, else
    'unknown'. Never raises — /health must never crash when unset."""
    try:
        version = (os.environ.get("VERSION") or os.environ.get("APP_VERSION") or "").strip()
        commit = (os.environ.get("COMMIT") or os.environ.get("GIT_COMMIT")
                  or os.environ.get("RENDER_GIT_COMMIT") or "").strip()
        if not version or not commit:
            # Best-effort VERSION file: backend/VERSION or repo-root VERSION.
            here = os.path.dirname(os.path.abspath(__file__))
            for path in (os.path.join(here, "VERSION"),
                         os.path.join(here, "..", "VERSION")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        parts = content.split()
                        if not version and parts:
                            version = parts[0]
                        if not commit and len(parts) > 1:
                            commit = parts[1]
                        break
                except OSError:
                    continue
        return version or "unknown", commit or "unknown"
    except Exception:
        return "unknown", "unknown"


def _extract_init_data(request: Request | None, body_init_data: str | None = None) -> str:
    """Collect Telegram initData from (in priority order): request body
    field, X-Telegram-Init-Data header, ?initData= query param. Returns
    '' when none was supplied (caller must 401)."""
    if body_init_data:
        return body_init_data
    if request is None:
        return ""
    hdr = (request.headers.get("x-telegram-init-data") or "").strip()
    if hdr:
        return hdr
    try:
        q = (request.query_params.get("initData") or "").strip()
    except Exception:
        q = ""
    return q


def _verify_session_owner(session_id: str, owner: str,
                          init_data: str = "", ws_token: str = "") -> str:
    """FIX 4: verify the caller owns session_id. Exactly one credential is
    required: a server-issued ws_token (verified against session+owner) OR
    HMAC-verified initData whose user equals the owner. Raises 401 when no
    credential is supplied or verification fails, 403 when the verified
    identity is not the owner. Returns the verified user id."""
    if ws_token:
        if not verify_session_token(ws_token, session_id, owner):
            raise HTTPException(401, "Invalid or expired session token")
        return str(owner)
    if init_data:
        if not TELEGRAM_BOT_TOKEN:
            raise HTTPException(500, "Telegram bot not configured")
        tg_user = verify_telegram_init_data(init_data, TELEGRAM_BOT_TOKEN)
        if str(tg_user.get("id")) != str(owner):
            raise HTTPException(403, "initData user does not own this session")
        return str(owner)
    raise HTTPException(401, "Missing session auth: ws_token or initData required")


def _iso_dt(value) -> str | None:
    """Best-effort ISO string for a Firestore datetime / epoch / string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        try:
            return value.isoformat()
        except Exception:
            return None
    if isinstance(value, (int, float)):
        try:
            sec = value / 1000.0 if value > 1e12 else value
            return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()
        except Exception:
            return None
    ts = getattr(value, "timestamp", None)
    if callable(ts):
        try:
            return datetime.fromtimestamp(ts(), tz=timezone.utc).isoformat()
        except Exception:
            return None
    if isinstance(value, str):
        return value
    return None


def _session_timer_fields(sess: dict) -> dict:
    """FIX 17: timer resync payload — frontend resyncs its countdown from
    these backend values, never from a local guess."""
    max_sec = int(sess.get("maxSeconds") or sess.get("max_duration_sec") or 0)
    started = _iso_dt(sess.get("created_at"))
    now_iso = datetime.now(timezone.utc).isoformat()
    elapsed = None
    remaining = None
    try:
        age = db._session_age_sec(sess) if hasattr(db, "_session_age_sec") else None
        if age is not None and max_sec:
            elapsed = max(0, int(age))
            remaining = max(0, int(max_sec - age))
    except Exception:
        pass
    return {"maxSeconds": max_sec, "max_duration_sec": max_sec,
            "started_at": started, "server_now": now_iso,
            "elapsed_sec": elapsed, "remaining_sec": remaining}


def _rate_check(key: str, limit: int, window_sec: int = _RATE_WINDOW_SEC) -> tuple[bool, int]:
    """Sliding-window check. Returns (allowed, retry_after_sec). Records the
    hit when allowed. Prunes stale timestamps on every call."""
    now = time.time()
    with _rate_lock:
        stamps = [t for t in _rate_buckets.get(key, []) if now - t < window_sec]
        if len(stamps) >= limit:
            retry = max(1, int(window_sec - (now - stamps[0]))) if stamps else 1
            _rate_buckets[key] = stamps
            return False, retry
        stamps.append(now)
        _rate_buckets[key] = stamps
        if len(_rate_buckets) > 10000:  # bound memory: drop fully-stale buckets
            for k in [k for k, v in _rate_buckets.items()
                      if not v or now - v[-1] >= window_sec][:1000]:
                _rate_buckets.pop(k, None)
        return True, 0


def _require_rate(key: str, limit: int):
    """Endpoint-level check: raises 429 + Retry-After when over budget."""
    allowed, retry = _rate_check(key, limit)
    if not allowed:
        raise HTTPException(429, "Too many requests — slow down.",
                            headers={"Retry-After": str(retry)})


def _rate_limit_per_min_default() -> int:
    try:
        return int(os.environ.get("RATE_LIMIT_PER_IP_PER_MIN", "120") or 120)
    except ValueError:
        return 120


@app.middleware("http")
async def _per_ip_rate_limit(request: Request, call_next):
    """Per-IP middleware: default 120/min. /health exempt (uptime checks
    must never 429). Returns 429 + Retry-After on excess."""
    if request.url.path == "/health":
        return await call_next(request)
    allowed, retry = _rate_check(f"ip:{_client_ip(request)}", _rate_limit_per_min_default())
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests — slow down."},
            headers={"Retry-After": str(retry)},
        )
    return await call_next(request)


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
    user_id: str = Field(..., min_length=1, description="Telegram user id (string)")
    test_type: str = Field(..., pattern="^(academic|business|general)$")
    plan: str = Field(..., pattern="^(free_trial|free_mini|paid)$")
    affiliate_code: str | None = None
    background_summary: str | None = Field(None, max_length=2000,
        description="Optional applicant background for sharper examiner questions")
    initData: str = Field(..., min_length=1,
        description="REQUIRED raw Telegram WebApp initData. It is HMAC-verified "
        "BEFORE any credit is consumed, its user id must equal user_id, and a "
        "WS session token (ws_token) is minted from it. A bare user_id with no "
        "initData is unsigned identity and is rejected — the backend trusts "
        "only HMAC-verified initData / server-issued tokens.")
    idempotency_key: str | None = Field(None, max_length=128,
        description="FIX 13: client-generated idempotency key (uuid). A retried "
        "start with the same (user_id, idempotency_key) returns the original "
        "session instead of charging a second credit.")


class SessionStartResp(BaseModel):
    session_id: str
    ephemeral_token: str = ""  # legacy field, unused — voice WS is fully server-driven
    expires_in_sec: int = 0
    max_duration_sec: int
    maxSeconds: int = 0  # FIX 17: timer alias — frontend resyncs from this
    ws_url_path: str
    ws_token: str = ""  # server-issued WS session token ("" when no initData was verified)


class SessionEndReq(BaseModel):
    session_id: str
    initData: str | None = Field(None, description="FIX 4: HMAC-verified initData "
        "for the session owner (or ws_token). One of the two is REQUIRED.")
    ws_token: str | None = Field(None, description="FIX 4: server-issued WS "
        "session token for this session (authoritative when present).")


class StarsInvoiceReq(BaseModel):
    user_id: str
    affiliate_code: str | None = None
    bundle_id: str | None = None  # reality | serious | panic — REQUIRED (None → 400)
    initData: str | None = Field(None, description="FIX 4: REQUIRED HMAC-verified "
        "initData for user_id. Also accepted via X-Telegram-Init-Data header. "
        "Unknown user_id 404s — never auto-provisioned.")


class RefundAbandonedReq(BaseModel):
    session_id: str
    initData: str | None = Field(None, description="FIX 14: HMAC-verified initData "
        "for the session owner (or ws_token). One of the two is REQUIRED.")
    ws_token: str | None = Field(None, description="FIX 14: server-issued WS "
        "session token for this session.")


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
    # Credit contract (frontend reads these; all initialized by create_user,
    # backfilled for legacy users by ensure_credit_fields — never a fresh trial).
    free_mini_used: bool = False
    interview_credits: int = 0
    trial_credits: int = 1
    trial_started_at: int = 0  # ms-epoch int, NOT a Timestamp
    credited_charges: list[str] = []


# ---------- Routes ----------

@app.get("/health")
async def health():
    # FIX 25: version/commit deploy parity — never crashes when unset.
    version, commit = _app_version()
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat(),
            "version": version, "commit": commit}


@app.get("/test-keys")
async def test_keys(request: Request):
    """Diagnostic: test Groq and Deepgram API keys.
    Admin-gated: 404 unless ADMIN_API_KEY is set, then requires a matching
    X-Admin-Key (constant-time compare). Rate-limited: 2/IP/min on top of
    the global per-IP budget — each call burns Groq + Deepgram quota.
    FIX 10: generic codes only — NO model list, NO upstream status codes,
    NO upstream error bodies, NO exception text. Provider internals are
    logged server-side only. Response shape:
    {"ok": bool, "groq": "ok"|"unavailable"|"missing_key",
     "deepgram": ..., "deepgram_stt": ...}."""
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(404, "Not found")
    provided = request.headers.get("X-Admin-Key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(401, "Invalid admin key")
    _require_rate(f"ip-test-keys:{_client_ip(request)}", 2)
    import httpx
    groq_key = os.environ.get("GROQ_API_KEY", "")
    dg_key = os.environ.get("DEEPGRAM_API_KEY", "")
    # Test Groq (chat completion only — never list models).
    if not groq_key:
        groq = "missing_key"
    else:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={"model": "openai/gpt-oss-20b",
                          "messages": [{"role": "user", "content": "Say hi"}],
                          "max_tokens": 5},
                )
            groq = "ok" if r.status_code == 200 else "unavailable"
            if r.status_code != 200:
                log.warning("test-keys: groq probe failed (status withheld from client)")
        except Exception:
            log.exception("test-keys: groq probe exception (withheld from client)")
            groq = "unavailable"
    # Test Deepgram TTS.
    if not dg_key:
        dg = "missing_key"
    else:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    "https://api.deepgram.com/v1/speak?model=aura-asteria-en",
                    headers={"Authorization": f"Token {dg_key}", "Content-Type": "application/json"},
                    json={"text": "hi"},
                )
            dg = "ok" if r.status_code == 200 else "unavailable"
            if r.status_code != 200:
                log.warning("test-keys: deepgram TTS probe failed (status withheld from client)")
        except Exception:
            log.exception("test-keys: deepgram TTS probe exception (withheld from client)")
            dg = "unavailable"
    # Test Deepgram STT WebSocket (the actual voice pipeline connection).
    if not dg_key:
        dg_stt = "missing_key"
    else:
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
            dg_stt = "ok"
        except Exception:
            log.exception("test-keys: deepgram STT probe exception (withheld from client)")
            dg_stt = "unavailable"
    ok = (groq == "ok" and dg == "ok" and dg_stt == "ok")
    return {"ok": ok, "groq": groq, "deepgram": dg, "deepgram_stt": dg_stt}


@app.post("/api/auth/telegram", response_model=TelegramAuthResp)
async def telegram_auth(req: TelegramAuthReq, request: Request):
    """
    Step 2-4 of Telegram auto-login:
    1. Verify initData HMAC-SHA256 signature using bot token (proves user is real)
    2. Check Firestore for existing user
    3. Create new user doc if first time ("Free 2-minute test available")
    4. Return verified user profile
    FIX 11: per-entity rate limits — 20/IP/min + 10/user/min on top of the
    global per-IP middleware budget.
    """
    # Per-IP bucket first (before any verification work).
    _require_rate(f"ip-auth:{_client_ip(request)}", 20)
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")

    # Step 2-3: Cryptographic verification
    tg_user = verify_telegram_init_data(req.initData, TELEGRAM_BOT_TOKEN)

    telegram_id = str(tg_user["id"])
    _require_rate(f"entity-auth:{telegram_id}", 10)

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
        # Legacy backfill: fill missing credit fields WITHOUT granting a
        # fresh trial (trial_credits derived from free_trial_used,
        # trial_started_at anchored to created_at).
        user = db.ensure_credit_fields(telegram_id) or user

    return TelegramAuthResp(
        telegram_id=telegram_id,
        first_name=tg_user.get("first_name", ""),
        username=tg_user.get("username", ""),
        is_new=is_new,
        free_trial_used=bool(user.get("free_trial_used")),
        paid=bool(user.get("paid")),
        free_mini_used=bool(user.get("free_mini_used", False)),
        interview_credits=int(user.get("interview_credits", 0) or 0),
        trial_credits=int(user.get("trial_credits", 0) or 0),
        trial_started_at=int(user.get("trial_started_at", 0) or 0),
        credited_charges=list(user.get("credited_charges") or []),
    )


@app.post("/session/start", response_model=SessionStartResp)
async def session_start(req: SessionStartReq, request: Request):
    # Entity rate limits: 10/IP/min + 3/user/min (on top of the global
    # per-IP middleware budget). Checked before any credit moves.
    _require_rate(f"ip-session-start:{_client_ip(request)}", 10)
    _require_rate(f"entity-session-start:{req.user_id}", 3)
    # Null-user guard: whitespace-only ids never reach Firestore (empty
    # string is already rejected by the pydantic min_length=1 → 422).
    if not req.user_id or not req.user_id.strip():
        raise HTTPException(400, "user_id must not be empty")

    # UNSIGNED IDENTITY DELETED: initData is REQUIRED (pydantic) and is
    # HMAC-verified BEFORE any credit is consumed or any session is
    # created. A bad initData (or one for a different user) must never
    # burn a credit — and a token is minted ONLY from verified initData,
    # so an attacker who starts a session for someone else's user_id gets
    # a session they cannot use on the WS (they do not hold the victim's
    # initData, and the WS re-checks ownership).
    tg_user = verify_telegram_init_data(req.initData, TELEGRAM_BOT_TOKEN)
    if str(tg_user.get("id")) != str(req.user_id):
        raise HTTPException(403, "initData user does not match user_id")

    user = db.get_user(req.user_id)
    if not user:
        # Identity was verified above, so creating the user for the
        # verified id is safe (this is the verified-identity path, not
        # the deleted bare-user_id path).
        user = db.create_user(req.user_id, affiliate_code=req.affiliate_code)
    else:
        user = db.ensure_credit_fields(req.user_id) or user

    # FIX 25 (re-derived): a second start must not overlap a live unscored
    # session. Return 409 SESSION_ACTIVE with the existing session_id —
    # BEFORE any credit/claim is consumed, so a Setup retry never burns
    # a credit or creates a duplicate. Frontend resumes the existing
    # session (WS accepts fresh ?initData=, no new token needed).
    active = db.find_active_unscored_session(req.user_id)
    if active:
        raise HTTPException(
            409,
            f"SESSION_ACTIVE session_id={active['session_id']} — "
            "an interview is already in progress for this user; "
            "resume it instead of starting a new one.")

    # FIX 13 (idempotency): a retried start with the same key returns the
    # ORIGINAL session — no second charge, no duplicate session. Checked
    # BEFORE any credit/claim moves and before the ended-unscored block
    # (a replay is not a new start, so it must not 409 on its own past).
    if req.idempotency_key:
        prior = db.get_session_by_idempotency_key(req.user_id, req.idempotency_key)
        if prior:
            prior_max = int(prior.get("maxSeconds") or prior.get("max_duration_sec") or 0)
            prior_token = mint_session_token(prior["session_id"], req.user_id,
                                             ttl_sec=prior_max + 600) if prior_max else ""
            prior_path = (f"/ws/voice/{prior['session_id']}?token={prior_token}"
                          if prior_token else f"/ws/voice/{prior['session_id']}")
            log.info("session_start idempotent replay user %s key=%s session=%s",
                     req.user_id, req.idempotency_key, prior["session_id"])
            return SessionStartResp(
                session_id=prior["session_id"],
                ephemeral_token="",
                expires_in_sec=0,
                max_duration_sec=prior_max,
                maxSeconds=prior_max,
                ws_url_path=prior_path,
                ws_token=prior_token,
            )

    # FIX 13 (ended-but-unscored): the previous session ended but was never
    # scored — starting a new one would charge AGAIN for an interview the
    # user never got scored. 409 with a DISTINCT code the frontend routes
    # to SCORE ME NOW (score the old session) or the refund-on-abandon
    # route — never a silent re-charge.
    ended_unscored = db.find_ended_unscored_session(req.user_id)
    if ended_unscored:
        raise HTTPException(
            409,
            f"ENDED_UNSCORED_SCORE_NOW session_id={ended_unscored['session_id']} — "
            "your previous interview ended but was never scored; "
            "score it (SCORE ME NOW) or refund it instead of starting a new one.")

    consumed_credit = False
    claimed_mini = False
    if req.plan == "free_trial":
        # FIX 25 guard (verified: already existed) — 402 with a
        # machine-readable code prefix; human text kept after the colon.
        if user.get("free_trial_used"):
            raise HTTPException(
                402,
                "TRIAL_ALREADY_USED: Free trial already used — upgrade to continue")
        max_dur = FREE_TRIAL_MAX_SEC
    elif req.plan == "free_mini":
        # Atomic claim BEFORE session creation. If the session create
        # below fails, the claim is released so the user can retry.
        if not db.claim_free_mini(req.user_id):
            raise HTTPException(
                402,
                "FREE_MINI_ALREADY_USED: Free mini interview already used — "
                "use your free trial or upgrade to continue")
        claimed_mini = True
        max_dur = FREE_MINI_MAX_SEC
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
            idempotency_key=req.idempotency_key,
            credit_consumed=consumed_credit,
        )
    except Exception:
        # Session creation failed after we consumed a credit / claimed the
        # mini — refund/release it so a failed start never burns anything.
        if consumed_credit:
            try:
                db.add_interview_credits(req.user_id, 1)
            except Exception:
                log.exception("credit refund failed for user %s", req.user_id)
        if claimed_mini:
            try:
                db.release_free_mini(req.user_id)
            except Exception:
                log.exception("free_mini release failed for user %s", req.user_id)
        raise
    log.info("session %s started for user %s plan=%s", session_id, req.user_id, req.plan)
    # Mint the WS token ONLY from verified initData (see above). Token TTL =
    # session length + 10 min grace for reconnects; the WS re-checks expiry.
    ws_token = ""
    ws_url_path = f"/ws/voice/{session_id}"
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
        maxSeconds=max_dur,
        ws_url_path=ws_url_path,
        ws_token=ws_token,
    )


@app.post("/session/end")
async def session_end(req: SessionEndReq, request: Request):
    # FIX 4 + FIX 11: auth + owner check BEFORE any flush/scoring work, and
    # per-entity rate limits (per IP + per session + per owner). Unknown
    # session_id -> 404, never created; missing/invalid credential -> 401;
    # verified non-owner -> 403. No auto-provisioning of any kind.
    _require_rate(f"ip-session-end:{_client_ip(request)}", 20)
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    owner = str(sess.get("user_id", "") or "")
    if not owner:
        raise HTTPException(404, "Session not found")
    _require_rate(f"entity-session-end:{req.session_id}", 10)
    _require_rate(f"entity-session-end-user:{owner}", 10)
    _verify_session_owner(req.session_id, owner,
                          init_data=_extract_init_data(request, req.initData),
                          ws_token=(req.ws_token or "").strip())

    # FLUSH-BEFORE-SCORE (VOICE WAVE 47/50 + integration pass): the
    # transcript lives in the voice_proxy task's memory and only reaches
    # DB on finish / timeup / socket close. A manual finish calls this
    # endpoint while the WS is still open (or just closed), so scoring
    # the DB transcript as-is 409s on "empty" even though the user spoke.
    # Wait for the flush to COMPLETE — never break on any non-empty
    # transcript: a stale/partial transcript from earlier turns is not
    # proof the final turn's flush landed. wait_for_flush_completion
    # returns False immediately when no live session exists in this
    # process (other-worker / already-closed); then fall back to polling
    # DB for the close-then-score race, breaking only on score /
    # denied / ended_at (flush landed), never on transcript alone.
    # The canonical frontend path sends {"type":"finish"} and waits for
    # the {"type":"flushed"} ack before calling here, so in practice the
    # wait usually returns at once (flush already complete).
    try:
        request_session_flush(req.session_id)
    except Exception:
        log.exception("request_session_flush failed for %s", req.session_id)
    flushed = await wait_for_flush_completion(req.session_id, timeout_sec=8.0)
    if not flushed:
        for _ in range(16):  # 16 x 0.5s = up to 8s
            if sess.get("score") is not None or sess.get("denied"):
                break
            if sess.get("ended_at") is not None:
                break  # socket-close flush already landed; transcript is final
            await asyncio.sleep(0.5)
            sess = db.get_session(req.session_id)
            if not sess:
                raise HTTPException(404, "Session not found")
    else:
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
            **_session_timer_fields(sess),
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
            **_session_timer_fields(sess),
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
            **_session_timer_fields(sess),
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
                    **_session_timer_fields(sess),
                }
        raise HTTPException(409, "Scoring in progress — retry shortly")
    if claim == "missing":
        raise HTTPException(404, "Session not found")

    # Re-fetch: the flush may have landed while we were claiming, and the
    # `sess` above can be up to one poll-loop iteration stale.
    sess = db.get_session(req.session_id)
    if not sess:
        # Session vanished mid-claim — release the claim we just took, or
        # a retry would sit behind a stale in_progress flag until TTL.
        db.release_scoring(req.session_id)
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
    # Trial accounting: ONLY plan=free_trial marks the trial used.
    # free_mini is claimed atomically at /session/start (claim_free_mini),
    # so ending a free_mini session must NEVER mark the trial used — the
    # old conflation 402-looped every user whose first session was a mini.
    # Paid sessions touch no trial state.
    if sess.get("plan") == "free_trial":
        db.mark_trial_used(sess["user_id"])
    log.info("session %s scored: %s (%s) nervousness=%d", req.session_id,
             result["total"], result["verdict"], nervousness)
    return {"session_id": req.session_id, **result, **_session_timer_fields(sess)}


@app.get("/session/status/{session_id}")
async def session_status(session_id: str, request: Request):
    """FIX 17: timer resync read — returns maxSeconds/max_duration_sec +
    started_at/server_now/elapsed/remaining so the frontend can resync its
    countdown from the backend. Auth + owner check identical to
    /session/end (ws_token via ?token=, initData via header/query)."""
    _require_rate(f"ip-session-status:{_client_ip(request)}", 20)
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    owner = str(sess.get("user_id", "") or "")
    if not owner:
        raise HTTPException(404, "Session not found")
    _require_rate(f"entity-session-status:{session_id}", 10)
    _verify_session_owner(
        session_id, owner,
        init_data=_extract_init_data(request, None),
        ws_token=(request.query_params.get("token") or "").strip())
    return {"session_id": session_id, "plan": sess.get("plan"),
            "score": sess.get("score"), "verdict": sess.get("verdict"),
            "denied": bool(sess.get("denied")), "ended": sess.get("ended_at") is not None,
            **_session_timer_fields(sess)}


@app.post("/session/refund-abandoned")
async def session_refund_abandoned(req: RefundAbandonedReq, request: Request):
    """FIX 14 (main part): refund-on-abandon route. Sessions abandoned after
    credit burn but never started/scored get their credit back.

    Rules (enforced transactionally in db.refund_abandoned_session):
    - caller must own the session (ws_token or HMAC initData), else 401/403;
      unknown session_id -> 404, never created.
    - abandoned = unscored + not denied + not already refunded + (ended OR
      older than max_duration + grace). Live sessions -> 409 SESSION_ACTIVE.
    - paid + credit consumed -> +1 credit (Increment, same txn as the
      refunded=True flag). Replays return already_refunded (200) — never a
      double-refund. free_mini -> releases the mini claim. free_trial /
      unlimited-paid -> no_refund_needed (nothing was burned).
    - Per-entity rate limits: 20/IP/min + 5/session/min + 5/user/min."""
    _require_rate(f"ip-refund:{_client_ip(request)}", 20)
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    owner = str(sess.get("user_id", "") or "")
    if not owner:
        raise HTTPException(404, "Session not found")
    _require_rate(f"entity-refund:{req.session_id}", 5)
    _require_rate(f"entity-refund-user:{owner}", 5)
    _verify_session_owner(req.session_id, owner,
                          init_data=_extract_init_data(request, req.initData),
                          ws_token=(req.ws_token or "").strip())
    res = db.refund_abandoned_session(req.session_id)
    status = res.get("status")
    if status == "not_found":
        raise HTTPException(404, "Session not found")
    if status == "session_active":
        raise HTTPException(409, "SESSION_ACTIVE: session is still live — finish or score it first")
    if status in ("already_scored", "denied"):
        raise HTTPException(409, f"{status}: session cannot be refunded")
    if status == "not_abandoned":
        raise HTTPException(409, "NOT_ABANDONED: session is not abandoned yet")
    # refunded / already_refunded / no_refund_needed -> 200 (idempotent).
    return {"ok": True, **res, **_session_timer_fields(db.get_session(req.session_id) or sess)}


class FeedbackAudioReq(BaseModel):
    session_id: str
    initData: str | None = Field(None, description="FIX 4: HMAC-verified initData "
        "for the session owner (or ws_token). One of the two is REQUIRED.")
    ws_token: str | None = Field(None, description="FIX 4: server-issued WS "
        "session token for this session.")


@app.post("/session/feedback-audio")
async def session_feedback_audio(req: FeedbackAudioReq, request: Request):
    """Synthesize the session verdict + feedback with the FRIENDLY 'Visa Coach'
    voice (aura-luna-en). Returns raw PCM16 24kHz mono audio bytes.
    The interview uses the strict officer voice; feedback uses the warm coach.
    FIX 4: auth + owner check (ws_token or HMAC initData) — unknown session
    404s, never created. FIX 11: per-entity rate limits (per IP + per
    session + per owner)."""
    from fastapi.responses import Response
    from realtime_proxy import deepgram_tts
    _require_rate(f"ip-feedback-audio:{_client_ip(request)}", 20)
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    owner = str(sess.get("user_id", "") or "")
    if not owner:
        raise HTTPException(404, "Session not found")
    _require_rate(f"entity-feedback-audio:{req.session_id}", 10)
    _require_rate(f"entity-feedback-audio-user:{owner}", 10)
    _verify_session_owner(req.session_id, owner,
                          init_data=_extract_init_data(request, req.initData),
                          ws_token=(req.ws_token or "").strip())
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
async def stars_invoice(req: StarsInvoiceReq, request: Request):
    """Create a Telegram Stars invoice link. bundle_id is REQUIRED
    (reality | serious | panic). The legacy bundle-less 750-Star flow is
    deleted: bundle_id None → 400, never an unlimited-access invoice.
    FIX 4: caller must prove ownership with HMAC-verified initData
    (body or X-Telegram-Init-Data header). Unknown user_id -> 404, NEVER
    auto-provisioned. FIX 11: per-entity rate limits (per IP + per user)."""
    _require_rate(f"ip-stars-invoice:{_client_ip(request)}", 10)
    _require_rate(f"entity-stars-invoice:{req.user_id}", 5)
    if not req.bundle_id:
        raise HTTPException(400, "bundle_id is required — legacy 750-Star flow removed")
    bundle = BUNDLES.get(req.bundle_id)
    if not bundle:
        raise HTTPException(400, f"Unknown bundle_id: {req.bundle_id}")
    init_data = _extract_init_data(request, req.initData)
    if not init_data:
        raise HTTPException(401, "Missing initData: HMAC-verified initData required")
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")
    tg_user = verify_telegram_init_data(init_data, TELEGRAM_BOT_TOKEN)
    if str(tg_user.get("id")) != str(req.user_id):
        raise HTTPException(403, "initData user does not match user_id")
    user = db.get_user(req.user_id)
    if not user:
        # FIX 4: never auto-provision an arbitrary user_id from an invoice
        # request — the account must already exist via verified auth.
        raise HTTPException(404, "User not found — authenticate first via /api/auth/telegram")

    stars = bundle["stars"]
    title = bundle["title"]
    description = bundle["description"]
    label = bundle["name"]
    payload_token = f"{req.user_id}:{bundle['id']}:{uuid.uuid4().hex[:12]}"

    db.record_stars_invoice(payload_token, req.user_id, req.affiliate_code,
                            bundle_id=bundle["id"])

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
        raise HTTPException(502, "Telegram payment error")
    return {"invoice_url": data["result"], "payload": payload_token,
            "stars": stars, "bundle_id": bundle["id"],
            "interviews": bundle["interviews"]}


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
    """Handle Telegram updates: /start welcome + pre_checkout_query + successful_payment.
    ONE atomic grant path: successful_payment is granted ONLY by
    db.grant_stars_payment() in a single transaction (duplicate charge +
    invoice exists + invoice unused + payload match + currency XTR +
    amount == expected bundle stars). Unknown payloads are rejected
    outright — there is NO mark_paid fallback and NO legacy path."""
    # SECURITY: fail-closed — reject all webhooks if no secret configured,
    # rather than accepting forged payments. Constant-time compare.
    if not TELEGRAM_WEBHOOK_SECRET:
        log.error("TELEGRAM_WEBHOOK_SECRET not set — rejecting webhook")
        raise HTTPException(503, "Webhook not configured")
    got = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not got or not hmac.compare_digest(got, TELEGRAM_WEBHOOK_SECRET):
        raise HTTPException(401, "Bad webhook secret")

    update = await request.json()
    # FIX 11: per-entity rate limits on the webhook (per IP + per charge +
    # per payload). Retries of the same charge/payload are capped so a
    # replay storm cannot hammer the grant transaction.
    _require_rate(f"ip-webhook:{_client_ip(request)}", 30)
    _webhook_entity = ""
    try:
        _pcq = update.get("pre_checkout_query") or {}
        _sp = (update.get("message") or {}).get("successful_payment") or {}
        _webhook_entity = (_sp.get("telegram_payment_charge_id") or ""
                           or _sp.get("invoice_payload") or ""
                           or _pcq.get("invoice_payload") or "")
    except Exception:
        _webhook_entity = ""
    if _webhook_entity:
        _require_rate(f"entity-webhook:{_webhook_entity}", 10)

    pcq = update.get("pre_checkout_query")
    if pcq:
        # Validate BEFORE approving: the invoice must exist, be unused,
        # belong to a known bundle, and the amount/currency must match
        # the bundle exactly. Anything else → ok=False (no charge happens).
        # Pre-checkout never writes anything (checkout hasn't completed).
        payload_token = pcq.get("invoice_payload", "") or ""
        inv = db.get_stars_invoice(payload_token) if payload_token else None
        bundle = BUNDLES.get((inv or {}).get("bundle_id") or "") if inv else None
        ok, err = False, "Unknown invoice"
        if inv is None:
            err = "Unknown invoice"
        elif inv.get("used"):
            err = "Invoice already used"
        elif bundle is None:
            err = "Unknown bundle for invoice"
        elif pcq.get("currency") != "XTR":
            err = "Currency must be XTR"
        elif int(pcq.get("total_amount") or 0) != bundle["stars"]:
            err = "Amount does not match bundle price"
        else:
            ok, err = True, ""
        await tg_api("answerPreCheckoutQuery", {
            "pre_checkout_query_id": pcq["id"],
            "ok": ok,
            **({"error_message": err} if not ok else {}),
        })
        if not ok:
            log.warning("pre_checkout REJECTED payload=%s reason=%s amount=%s currency=%s",
                        payload_token, err, pcq.get("total_amount"), pcq.get("currency"))
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
        payload_token = sp.get("invoice_payload", "") or ""
        # Unknown payload → rejected outright. NEVER a mark_paid fallback:
        # an attacker can send any payload string; granting unlimited
        # access for it was the old 0-Star / unknown-payload hole.
        inv = db.get_stars_invoice(payload_token) if payload_token else None
        if inv is None:
            log.warning("successful_payment REJECTED unknown payload=%s charge=%s",
                        payload_token, charge_id)
            return {"ok": True}
        # Bundle resolved from the STORED invoice, never from client input.
        bundle = BUNDLES.get(inv.get("bundle_id") or "")
        if bundle is None:
            log.warning("successful_payment REJECTED unknown/legacy bundle=%s charge=%s payload=%s",
                        inv.get("bundle_id"), charge_id, payload_token)
            return {"ok": True}
        # Single atomic grant: duplicate + invoice + payload + currency +
        # amount checks, then payment record + invoice used + credits +
        # credited_charges, all in ONE transaction. Non-granted = nothing
        # written, nothing granted by any other path.
        res = db.grant_stars_payment(
            charge_id=charge_id,
            payload_token=payload_token,
            currency=sp.get("currency", "") or "",
            total_amount=int(sp.get("total_amount", 0) or 0),
            expected_stars=bundle["stars"],
            credit_count=bundle["interviews"],
        )
        status = res.get("status")
        if status == "granted":
            log.info("user %s granted +%d credits (%s Stars, charge %s, new_total=%s)",
                     res.get("user_id"), res.get("credits_granted"),
                     sp.get("total_amount"), charge_id, res.get("new_total"))
        elif status == "duplicate":
            log.info("duplicate webhook for charge %s — credits not re-granted", charge_id)
        elif status in ("invoice_used",):
            log.info("webhook for already-used invoice charge %s — not re-granted", charge_id)
        else:
            # Mismatch (amount/currency/payload): Telegram already charged
            # the user, but we do NOT grant on a mismatch — this is a
            # support case (log carries the charge id for manual review).
            # Return {"ok": True} so Telegram does not retry a payment
            # that can never validate.
            log.warning("successful_payment NOT GRANTED status=%s charge=%s payload=%s "
                        "amount=%s currency=%s expected_stars=%s",
                        status, charge_id, payload_token,
                        sp.get("total_amount"), sp.get("currency"), bundle["stars"])
    return {"ok": True}


@app.get("/payment/credits/{user_id}")
async def payment_credits(user_id: str, request: Request):
    """Return the user's remaining interview credits (single source of truth).
    FIX 4: caller must own the id — HMAC-verified initData (header
    X-Telegram-Init-Data or ?initData=) whose user equals user_id, else
    401/403. Unknown user_id -> 404, never auto-provisioned. FIX 11:
    per-entity rate limits (per IP + per user)."""
    _require_rate(f"ip-credits:{_client_ip(request)}", 20)
    _require_rate(f"entity-credits:{user_id}", 10)
    init_data = _extract_init_data(request, None)
    if not init_data:
        raise HTTPException(401, "Missing initData: HMAC-verified initData required")
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")
    tg_user = verify_telegram_init_data(init_data, TELEGRAM_BOT_TOKEN)
    if str(tg_user.get("id")) != str(user_id):
        raise HTTPException(403, "initData user does not match user_id")
    if db.get_user(user_id) is None:
        raise HTTPException(404, "User not found")
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
    initData: str | None = Field(None, description="FIX 3: REQUIRED HMAC-verified "
        "initData for user_id (claimant). Also accepted via "
        "X-Telegram-Init-Data header. Claimant must already exist — never "
        "auto-minted.")


@app.post("/referral/claim")
async def referral_claim(req: ReferralClaimReq, request: Request):
    """Claim a referral bonus: +1 free interview for the new user AND +1 for
    the referrer. One claim per user ever; self-referral rejected.
    Rate-limited: 10/IP/min + 3/user/min (referral farming throttle).
    FIX 3: requires verified initData/token (HMAC-verified initData whose
    user equals user_id); rejects a non-existent claimant with 404 and
    NEVER auto-mints a user doc for a claimant that doesn't exist."""
    _require_rate(f"ip-referral:{_client_ip(request)}", 10)
    _require_rate(f"entity-referral:{req.user_id}", 3)
    if not req.user_id or not req.referrer_id:
        raise HTTPException(400, "user_id and referrer_id are required")
    if req.user_id == req.referrer_id:
        raise HTTPException(400, "self_referral")
    init_data = _extract_init_data(request, req.initData)
    if not init_data:
        raise HTTPException(401, "Missing initData: HMAC-verified initData required")
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")
    tg_user = verify_telegram_init_data(init_data, TELEGRAM_BOT_TOKEN)
    if str(tg_user.get("id")) != str(req.user_id):
        raise HTTPException(403, "initData user does not match user_id")
    if db.get_user(req.user_id) is None:
        raise HTTPException(404, "Claimant not found — authenticate first via /api/auth/telegram")
    try:
        res = db.claim_referral(req.user_id, req.referrer_id)
    except ValueError as e:
        msg = str(e)
        if msg in ("claimant_not_found", "referrer_not_found"):
            raise HTTPException(404, msg)
        raise HTTPException(400, msg)
    return {"ok": True, **res}


@app.get("/referral/stats/{user_id}")
async def referral_stats(user_id: str, request: Request):
    """Referral stats for the invite UI: friends joined + free interviews earned.
    FIX 3: requires verified initData (header or ?initData=) for this
    user_id; non-existent user -> 404. FIX 11: per-entity rate limits."""
    _require_rate(f"ip-referral-stats:{_client_ip(request)}", 20)
    _require_rate(f"entity-referral-stats:{user_id}", 10)
    init_data = _extract_init_data(request, None)
    if not init_data:
        raise HTTPException(401, "Missing initData: HMAC-verified initData required")
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "Telegram bot not configured")
    tg_user = verify_telegram_init_data(init_data, TELEGRAM_BOT_TOKEN)
    if str(tg_user.get("id")) != str(user_id):
        raise HTTPException(403, "initData user does not match user_id")
    if db.get_user(user_id) is None:
        raise HTTPException(404, "User not found")
    return db.get_referral_stats(user_id)


@app.get("/affiliate/stats/{code}")
async def affiliate_stats(code: str):
    stats = db.get_affiliate_stats(code)
    if not stats:
        raise HTTPException(404, "Affiliate not found")
    return stats


# ---------- MockGrill Support: user feedback / bug reports ----------
# Users report bugs/errors/suggestions from the Mini App (Settings ->
# MockGrill Support). Reports land in the Firestore `feedback` collection.
# The assistant reads them via GET /feedback/list (admin-gated) and fixes
# what came in — no report is ever shown to other users.

FEEDBACK_TYPES = {"bug", "error", "suggestion", "other"}
FEEDBACK_MAX_PER_HOUR = 5  # per user_id, in-process sliding window
_feedback_rate: dict[str, list[float]] = {}  # user_id -> [epoch seconds]


def _feedback_rate_check(user_id: str) -> tuple[bool, int]:
    """Sliding-window rate limit: max FEEDBACK_MAX_PER_HOUR reports/hour/user.
    Returns (allowed, retry_after_sec). Records the hit when allowed.
    FIX 11: the 429 path returns Retry-After (see submit_feedback)."""
    now = time.time()
    stamps = [s for s in _feedback_rate.get(user_id, []) if now - s < 3600]
    if len(stamps) >= FEEDBACK_MAX_PER_HOUR:
        _feedback_rate[user_id] = stamps
        retry = max(1, int(3600 - (now - stamps[0]))) if stamps else 1
        return False, retry
    stamps.append(now)
    _feedback_rate[user_id] = stamps
    return True, 0


def _feedback_rate_ok(user_id: str) -> bool:
    """Back-compat wrapper — prefer _feedback_rate_check for Retry-After."""
    allowed, _ = _feedback_rate_check(user_id)
    return allowed


class FeedbackReq(BaseModel):
    user_id: str
    type: str
    message: str
    screen: str | None = None
    session_id: str | None = None
    app_version: str | None = None
    user_agent: str | None = None
    first_name: str | None = None
    username: str | None = None


@app.post("/feedback")
async def submit_feedback(req: FeedbackReq):
    """Accept one feedback / bug report from the Mini App."""
    if not req.user_id or not str(req.user_id).strip():
        raise HTTPException(400, "user_id is required")
    if req.type not in FEEDBACK_TYPES:
        raise HTTPException(400, "type must be one of: bug, error, suggestion, other")
    msg = (req.message or "").strip()
    if len(msg) < 10:
        raise HTTPException(400, "message must be at least 10 characters")
    if len(msg) > 2000:
        raise HTTPException(400, "message must be at most 2000 characters")
    _fb_allowed, _fb_retry = _feedback_rate_check(str(req.user_id))
    if not _fb_allowed:
        raise HTTPException(429, "Too many reports — please try again later",
                            headers={"Retry-After": str(_fb_retry)})
    rec = db.save_feedback(
        user_id=str(req.user_id),
        type=req.type,
        message=msg,
        screen=req.screen,
        session_id=req.session_id,
        app_version=req.app_version,
        user_agent=req.user_agent,
        first_name=req.first_name or "",
        username=req.username or "",
    )
    return {"ok": True, "id": rec["id"]}


@app.get("/feedback/list")
async def feedback_list(request: Request, status: str | None = None):
    """List feedback reports, newest first. Admin-gated: requires the
    X-Admin-Key header to match ADMIN_API_KEY. 404 when no admin key is
    configured (endpoint stays hidden), 401 on mismatch.
    ?status=new|read|fixed filters; omitted returns all."""
    import hmac
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(404, "Not found")
    provided = request.headers.get("X-Admin-Key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(401, "Invalid admin key")
    if status is not None and status not in ("new", "read", "fixed"):
        raise HTTPException(400, "status must be one of: new, read, fixed")
    return {"ok": True, "feedback": db.list_feedback(status=status, limit=100)}


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
async def notify_daily_grilly(req: DailyGrillyReq, request: Request):
    """Daily retention push — requires CRON_SECRET env var to match req.secret
    (constant-time compare). Prevents anyone from triggering mass Telegram
    messages. Idempotent per date: daily_grilly_sends/{date} is claimed
    before sending and marked sent after; a duplicate cron run the same
    day skips sending (returns already_sent).
    FIX 11: per-entity rate limits (per IP + per date).
    FIX 24: per-recipient sent records (daily_grilly_sends/{date}/
    recipients/{telegram_id}); a retry skips already-sent recipients.
    Catastrophic failure does NOT delete the whole claim — the day doc is
    marked 'partial' with per-recipient records intact, so a retry resumes
    instead of re-messaging everyone or losing progress."""
    import os
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or not req.secret or not hmac.compare_digest(req.secret, expected):
        raise HTTPException(403, "Invalid or missing cron secret")
    from datetime import date
    today = date.today().isoformat()
    _require_rate(f"ip-daily-grilly:{_client_ip(request)}", 5)
    _require_rate(f"entity-daily-grilly:{today}", 5)
    if req.dry_run:
        targets = db.get_daily_push_targets(limit=max(1, min(req.limit, 500)))
        return {"ok": True, "dry_run": True, "targets": len(targets),
                "sample": targets[:3], "date": today}
    if not db.claim_daily_grilly_send(today):
        log.info("daily-grilly: %s already sent/claimed — skipping duplicate run", today)
        return {"ok": True, "already_sent": True, "date": today}
    # Resume counts from a prior partial send (claim preserves them).
    _prior = db.get_daily_grilly_send(today) or {}
    sent, failed, skipped = int(_prior.get("sent") or 0), int(_prior.get("failed") or 0), 0
    q = DAILY_GRILLY_QUESTIONS[date.today().timetuple().tm_yday % len(DAILY_GRILLY_QUESTIONS)]
    try:
        targets = db.get_daily_push_targets(limit=max(1, min(req.limit, 500)))
        # Rotate the question daily — deterministic so a retry same-day is idempotent
        for t in targets:
            chat_id = t.get("telegram_id")
            # FIX 24: skip recipients who already got today's push.
            if chat_id and db.has_daily_grilly_recipient(today, str(chat_id)):
                skipped += 1
                continue
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
                # Record per-recipient success BEFORE counting, so a crash
                # immediately after still leaves this recipient marked.
                if chat_id:
                    db.mark_daily_grilly_recipient_sent(today, str(chat_id))
                sent += 1
            except Exception as e:
                log.warning("daily-grilly failed for %s: %s", chat_id, str(e)[:120])
                failed += 1
        log.info("daily-grilly: sent=%d failed=%d skipped=%d question=%s", sent, failed, skipped, q[:40])
    except Exception:
        # Catastrophic failure — mark PARTIAL (never delete the claim).
        # Per-recipient records + counts survive, so a retry resumes and
        # skips already-sent recipients. Return a partial payload (200)
        # instead of raising, so the cron run is visibly partial, not lost.
        log.exception("daily-grilly catastrophic failure — marking partial for %s", today)
        try:
            db.mark_daily_grilly_partial(today, sent, failed)
        except Exception:
            log.exception("daily-grilly: mark_partial failed for %s", today)
        return {"ok": False, "partial": True, "sent": sent, "failed": failed,
                "skipped": skipped, "question": q, "date": today}
    db.mark_daily_grilly_sent(today, sent, failed)
    return {"ok": True, "sent": sent, "failed": failed, "skipped": skipped,
            "question": q, "date": today}


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






"""
Firestore layer for MockGrill.

Collections:
  users      {telegram_id(str, doc id), paid:bool, free_trial_used:bool,
              free_mini_used:bool, interview_credits:int, trial_credits:int,
              trial_started_at:int (ms epoch), credited_charges:[charge_id],
              affiliate_code:str|None, created_at, paid_at}
  sessions   {user_id, test_type, plan, max_duration_sec, transcript:[{role,text}],
              score:int|None, verdict, breakdown, weaknesses, feedback,
              duration_sec, affiliate_code, created_at, ended_at,
              scoring_in_progress:bool, claimed_at:int (epoch sec, scoring TTL)}
  affiliates {code(doc id), name, revenue_share (0-1), total_earned_stars, created_at}
  payments   {payment_id(doc id = Telegram charge id), user_id, amount_stars,
              status, affiliate_code, created_at}
  stars_invoices {payload(doc id), user_id, affiliate_code, used:bool,
              created_at, used_at}
  daily_grilly_sends {date(doc id = YYYY-MM-DD), status: claimed|sent,
              claimed_at, sent_at, sent, failed}
  feedback    {id(doc id), user_id, type, message, screen, session_id,
              app_version, user_agent, first_name, username, status,
              created_at}

Auth: firebase-admin with service account JSON whose path is in
FIREBASE_CRED_PATH. Lazily initialized on first use.
"""
import logging
import os
import time
import uuid
from datetime import datetime, timezone

log = logging.getLogger("englishtest.db")

_db = None
_db_available = None  # None=unknown, True=ok, False=fallback mode
_mem_users = {}
_mem_sessions = {}


def _now():
    return datetime.now(timezone.utc)


def is_db_available() -> bool:
    """True if Firestore is configured and reachable. Uses in-memory fallback otherwise."""
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        get_db()
        _db_available = True
    except Exception as e:
        log.warning("Firestore unavailable, using in-memory fallback: %s", str(e)[:100])
        _db_available = False
    return _db_available


def get_db():
    global _db
    if _db is not None:
        return _db
    import firebase_admin
    from firebase_admin import credentials, firestore

    # Try JSON content from env var first (easier for API-based setup)
    cred_json = os.environ.get("FIREBASE_CRED_JSON", "")
    if cred_json:
        import json
        try:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
        except Exception as e:
            raise RuntimeError(f"FIREBASE_CRED_JSON invalid: {str(e)[:100]}")
    else:
        cred_path = os.environ.get("FIREBASE_CRED_PATH", "")
        if not cred_path or not os.path.exists(cred_path):
            raise RuntimeError("FIREBASE_CRED_PATH not set or file missing")
        cred = credentials.Certificate(cred_path)
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred)
    _db = firestore.client()
    return _db


# ---------- users ----------

def get_user(user_id: str) -> dict | None:
    if not is_db_available():
        return _mem_users.get(str(user_id))
    doc = get_db().collection("users").document(str(user_id)).get()
    return doc.to_dict() if doc.exists else None


def create_user(user_id: str, affiliate_code: str | None = None,
                first_name: str = "", username: str = "") -> dict:
    # Credit fields are initialized here for every new user. trial_started_at
    # is a ms-epoch INT (not a Firestore Timestamp) so auth responses and the
    # frontend can compare it with Date.now() with no conversion.
    data = {
        "telegram_id": str(user_id),
        "first_name": first_name,
        "username": username,
        "paid": False,
        "free_trial_used": False,
        "free_mini_used": False,
        "interview_credits": 0,
        "trial_credits": 1,
        "trial_started_at": int(time.time() * 1000),
        "credited_charges": [],
        "affiliate_code": affiliate_code,
        "created_at": _now(),
        "paid_at": None,
    }
    if not is_db_available():
        _mem_users[str(user_id)] = data
        return data
    get_db().collection("users").document(str(user_id)).set(data)
    return data


def _trial_started_ms_from_created(created_at) -> int:
    """Legacy backfill: trial_started_at anchored to the user's created_at,
    NEVER to now — a legacy user must not get a fresh trial window."""
    if isinstance(created_at, datetime):
        return int(created_at.timestamp() * 1000)
    if isinstance(created_at, (int, float)):
        return int(created_at)
    # Timestamp-like objects (Firestore Timestamp) expose timestamp()
    ts = getattr(created_at, "timestamp", None)
    if callable(ts):
        try:
            return int(ts() * 1000)
        except Exception:
            pass
    return int(time.time() * 1000)


def ensure_credit_fields(user_id: str) -> dict | None:
    """Backfill credit fields on a legacy user doc that predates them.
    Only fills MISSING keys — never overwrites existing values, and never
    grants a fresh trial: a legacy user with free_trial_used=True backfills
    trial_credits=0, and trial_started_at comes from created_at. Returns
    the merged user dict, or None if the user does not exist."""
    uid = str(user_id)
    if not is_db_available():
        u = _mem_users.get(uid)
        if u is None:
            return None
        missing = {}
        if "free_mini_used" not in u:
            missing["free_mini_used"] = False
        if "interview_credits" not in u:
            missing["interview_credits"] = 0
        if "trial_credits" not in u:
            missing["trial_credits"] = 0 if u.get("free_trial_used") else 1
        if "trial_started_at" not in u:
            missing["trial_started_at"] = _trial_started_ms_from_created(u.get("created_at"))
        if "credited_charges" not in u:
            missing["credited_charges"] = []
        u.update(missing)
        return u
    doc = get_db().collection("users").document(uid).get()
    if not doc.exists:
        return None
    d = doc.to_dict() or {}
    missing = {}
    if "free_mini_used" not in d:
        missing["free_mini_used"] = False
    if "interview_credits" not in d:
        missing["interview_credits"] = 0
    if "trial_credits" not in d:
        missing["trial_credits"] = 0 if d.get("free_trial_used") else 1
    if "trial_started_at" not in d:
        missing["trial_started_at"] = _trial_started_ms_from_created(d.get("created_at"))
    if "credited_charges" not in d:
        missing["credited_charges"] = []
    if missing:
        try:
            get_db().collection("users").document(uid).update(missing)
        except Exception:
            log.warning("ensure_credit_fields update failed for %s", uid)
        d.update(missing)
    return d


def update_user(user_id: str, fields: dict):
    """Update user profile fields (best-effort)."""
    if not is_db_available():
        if str(user_id) in _mem_users:
            _mem_users[str(user_id)].update(fields)
        return
    try:
        get_db().collection("users").document(str(user_id)).update(fields)
    except Exception:
        pass  # doc may not exist yet; create_user handles it


def mark_trial_used(user_id: str):
    if not is_db_available():
        if str(user_id) in _mem_users:
            _mem_users[str(user_id)]["free_trial_used"] = True
        return
    get_db().collection("users").document(str(user_id)).update(
        {"free_trial_used": True})


# ---------- free_mini claim (atomic) ----------

def claim_free_mini(user_id: str) -> bool:
    """Atomically claim the user's one free_mini session. Returns True iff
    THIS caller claimed it (flag was False, now True). False means the
    free_mini was already used — caller must 402, never start a session.
    Call release_free_mini() if session creation fails after a claim."""
    uid = str(user_id)
    if not is_db_available():
        u = _mem_users.get(uid)
        if u is None or u.get("free_mini_used"):
            return False
        u["free_mini_used"] = True
        return True
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("users").document(uid)

    @_fs.transactional
    def _claim(transaction):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            return False
        if (snap.to_dict() or {}).get("free_mini_used"):
            return False
        transaction.set(ref, {"free_mini_used": True}, merge=True)
        return True

    return _claim(db.transaction())


def release_free_mini(user_id: str):
    """Release a free_mini claim after session creation failed, so the user
    can retry. ONLY call after this caller successfully claimed (claim
    returned True) and session creation then failed — never to reset a
    legitimately consumed free_mini."""
    uid = str(user_id)
    if not is_db_available():
        if uid in _mem_users:
            _mem_users[uid]["free_mini_used"] = False
        return
    try:
        get_db().collection("users").document(uid).update({"free_mini_used": False})
    except Exception:
        pass


# ---------- sessions ----------

def create_session(user_id: str, test_type: str, plan: str,
                   max_duration_sec: int, affiliate_code: str | None = None,
                   background_summary: str | None = None,
                   idempotency_key: str | None = None,
                   credit_consumed: bool = False) -> str:
    sid = uuid.uuid4().hex[:16]
    data = {
        "user_id": str(user_id),
        "test_type": test_type,
        "plan": plan,
        "max_duration_sec": max_duration_sec,
        # FIX 17: timer alias — frontend resyncs from maxSeconds.
        "maxSeconds": max_duration_sec,
        "transcript": [],
        "score": None,
        "verdict": None,
        "breakdown": None,
        "weaknesses": [],
        "feedback": "",
        "duration_sec": None,
        "affiliate_code": affiliate_code,
        "background_summary": (background_summary or "")[:2000],
        "created_at": _now(),
        "ended_at": None,
        # FIX 13/14: idempotency + refund bookkeeping.
        "idempotency_key": idempotency_key or None,
        "credit_consumed": bool(credit_consumed),
        "refunded": False,
        "refunded_at": None,
    }
    if not is_db_available():
        _mem_sessions[sid] = data
        return sid
    get_db().collection("sessions").document(sid).set(data)
    return sid


def get_session(session_id: str) -> dict | None:
    if not is_db_available():
        return _mem_sessions.get(session_id)
    doc = get_db().collection("sessions").document(session_id).get()
    return doc.to_dict() if doc.exists else None


# FIX 25 (re-derived): a second /session/start for the same user must not
# overlap an active unscored session — Setup retries return 409 with the
# existing session_id instead of creating a duplicate (and burning a
# credit). "Active" = unscored, not denied, not ended, AND created within
# max_duration + 30 min grace. Older sessions are stale orphans (the voice
# proxy's timeup should have ended them); they do not block a new start,
# otherwise one crashed session would lock the user out forever.
ACTIVE_SESSION_GRACE_SEC = 1800


def _session_age_sec(sess: dict) -> float | None:
    """Age of a session in seconds, or None when created_at is unparseable."""
    created = sess.get("created_at")
    if created is None:
        return None
    if isinstance(created, datetime):
        return (_now() - created).total_seconds()
    if isinstance(created, (int, float)):
        return time.time() - (created / 1000.0 if created > 1e12 else created)
    ts = getattr(created, "timestamp", None)
    if callable(ts):
        try:
            return time.time() - ts()
        except Exception:
            return None
    return None


def _session_is_live(sess: dict) -> bool:
    if sess.get("score") is not None or sess.get("denied") or sess.get("ended_at") is not None:
        return False
    age = _session_age_sec(sess)
    if age is None:
        return False
    max_dur = int(sess.get("max_duration_sec") or sess.get("maxSeconds") or 900)
    return age < (max_dur + ACTIVE_SESSION_GRACE_SEC)


def find_active_unscored_session(user_id: str) -> dict | None:
    """Return the user's live unscored session (with session_id) or None.
    Equality-only Firestore filters (no composite index needed)."""
    uid = str(user_id)
    if not is_db_available():
        for sid, s in _mem_sessions.items():
            if str(s.get("user_id", "")) == uid and _session_is_live(s):
                return {"session_id": sid, **s}
        return None
    try:
        q = (get_db().collection("sessions")
             .where("user_id", "==", uid)
             .where("score", "==", None)
             .where("ended_at", "==", None)
             .limit(10).stream())
        for doc in q:
            d = doc.to_dict() or {}
            if _session_is_live(d):
                return {"session_id": doc.id, **d}
    except Exception as e:
        log.warning("find_active_unscored_session failed: %s", str(e)[:100])
    return None


def get_session_by_idempotency_key(user_id: str, idempotency_key: str) -> dict | None:
    """FIX 13: idempotent replay lookup — return the session already created
    for (user_id, idempotency_key), or None. Lets a retried /session/start
    return the original session instead of charging a second credit."""
    if not idempotency_key:
        return None
    uid = str(user_id)
    if not is_db_available():
        for sid, s in _mem_sessions.items():
            if str(s.get("user_id", "")) == uid and s.get("idempotency_key") == idempotency_key:
                return {"session_id": sid, **s}
        return None
    try:
        # Single-field query + Python filter (no new composite index needed).
        q = (get_db().collection("sessions")
             .where("user_id", "==", uid)
             .limit(20).stream())
        for doc in q:
            d = doc.to_dict() or {}
            if d.get("idempotency_key") == idempotency_key:
                return {"session_id": doc.id, **d}
    except Exception as e:
        log.warning("get_session_by_idempotency_key failed: %s", str(e)[:100])
    return None


def find_ended_unscored_session(user_id: str) -> dict | None:
    """FIX 13: return the user's ended-but-unscored, not-yet-refunded session
    (with session_id), or None. Caller must 409 with ENDED_UNSCORED_SCORE_NOW
    (frontend routes to SCORE ME NOW) instead of charging for a new session.
    Refunded sessions do NOT block — the credit is already back."""
    uid = str(user_id)
    if not is_db_available():
        for sid, s in _mem_sessions.items():
            if str(s.get("user_id", "")) != uid:
                continue
            if s.get("score") is not None or s.get("denied") or s.get("refunded"):
                continue
            if s.get("ended_at") is None:
                continue
            return {"session_id": sid, **s}
        return None
    try:
        q = (get_db().collection("sessions")
             .where("user_id", "==", uid)
             .where("score", "==", None)
             .limit(20).stream())
        for doc in q:
            d = doc.to_dict() or {}
            if d.get("denied") or d.get("refunded"):
                continue
            if d.get("ended_at") is None:
                continue
            return {"session_id": doc.id, **d}
    except Exception as e:
        log.warning("find_ended_unscored_session failed: %s", str(e)[:100])
    return None


def _turn_key(turn):
    """Identity of one transcript turn for merge-by-turn dedup.

    Transcript entries are dicts {"role", "text", ...}; non-dict entries
    fall back to the entry itself. Two turns with the same (role, text)
    are the same turn for merge purposes."""
    if isinstance(turn, dict):
        return (turn.get("role"), turn.get("text"))
    return (None, turn)


def _merge_transcripts(stored, incoming):
    """MERGE-by-turn: union of stored + incoming, stored order first.

    A flush must NEVER delete turns: a shorter flush (e.g. a handler
    whose local transcript missed early turns) only ADDS its new turns.
    Idempotent — merging the same transcript twice changes nothing.
    Exact-duplicate turns across the merge boundary collapse (a repeated
    sentence spoken twice keeps one copy); duplicates inside one flush
    are preserved by the prefix fast path below."""
    stored = stored or []
    incoming = incoming or []
    if not incoming:
        return list(stored)
    if not stored:
        return list(incoming)
    # Fast path: stored is a prefix of incoming (normal progressive flush
    # from one handler) — incoming is the fuller transcript, take it whole.
    if len(incoming) >= len(stored) and all(
        _turn_key(incoming[i]) == _turn_key(stored[i])
        for i in range(len(stored))
    ):
        return list(incoming)
    seen = {_turn_key(t) for t in stored}
    merged = list(stored)
    for t in incoming:
        if _turn_key(t) not in seen:
            seen.add(_turn_key(t))
            merged.append(t)
    return merged


def append_transcript(session_id: str, transcript: list[dict]):
    # MERGE-by-turn: never full-replace. A shorter flush must not delete
    # turns an earlier flush already persisted (see _merge_transcripts).
    if not is_db_available():
        if session_id in _mem_sessions:
            _mem_sessions[session_id]["transcript"] = _merge_transcripts(
                _mem_sessions[session_id].get("transcript"), transcript)
        return
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("sessions").document(session_id)

    @_fs.transactional
    def _merge(transaction):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            return
        stored = (snap.to_dict() or {}).get("transcript") or []
        transaction.set(ref, {"transcript": _merge_transcripts(stored, transcript)},
                        merge=True)

    _merge(db.transaction())


def mark_session_ended(session_id: str, duration_sec: int):
    if not is_db_available():
        if session_id in _mem_sessions:
            _mem_sessions[session_id].update({"ended_at": _now(), "duration_sec": duration_sec})
        return
    get_db().collection("sessions").document(session_id).update(
        {"ended_at": _now(), "duration_sec": duration_sec})


def update_session_scored(session_id: str, score: int, verdict: str,
                          breakdown: dict, weaknesses: list, feedback: str):
    if not is_db_available():
        if session_id in _mem_sessions:
            _mem_sessions[session_id].update({
                "score": score, "verdict": verdict, "breakdown": breakdown,
                "weaknesses": weaknesses, "feedback": feedback,
            })
            _mem_sessions[session_id].pop("scoring_in_progress", None)
            _mem_sessions[session_id].pop("claimed_at", None)
        return
    from google.cloud import firestore as _fs
    get_db().collection("sessions").document(session_id).update({
        "score": score,
        "verdict": verdict,
        "breakdown": breakdown,
        "weaknesses": weaknesses,
        "feedback": feedback,
        "scoring_in_progress": False,
        "claimed_at": _fs.DELETE_FIELD,
    })


SCORING_CLAIM_TTL_SEC = 300


def claim_scoring(session_id: str) -> str:
    """Atomically claim the right to score a session.
    Returns "claimed" (this caller should score), "scored" (already done),
    "in_progress" (another request is scoring), or "missing" (no session).
    The claim carries claimed_at (epoch seconds) with a 300s TTL: a crash
    mid-scoring no longer blocks /session/end forever — a later caller
    steals the stale claim in the SAME transaction that checks it. A
    legacy claim with no claimed_at is treated as stale (deploy replaces
    all scorers at once; any genuinely-live pre-deploy claim is seconds
    old and its owner re-claims or the poll loop finds the result)."""
    if not is_db_available():
        s = _mem_sessions.get(session_id)
        if s is None:
            return "missing"
        if s.get("score") is not None:
            return "scored"
        if s.get("scoring_in_progress"):
            claimed_at = s.get("claimed_at")
            if claimed_at is not None:
                try:
                    fresh = (time.time() - float(claimed_at)) < SCORING_CLAIM_TTL_SEC
                except (TypeError, ValueError):
                    fresh = False  # unparseable -> stale, steal it
                if fresh:
                    return "in_progress"
            # stale (or legacy, no claimed_at) — steal it
        s["scoring_in_progress"] = True
        s["claimed_at"] = int(time.time())
        return "claimed"
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("sessions").document(session_id)

    @_fs.transactional
    def _claim(transaction):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            return "missing"
        d = snap.to_dict() or {}
        if d.get("score") is not None:
            return "scored"
        if d.get("scoring_in_progress"):
            claimed_at = d.get("claimed_at")
            if claimed_at is not None:
                try:
                    age = time.time() - float(claimed_at)
                except (TypeError, ValueError):
                    age = SCORING_CLAIM_TTL_SEC  # unparseable -> stale
                if age < SCORING_CLAIM_TTL_SEC:
                    return "in_progress"
            # stale (or legacy, no claimed_at) — steal it in this txn
        transaction.set(ref, {"scoring_in_progress": True,
                              "claimed_at": int(time.time())}, merge=True)
        return "claimed"

    return _claim(db.transaction())


def release_scoring(session_id: str):
    """Clear a scoring claim (after a failed scoring attempt). Clears BOTH
    scoring_in_progress and claimed_at so no stale claim survives."""
    if not is_db_available():
        s = _mem_sessions.get(session_id) or {}
        s.pop("scoring_in_progress", None)
        s.pop("claimed_at", None)
        return
    try:
        from google.cloud import firestore as _fs
        get_db().collection("sessions").document(session_id).update(
            {"scoring_in_progress": False, "claimed_at": _fs.DELETE_FIELD})
    except Exception:
        pass


# ---------- DS-160 attack vectors (Feature 1: Dynamic Attacker) ----------

def get_last_attack_vector(user_id: str) -> str | None:
    """Return the attack vector used in the user's previous session (or None)."""
    uid = str(user_id)
    if not is_db_available():
        return (_mem_users.get(uid) or {}).get("last_attack_vector")
    try:
        doc = get_db().collection("users").document(uid).get()
        if not doc.exists:
            return None
        return (doc.to_dict() or {}).get("last_attack_vector")
    except Exception:
        return None


def set_last_attack_vector(user_id: str, vector: str):
    """Remember which attack vector was used so the next session picks a different one."""
    uid = str(user_id)
    if not is_db_available():
        _mem_users.setdefault(uid, {})["last_attack_vector"] = vector
        return
    try:
        get_db().collection("users").document(uid).set(
            {"last_attack_vector": vector}, merge=True)
    except Exception:
        pass


# ---------- session analytics: nervousness, facts, over-explaining ----------

def update_session_analytics(session_id: str, fields: dict):
    """Best-effort analytics update (nervousness_score, filler_count,
    word_count, facts_mentioned, over_explain_count). Never raises."""
    try:
        if not is_db_available():
            if session_id in _mem_sessions:
                _mem_sessions[session_id].update(fields)
            return
        get_db().collection("sessions").document(session_id).update(fields)
    except Exception as e:
        log.warning("update_session_analytics failed: %s", str(e)[:100])


def mark_session_denied(session_id: str, reason: str):
    """Instant cut-off: the officer ended the interview (over-explaining)."""
    fields = {"denied": True, "denied_reason": reason, "ended_at": _now()}
    try:
        if not is_db_available():
            if session_id in _mem_sessions:
                _mem_sessions[session_id].update(fields)
            return
        get_db().collection("sessions").document(session_id).update(fields)
    except Exception as e:
        log.warning("mark_session_denied failed: %s", str(e)[:100])


def get_daily_push_targets(limit: int = 100) -> list[dict]:
    """Users for the daily Grilly push: trial used but never paid (no Stars purchase).
    Returns [{telegram_id, first_name}]. Best-effort, capped at limit."""
    targets = []
    try:
        if not is_db_available():
            for uid, u in _mem_users.items():
                if u.get("free_trial_used") and not u.get("paid"):
                    targets.append({"telegram_id": str(uid),
                                    "first_name": u.get("first_name", "")})
                    if len(targets) >= limit:
                        break
            return targets
        q = (get_db().collection("users")
             .where("free_trial_used", "==", True)
             .where("paid", "==", False)
             .limit(limit).stream())
        for doc in q:
            d = doc.to_dict() or {}
            targets.append({"telegram_id": doc.id,
                            "first_name": d.get("first_name", "")})
    except Exception as e:
        log.warning("get_daily_push_targets failed: %s", str(e)[:100])
    return targets


# ---------- daily-grilly idempotency (one send per date) ----------
# daily_grilly_sends/{YYYY-MM-DD}: claim BEFORE sending, mark sent AFTER the
# loop, release (delete) on catastrophic failure so a retry can proceed.
# A claim older than 1h is stealable (the send loop takes minutes; a
# process that died mid-send must not block the whole day).

_mem_daily_grilly = {}  # date_str -> {status, claimed_at, ...}
_mem_daily_grilly_recipients: dict[str, dict] = {}  # "YYYY-MM-DD:telegram_id" -> {sent_at}
DAILY_GRILLY_CLAIM_TTL_SEC = 3600


def get_daily_grilly_send(date_str: str) -> dict | None:
    """Return the daily_grilly_sends/{date} doc (or None). Used to resume a
    partial send with its prior sent/failed counts."""
    if not is_db_available():
        cur = _mem_daily_grilly.get(date_str)
        return dict(cur) if cur else None
    try:
        doc = get_db().collection("daily_grilly_sends").document(date_str).get()
        return doc.to_dict() if doc.exists else None
    except Exception:
        return None


def has_daily_grilly_recipient(date_str: str, telegram_id: str) -> bool:
    """FIX 24: True iff this recipient already got today's push."""
    key = f"{date_str}:{telegram_id}"
    if not is_db_available():
        return key in _mem_daily_grilly_recipients
    try:
        doc = (get_db().collection("daily_grilly_sends").document(date_str)
               .collection("recipients").document(str(telegram_id)).get())
        return doc.exists
    except Exception:
        return False


def mark_daily_grilly_recipient_sent(date_str: str, telegram_id: str):
    """FIX 24: record a per-recipient send so a retry skips this user."""
    key = f"{date_str}:{telegram_id}"
    if not is_db_available():
        _mem_daily_grilly_recipients[key] = {"sent_at": int(time.time())}
        return
    try:
        (get_db().collection("daily_grilly_sends").document(date_str)
         .collection("recipients").document(str(telegram_id))
         .set({"sent_at": int(time.time()), "telegram_id": str(telegram_id)}, merge=True))
    except Exception as e:
        log.warning("mark_daily_grilly_recipient_sent failed: %s", str(e)[:100])


def claim_daily_grilly_send(date_str: str) -> bool:
    """Claim today's daily-grilly send. True iff THIS caller owns the send.
    False when already sent, or another live claim owns it.
    FIX 24: a 'partial' send (catastrophic failure mid-loop) is
    reclaimable — the caller resumes and per-recipient records skip users
    who already got today's push."""
    now = int(time.time())
    if not is_db_available():
        cur = _mem_daily_grilly.get(date_str)
        if cur and cur.get("status") == "sent":
            return False
        if cur and cur.get("status") == "claimed" and (now - int(cur.get("claimed_at", 0))) < DAILY_GRILLY_CLAIM_TTL_SEC:
            return False
        # 'partial' or stale claim or first claim — (re)claim, preserving
        # prior sent/failed counts and per-recipient records.
        prev = dict(cur) if cur else {}
        prev.update({"status": "claimed", "claimed_at": now, "date": date_str})
        _mem_daily_grilly[date_str] = prev
        return True
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("daily_grilly_sends").document(date_str)

    @_fs.transactional
    def _claim(transaction):
        snap = ref.get(transaction=transaction)
        if snap.exists:
            d = snap.to_dict() or {}
            if d.get("status") == "sent":
                return False
            if d.get("status") == "claimed":
                try:
                    age = now - int(d.get("claimed_at") or 0)
                except (TypeError, ValueError):
                    age = DAILY_GRILLY_CLAIM_TTL_SEC
                if age < DAILY_GRILLY_CLAIM_TTL_SEC:
                    return False  # live claim — another sender owns today
                # stale claim — steal it in this txn
            # 'partial' falls through: resume the send in this txn.
        transaction.set(ref, {"status": "claimed", "claimed_at": now,
                              "date": date_str}, merge=True)
        return True

    return _claim(db.transaction())


def mark_daily_grilly_sent(date_str: str, sent: int, failed: int):
    """Mark today's send complete. Called AFTER the send loop, even when
    some individual sends failed — a retry must not re-message users who
    already got today's push (failed users simply miss one day)."""
    if not is_db_available():
        _mem_daily_grilly[date_str] = {"status": "sent", "sent_at": int(time.time()),
                                       "sent": sent, "failed": failed}
        return
    get_db().collection("daily_grilly_sends").document(date_str).set(
        {"status": "sent", "sent_at": int(time.time()),
         "sent": sent, "failed": failed, "date": date_str}, merge=True)


def mark_daily_grilly_partial(date_str: str, sent: int, failed: int):
    """FIX 24: record a partial send after a catastrophic failure. The day
    doc stays (status='partial') with its per-recipient records intact, so
    a retry resumes and skips already-sent recipients instead of
    re-messaging them or losing the whole claim."""
    if not is_db_available():
        cur = _mem_daily_grilly.get(date_str) or {}
        cur.update({"status": "partial", "sent": int(sent), "failed": int(failed),
                    "date": date_str, "partial_at": int(time.time())})
        _mem_daily_grilly[date_str] = cur
        return
    try:
        get_db().collection("daily_grilly_sends").document(date_str).set(
            {"status": "partial", "sent": int(sent), "failed": int(failed),
             "date": date_str, "partial_at": int(time.time())}, merge=True)
    except Exception:
        pass


def release_daily_grilly_send(date_str: str):
    """DEPRECATED (FIX 24): previously deleted the whole day claim on
    catastrophic failure, which lost per-recipient progress. Now it
    transitions the claim to 'partial' (preserving per-recipient records)
    instead of deleting anything. New code must call
    mark_daily_grilly_partial() directly. A completed send is never
    released (see mark_sent)."""
    cur = get_daily_grilly_send(date_str) or {}
    mark_daily_grilly_partial(date_str, int(cur.get("sent") or 0), int(cur.get("failed") or 0))


# ---------- stars invoices (pending, created before checkout) ----------

_mem_invoices = {}
_mem_payments = {}
_mem_feedback = []  # list[dict] — in-memory fallback for the feedback collection


def record_stars_invoice(payload_token: str, user_id: str,
                         affiliate_code: str | None = None,
                         bundle_id: str | None = None):
    data = {
        "payload": payload_token,
        "user_id": str(user_id),
        "affiliate_code": affiliate_code,
        "bundle_id": bundle_id,
        "used": False,
        "created_at": _now(),
        "used_at": None,
    }
    if not is_db_available():
        _mem_invoices[payload_token] = data
        return
    get_db().collection("stars_invoices").document(payload_token).set(data)


def get_stars_invoice(payload_token: str) -> dict | None:
    if not is_db_available():
        return _mem_invoices.get(payload_token)
    doc = get_db().collection("stars_invoices").document(payload_token).get()
    return doc.to_dict() if doc.exists else None


# ---------- interview credits (pay-per-use bundles) ----------

def add_interview_credits(user_id: str, n: int) -> int:
    """Add n interview credits. Returns the new total.
    FIX 13: transactional Increment — no read-modify-write race. The old
    get-then-set lost concurrent grants/refunds (two writers read the same
    total and one increment vanished)."""
    uid = str(user_id)
    if not is_db_available():
        cur = (_mem_users.get(uid) or {}).get("interview_credits", 0)
        new_total = cur + int(n)
        _mem_users.setdefault(uid, {})["interview_credits"] = new_total
        return new_total
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("users").document(uid)

    @_fs.transactional
    def _add(transaction):
        # Server-side atomic increment inside the transaction; no
        # client-side read-modify-write. set(merge=True) also creates the
        # doc when the user row is missing.
        transaction.set(ref, {"interview_credits": _fs.Increment(int(n))}, merge=True)

    _add(db.transaction())
    doc = ref.get()
    if doc.exists:
        return int((doc.to_dict() or {}).get("interview_credits", 0) or 0)
    return int(n)


def get_interview_credits(user_id: str) -> int:
    uid = str(user_id)
    if not is_db_available():
        return int((_mem_users.get(uid) or {}).get("interview_credits", 0))
    doc = get_db().collection("users").document(uid).get()
    if not doc.exists:
        return 0
    return int((doc.to_dict() or {}).get("interview_credits", 0) or 0)


def use_interview_credit(user_id: str) -> bool:
    """Consume one credit atomically via Firestore transaction.
    Returns True if a credit was available."""
    uid = str(user_id)
    if not is_db_available():
        cur = int((_mem_users.get(uid) or {}).get("interview_credits", 0))
        if cur <= 0:
            return False
        _mem_users[uid]["interview_credits"] = cur - 1
        return True
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("users").document(uid)

    @_fs.transactional
    def _consume(transaction):
        snap = ref.get(transaction=transaction)
        cur = int((snap.to_dict() or {}).get("interview_credits", 0) or 0) if snap.exists else 0
        if cur <= 0:
            return False
        transaction.set(ref, {"interview_credits": cur - 1}, merge=True)
        return True

    return _consume(db.transaction())


# ---------- refund-on-abandon (FIX 14) ----------
def refund_abandoned_session(session_id: str) -> dict:
    """FIX 14: refund a session abandoned after credit burn but never
    started/scored. Transactional + idempotent, never double-refunds.

    Abandoned = score is None, not denied, not already refunded, AND
    (ended_at is not None OR age > max_duration + grace). Live sessions
    (see _session_is_live) are NOT abandoned — caller gets
    status='session_active' and must wait or finish/score first.

    Credit rule: only plan='paid' sessions that actually consumed a credit
    (credit_consumed=True, or legacy sessions where the user is not
    unlimited-paid) get +1 credit back via Increment inside the same
    transaction that marks refunded=True. plan='free_mini' releases the
    mini claim instead (no interview credit was burned). plan='free_trial'
    and unlimited-paid sessions get status='no_refund_needed' (the trial
    is only marked used on scoring, so nothing was burned).

    Returns {"status": "refunded"|"already_refunded"|"not_found"|
    "already_scored"|"denied"|"session_active"|"not_abandoned"|
    "no_refund_needed", "new_total": int|None, ...}. Any non-refunded
    status means NOTHING was written (except already_refunded, which is
    a safe idempotent replay returning the original outcome)."""
    if not is_db_available():
        s = _mem_sessions.get(session_id)
        if s is None:
            return {"status": "not_found", "session_id": session_id}
        if s.get("score") is not None:
            return {"status": "already_scored", "session_id": session_id}
        if s.get("denied"):
            return {"status": "denied", "session_id": session_id}
        if s.get("refunded"):
            u = _mem_users.get(str(s.get("user_id", "")), {})
            return {"status": "already_refunded", "session_id": session_id,
                    "user_id": str(s.get("user_id", "")),
                    "new_total": int(u.get("interview_credits", 0) or 0)}
        if _session_is_live(s):
            return {"status": "session_active", "session_id": session_id}
        age = _session_age_sec(s)
        max_dur = int(s.get("max_duration_sec") or s.get("maxSeconds") or 900)
        if s.get("ended_at") is None:
            if age is None or age <= (max_dur + ACTIVE_SESSION_GRACE_SEC):
                return {"status": "not_abandoned", "session_id": session_id}
        plan = s.get("plan")
        uid = str(s.get("user_id", ""))
        if plan == "free_trial":
            return {"status": "no_refund_needed", "session_id": session_id,
                    "reason": "free_trial burns nothing until scored"}
        if plan == "free_mini":
            s["refunded"] = True
            s["refunded_at"] = _now()
            if uid in _mem_users:
                _mem_users[uid]["free_mini_used"] = False
            return {"status": "refunded", "session_id": session_id,
                    "user_id": uid, "refund_type": "free_mini_release",
                    "new_total": int((_mem_users.get(uid) or {}).get("interview_credits", 0) or 0)}
        if plan == "paid":
            u = _mem_users.get(uid) or {}
            consumed = s.get("credit_consumed")
            if consumed is None:
                consumed = not bool(u.get("paid"))
            if not consumed or bool(u.get("paid")):
                return {"status": "no_refund_needed", "session_id": session_id,
                        "reason": "unlimited-paid session consumed no credit"}
            new_total = int(u.get("interview_credits", 0) or 0) + 1
            u["interview_credits"] = new_total
            _mem_users[uid] = u
            s["refunded"] = True
            s["refunded_at"] = _now()
            return {"status": "refunded", "session_id": session_id,
                    "user_id": uid, "refund_type": "credit",
                    "credits_refunded": 1, "new_total": new_total}
        return {"status": "no_refund_needed", "session_id": session_id,
                "reason": f"unknown plan {plan!r}"}
    from google.cloud import firestore as _fs
    db = get_db()
    sref = db.collection("sessions").document(session_id)

    @_fs.transactional
    def _refund(transaction):
        snap = sref.get(transaction=transaction)
        if not snap.exists:
            return {"status": "not_found", "session_id": session_id}
        s = snap.to_dict() or {}
        if s.get("score") is not None:
            return {"status": "already_scored", "session_id": session_id}
        if s.get("denied"):
            return {"status": "denied", "session_id": session_id}
        if s.get("refunded"):
            return {"status": "already_refunded", "session_id": session_id,
                    "user_id": str(s.get("user_id", ""))}
        if _session_is_live(s):
            return {"status": "session_active", "session_id": session_id}
        age = _session_age_sec(s)
        max_dur = int(s.get("max_duration_sec") or s.get("maxSeconds") or 900)
        if s.get("ended_at") is None:
            if age is None or age <= (max_dur + ACTIVE_SESSION_GRACE_SEC):
                return {"status": "not_abandoned", "session_id": session_id}
        plan = s.get("plan")
        uid = str(s.get("user_id", ""))
        uref = db.collection("users").document(uid)
        usnap = uref.get(transaction=transaction)
        udata = usnap.to_dict() if usnap.exists else {}
        if plan == "free_trial":
            return {"status": "no_refund_needed", "session_id": session_id,
                    "reason": "free_trial burns nothing until scored"}
        if plan == "free_mini":
            transaction.set(sref, {"refunded": True, "refunded_at": _now()}, merge=True)
            transaction.set(uref, {"free_mini_used": False}, merge=True)
            return {"status": "refunded", "session_id": session_id,
                    "user_id": uid, "refund_type": "free_mini_release",
                    "new_total": int(udata.get("interview_credits", 0) or 0)}
        if plan == "paid":
            consumed = s.get("credit_consumed")
            if consumed is None:
                consumed = not bool(udata.get("paid"))
            if not consumed or bool(udata.get("paid")):
                return {"status": "no_refund_needed", "session_id": session_id,
                        "reason": "unlimited-paid session consumed no credit"}
            # Atomic: mark refunded + Increment credit in ONE transaction.
            # The refunded flag is checked first in this txn, so a
            # concurrent second refund sees refunded=True and returns
            # already_refunded without a second Increment.
            transaction.set(sref, {"refunded": True, "refunded_at": _now()}, merge=True)
            transaction.set(uref, {"interview_credits": _fs.Increment(1)}, merge=True)
            new_total = int(udata.get("interview_credits", 0) or 0) + 1
            return {"status": "refunded", "session_id": session_id,
                    "user_id": uid, "refund_type": "credit",
                    "credits_refunded": 1, "new_total": new_total}
        return {"status": "no_refund_needed", "session_id": session_id,
                "reason": f"unknown plan {plan!r}"}

    return _refund(db.transaction())


# ---------- referrals ----------
def claim_referral(user_id: str, referrer_id: str) -> dict:
    """Claim a referral bonus: the new user gets +1 interview credit AND the
    referrer gets +1 credit plus referral_count/referral_earned incremented.
    One claim per user ever (idempotent guard).
    FIX 3: NEVER auto-mint a user doc for a claimant that doesn't exist —
    the claimant doc must already exist (created via verified
    /api/auth/telegram). Raises ValueError("claimant_not_found") when it
    doesn't. Raises ValueError("self_referral" | "already_claimed" |
    "referrer_not_found" | "claimant_not_found").
    """
    uid, rid = str(user_id), str(referrer_id)
    if uid == rid:
        raise ValueError("self_referral")
    if not is_db_available():
        if uid not in _mem_users:
            raise ValueError("claimant_not_found")
        u = _mem_users.get(uid) or {}
        if u.get("referral_claimed"):
            raise ValueError("already_claimed")
        r = _mem_users.get(rid)
        if not r:
            raise ValueError("referrer_not_found")
        u["interview_credits"] = int(u.get("interview_credits", 0)) + 1
        u["referred_by"] = rid
        u["referral_claimed"] = True
        _mem_users[uid] = u
        r["interview_credits"] = int(r.get("interview_credits", 0)) + 1
        r["referral_count"] = int(r.get("referral_count", 0)) + 1
        r["referral_earned"] = int(r.get("referral_earned", 0)) + 1
        _mem_users[rid] = r
        return {"user_id": uid, "referrer_id": rid,
                "user_credits": u["interview_credits"],
                "referrer_credits": r["interview_credits"],
                "referral_count": r["referral_count"]}
    db = get_db()
    uref = db.collection("users").document(uid)
    rref = db.collection("users").document(rid)
    from google.cloud import firestore as _fs

    @_fs.transactional
    def _claim(transaction):
        udoc = uref.get(transaction=transaction)
        if not udoc.exists:
            # FIX 3: never auto-mint a claimant doc — caller must 404.
            raise ValueError("claimant_not_found")
        udata = (udoc.to_dict() or {})
        if udata.get("referral_claimed"):
            raise ValueError("already_claimed")
        rdoc = rref.get(transaction=transaction)
        if not rdoc.exists:
            raise ValueError("referrer_not_found")
        rdata = rdoc.to_dict() or {}
        # Credit the new user (+1 interview), mark claimed
        u_credits = int(udata.get("interview_credits", 0) or 0) + 1
        transaction.set(uref, {
            "interview_credits": u_credits,
            "referred_by": rid,
            "referral_claimed": True,
        }, merge=True)
        # Credit the referrer (+1 interview, bump counters)
        r_credits = int(rdata.get("interview_credits", 0) or 0) + 1
        r_count = int(rdata.get("referral_count", 0) or 0) + 1
        r_earned = int(rdata.get("referral_earned", 0) or 0) + 1
        transaction.set(rref, {
            "interview_credits": r_credits,
            "referral_count": r_count,
            "referral_earned": r_earned,
        }, merge=True)
        return {"user_id": uid, "referrer_id": rid,
                "user_credits": u_credits,
                "referrer_credits": r_credits,
                "referral_count": r_count}

    return _claim(db.transaction())


def get_referral_stats(user_id: str) -> dict:
    """Referral stats for the invite UI: friends joined + free interviews earned."""
    uid = str(user_id)
    if not is_db_available():
        u = _mem_users.get(uid) or {}
        return {"user_id": uid,
                "referral_count": int(u.get("referral_count", 0) or 0),
                "referral_earned": int(u.get("referral_earned", 0) or 0)}
    doc = get_db().collection("users").document(uid).get()
    d = (doc.to_dict() or {}) if doc.exists else {}
    return {"user_id": uid,
            "referral_count": int(d.get("referral_count", 0) or 0),
            "referral_earned": int(d.get("referral_earned", 0) or 0)}


# ---------- payments (Telegram Stars) ----------
# ONE atomic grant path: grant_stars_payment(). The old split path
# (record_payment txn + add_interview_credits read-modify-write +
# mark_invoice_used, plus the dead affiliate block after record_payment's
# `return`) is deleted — it could double-grant under concurrency and the
# affiliate block after `return` never executed. Affiliate revenue-share
# bookkeeping is intentionally NOT re-added: it was dead code, and adding
# a second write path would break the one-path rule. If affiliate payouts
# are wanted, add them INSIDE grant_stars_payment's transaction.

def _grant_checks(payload_token: str, currency: str, total_amount: int,
                  expected_stars: int, inv: dict | None) -> str | None:
    """Pure validation shared by both backends. Returns an error status or
    None when the payment may be granted. Caller resolves the bundle from
    the STORED invoice (never from client-controlled payload) and passes
    its expected stars."""
    parts = (payload_token or "").split(":")
    if len(parts) < 3 or not all(parts[:3]):
        return "invalid_payload"
    if inv is None:
        return "unknown_invoice"
    if inv.get("used"):
        return "invoice_used"
    if str(inv.get("user_id", "")) != parts[0] or str(inv.get("bundle_id", "")) != parts[1]:
        return "payload_mismatch"
    if currency != "XTR":
        return "currency_mismatch"
    try:
        if int(total_amount) != int(expected_stars):
            return "amount_mismatch"
    except (TypeError, ValueError):
        return "amount_mismatch"
    return None


def grant_stars_payment(charge_id: str, payload_token: str, currency: str,
                        total_amount: int, expected_stars: int,
                        credit_count: int) -> dict:
    """Grant a Stars payment in ONE transaction. Checks, in order:
    duplicate charge (payments doc) -> invoice exists -> invoice not used
    -> invoice user/bundle match the payload -> currency == XTR ->
    amount == expected bundle stars -> charge not already in the user's
    credited_charges. Then, atomically: record payment + mark invoice used
    + grant credits + append charge to credited_charges.
    Returns {"status": "granted"|"duplicate"|"unknown_invoice"|
    "invoice_used"|"payload_mismatch"|"currency_mismatch"|"amount_mismatch"|
    "invalid_payload", ...}. Any non-granted status means NOTHING was
    written — the caller must not grant credits by any other path, and
    must NEVER fall back to mark_paid."""
    if not is_db_available():
        if charge_id in _mem_payments:
            return {"status": "duplicate", "charge_id": charge_id}
        inv = _mem_invoices.get(payload_token)
        err = _grant_checks(payload_token, currency, total_amount, expected_stars, inv)
        if err:
            return {"status": err, "charge_id": charge_id}
        user_id = str(inv["user_id"])
        u = _mem_users.setdefault(user_id, {})
        credited = list(u.get("credited_charges") or [])
        if charge_id in credited:
            return {"status": "duplicate", "charge_id": charge_id}
        new_total = int(u.get("interview_credits", 0) or 0) + int(credit_count)
        u["interview_credits"] = new_total
        u["credited_charges"] = credited + [charge_id]
        _mem_payments[charge_id] = {
            "payment_id": charge_id, "user_id": user_id,
            "amount_stars": int(total_amount), "status": "captured",
            "affiliate_code": inv.get("affiliate_code"), "created_at": _now(),
        }
        inv["used"] = True
        inv["used_at"] = _now()
        return {"status": "granted", "charge_id": charge_id, "user_id": user_id,
                "credits_granted": int(credit_count), "new_total": new_total}
    from google.cloud import firestore as _fs
    db = get_db()
    pay_ref = db.collection("payments").document(charge_id)
    inv_ref = db.collection("stars_invoices").document(payload_token)

    @_fs.transactional
    def _grant(transaction):
        if pay_ref.get(transaction=transaction).exists:
            return {"status": "duplicate", "charge_id": charge_id}
        inv_snap = inv_ref.get(transaction=transaction)
        inv = inv_snap.to_dict() if inv_snap.exists else None
        err = _grant_checks(payload_token, currency, total_amount, expected_stars, inv)
        if err:
            return {"status": err, "charge_id": charge_id}
        user_id = str(inv["user_id"])
        uref = db.collection("users").document(user_id)
        usnap = uref.get(transaction=transaction)
        udata = usnap.to_dict() if usnap.exists else {}
        credited = list(udata.get("credited_charges") or [])
        if charge_id in credited:
            return {"status": "duplicate", "charge_id": charge_id}
        new_total = int(udata.get("interview_credits", 0) or 0) + int(credit_count)
        transaction.set(pay_ref, {
            "payment_id": charge_id, "user_id": user_id,
            "amount_stars": int(total_amount), "status": "captured",
            "affiliate_code": inv.get("affiliate_code"), "created_at": _now(),
        })
        transaction.set(inv_ref, {"used": True, "used_at": _now()}, merge=True)
        transaction.set(uref, {
            "interview_credits": new_total,
            "credited_charges": credited + [charge_id],
        }, merge=True)
        return {"status": "granted", "charge_id": charge_id, "user_id": user_id,
                "credits_granted": int(credit_count), "new_total": new_total}

    return _grant(db.transaction())


# ---------- affiliates ----------

def get_affiliate_stats(code: str) -> dict | None:
    doc = get_db().collection("affiliates").document(code).get()
    if not doc.exists:
        return None
    d = doc.to_dict()
    # count referred paid users
    users = get_db().collection("users").where("affiliate_code", "==", code)\
        .where("paid", "==", True).stream()
    paid_count = sum(1 for _ in users)
    return {
        "code": code,
        "name": d.get("name", ""),
        "revenue_share": d.get("revenue_share", 0.2),
        "total_earned_stars": d.get("total_earned_stars", 0),
        "paid_users_referred": paid_count,
    }


# ---------- feedback / bug reports (MockGrill Support) ----------

def _normalize_feedback(d: dict) -> dict:
    """Copy a feedback doc, converting Firestore datetimes to ISO strings
    so the JSON API never leaks a non-serializable value."""
    out = dict(d or {})
    created = out.get("created_at")
    if isinstance(created, datetime):
        out["created_at"] = created.isoformat()
    return out


def save_feedback(user_id: str, type: str, message: str, screen: str | None = None,
                  session_id: str | None = None, app_version: str | None = None,
                  user_agent: str | None = None, first_name: str = "",
                  username: str = "") -> dict:
    """Store one user feedback / bug report. Returns the stored record
    (created_at as ISO string). Same shape in Firestore and mem fallback."""
    data = {
        "id": uuid.uuid4().hex,
        "user_id": str(user_id),
        "type": type,
        "message": message,
        "screen": screen or None,
        "session_id": session_id or None,
        "app_version": app_version or None,
        "user_agent": (user_agent or "")[:500],
        "first_name": first_name or "",
        "username": username or "",
        "status": "new",
        "created_at": _now(),
    }
    if not is_db_available():
        _mem_feedback.append(data)
        return _normalize_feedback(data)
    get_db().collection("feedback").document(data["id"]).set(data)
    return _normalize_feedback(data)


def list_feedback(status: str | None = None, limit: int = 100) -> list[dict]:
    """Newest feedback first. status=None returns all; otherwise filters
    to that status. limit clamped to 1..100 by the caller."""
    if not is_db_available():
        items = [_normalize_feedback(d) for d in _mem_feedback]
        if status:
            items = [d for d in items if d.get("status") == status]
        items.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return items[:limit]
    q = get_db().collection("feedback")
    if status:
        q = q.where("status", "==", status)
    q = q.order_by("created_at", direction="DESCENDING").limit(limit)
    return [_normalize_feedback(doc.to_dict()) for doc in q.stream()]

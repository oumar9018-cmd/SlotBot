"""
Firestore layer for Strict English Voice Test.

Collections:
  users      {telegram_id(str, doc id), paid:bool, free_trial_used:bool,
              affiliate_code:str|None, created_at, paid_at}
  sessions   {user_id, test_type, plan, max_duration_sec, transcript:[{role,text}],
              score:int|None, verdict, breakdown, weaknesses, feedback,
              duration_sec, affiliate_code, created_at, ended_at}
  affiliates {code(doc id), name, revenue_share (0-1), total_earned_stars, created_at}
  payments   {payment_id(doc id = Telegram charge id), user_id, amount_stars,
              status, affiliate_code, created_at}
  stars_invoices {payload(doc id), user_id, affiliate_code, used:bool,
              created_at, used_at}

Auth: firebase-admin with service account JSON whose path is in
FIREBASE_CRED_PATH. Lazily initialized on first use.
"""
import logging
import os
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
    data = {
        "telegram_id": str(user_id),
        "first_name": first_name,
        "username": username,
        "paid": False,
        "free_trial_used": False,
        "affiliate_code": affiliate_code,
        "created_at": _now(),
        "paid_at": None,
    }
    if not is_db_available():
        _mem_users[str(user_id)] = data
        return data
    get_db().collection("users").document(str(user_id)).set(data)
    return data


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


def mark_paid(user_id: str):
    if not is_db_available():
        if str(user_id) in _mem_users:
            _mem_users[str(user_id)]["paid"] = True
        return
    get_db().collection("users").document(str(user_id)).update(
        {"paid": True, "paid_at": _now()})


# ---------- sessions ----------

def create_session(user_id: str, test_type: str, plan: str,
                   max_duration_sec: int, affiliate_code: str | None = None,
                   background_summary: str | None = None) -> str:
    sid = uuid.uuid4().hex[:16]
    data = {
        "user_id": str(user_id),
        "test_type": test_type,
        "plan": plan,
        "max_duration_sec": max_duration_sec,
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


def append_transcript(session_id: str, transcript: list[dict]):
    # Replace with the full accumulated transcript (proxy owns ordering)
    if not is_db_available():
        if session_id in _mem_sessions:
            _mem_sessions[session_id]["transcript"] = transcript
        return
    get_db().collection("sessions").document(session_id).update(
        {"transcript": transcript})


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
        return
    get_db().collection("sessions").document(session_id).update({
        "score": score,
        "verdict": verdict,
        "breakdown": breakdown,
        "weaknesses": weaknesses,
        "feedback": feedback,
        "scoring_in_progress": False,
    })


def claim_scoring(session_id: str) -> str:
    """Atomically claim the right to score a session.
    Returns "claimed" (this caller should score), "scored" (already done),
    or "in_progress" (another request is scoring). Prevents double Groq
    scoring calls from concurrent /session/end requests."""
    if not is_db_available():
        s = _mem_sessions.get(session_id) or {}
        if s.get("score") is not None:
            return "scored"
        if s.get("scoring_in_progress"):
            return "in_progress"
        s["scoring_in_progress"] = True
        _mem_sessions[session_id] = s
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
            return "in_progress"
        transaction.set(ref, {"scoring_in_progress": True}, merge=True)
        return "claimed"

    return _claim(db.transaction())


def release_scoring(session_id: str):
    """Clear a stale scoring_in_progress flag (after a failed scoring attempt)."""
    if not is_db_available():
        s = _mem_sessions.get(session_id) or {}
        s.pop("scoring_in_progress", None)
        return
    try:
        get_db().collection("sessions").document(session_id).update(
            {"scoring_in_progress": False})
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


# ---------- stars invoices (pending, created before checkout) ----------

_mem_invoices = {}
_mem_payments = {}


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


def mark_invoice_used(payload_token: str):
    if not is_db_available():
        if payload_token in _mem_invoices:
            _mem_invoices[payload_token]["used"] = True
            _mem_invoices[payload_token]["used_at"] = _now()
        return
    get_db().collection("stars_invoices").document(payload_token).update(
        {"used": True, "used_at": _now()})


# ---------- interview credits (pay-per-use bundles) ----------

def add_interview_credits(user_id: str, n: int) -> int:
    """Add n interview credits. Returns the new total."""
    uid = str(user_id)
    if not is_db_available():
        cur = (_mem_users.get(uid) or {}).get("interview_credits", 0)
        new_total = cur + int(n)
        _mem_users.setdefault(uid, {})["interview_credits"] = new_total
        return new_total
    ref = get_db().collection("users").document(uid)
    doc = ref.get()
    cur = (doc.to_dict() or {}).get("interview_credits", 0) if doc.exists else 0
    new_total = int(cur) + int(n)
    ref.set({"interview_credits": new_total}, merge=True)
    return new_total


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


# ---------- referrals ----------
def claim_referral(user_id: str, referrer_id: str) -> dict:
    """Claim a referral bonus: the new user gets +1 interview credit AND the
    referrer gets +1 credit plus referral_count/referral_earned incremented.
    One claim per user ever (idempotent guard).
    Raises ValueError("self_referral" | "already_claimed" | "referrer_not_found").
    """
    uid, rid = str(user_id), str(referrer_id)
    if uid == rid:
        raise ValueError("self_referral")
    if not is_db_available():
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
        udata = (udoc.to_dict() or {}) if udoc.exists else {}
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

def get_payment(payment_id: str) -> dict | None:
    """Idempotency check — returns existing payment record if any."""
    if not is_db_available():
        return _mem_payments.get(payment_id)
    doc = get_db().collection("payments").document(payment_id).get()
    return doc.to_dict() if doc.exists else None


def record_payment(payment_id: str, user_id: str, amount_stars: int,
                   status: str, affiliate_code: str | None = None) -> bool:
    """Record a payment idempotently. Returns True if this was a NEW payment
    (caller should grant credits), False if it was already recorded
    (caller must NOT grant credits again). Uses a Firestore transaction so
    concurrent webhook retries can't double-credit."""
    data = {
        "payment_id": payment_id,
        "user_id": str(user_id),
        "amount_stars": int(amount_stars),
        "status": status,
        "affiliate_code": affiliate_code,
        "created_at": _now(),
    }
    if not is_db_available():
        if payment_id in _mem_payments:
            return False
        _mem_payments[payment_id] = data
        return True
    from google.cloud import firestore as _fs
    db = get_db()
    ref = db.collection("payments").document(payment_id)

    @_fs.transactional
    def _record(transaction):
        snap = ref.get(transaction=transaction)
        if snap.exists:
            return False
        transaction.set(ref, data)
        return True

    return _record(db.transaction())

    # affiliate revenue share bookkeeping on capture
    if status == "captured" and affiliate_code:
        aff = db.collection("affiliates").document(affiliate_code).get()
        if aff.exists:
            share = float(aff.to_dict().get("revenue_share", 0.2))
            earned = int(amount_stars * share)
            ref_aff = db.collection("affiliates").document(affiliate_code)
            # atomic increment
            from firebase_admin import firestore as fs
            ref_aff.update({"total_earned_stars": fs.Increment(earned)})
            log.info("affiliate %s earned %d stars", affiliate_code, earned)


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

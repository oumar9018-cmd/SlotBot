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


def _now():
    return datetime.now(timezone.utc)


def get_db():
    global _db
    if _db is not None:
        return _db
    import firebase_admin
    from firebase_admin import credentials, firestore

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
    get_db().collection("users").document(str(user_id)).set(data)
    return data


def update_user(user_id: str, fields: dict):
    """Update user profile fields (best-effort)."""
    try:
        get_db().collection("users").document(str(user_id)).update(fields)
    except Exception:
        pass  # doc may not exist yet; create_user handles it


def mark_trial_used(user_id: str):
    get_db().collection("users").document(str(user_id)).update(
        {"free_trial_used": True})


def mark_paid(user_id: str):
    get_db().collection("users").document(str(user_id)).update(
        {"paid": True, "paid_at": _now()})


# ---------- sessions ----------

def create_session(user_id: str, test_type: str, plan: str,
                   max_duration_sec: int, affiliate_code: str | None = None,
                   background_summary: str | None = None) -> str:
    sid = uuid.uuid4().hex[:16]
    get_db().collection("sessions").document(sid).set({
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
    })
    return sid


def get_session(session_id: str) -> dict | None:
    doc = get_db().collection("sessions").document(session_id).get()
    return doc.to_dict() if doc.exists else None


def append_transcript(session_id: str, transcript: list[dict]):
    # Replace with the full accumulated transcript (proxy owns ordering)
    get_db().collection("sessions").document(session_id).update(
        {"transcript": transcript})


def mark_session_ended(session_id: str, duration_sec: int):
    get_db().collection("sessions").document(session_id).update(
        {"ended_at": _now(), "duration_sec": duration_sec})


def update_session_scored(session_id: str, score: int, verdict: str,
                          breakdown: dict, weaknesses: list, feedback: str):
    get_db().collection("sessions").document(session_id).update({
        "score": score,
        "verdict": verdict,
        "breakdown": breakdown,
        "weaknesses": weaknesses,
        "feedback": feedback,
    })


# ---------- stars invoices (pending, created before checkout) ----------

def record_stars_invoice(payload_token: str, user_id: str,
                         affiliate_code: str | None = None):
    get_db().collection("stars_invoices").document(payload_token).set({
        "payload": payload_token,
        "user_id": str(user_id),
        "affiliate_code": affiliate_code,
        "used": False,
        "created_at": _now(),
        "used_at": None,
    })


def get_stars_invoice(payload_token: str) -> dict | None:
    doc = get_db().collection("stars_invoices").document(payload_token).get()
    return doc.to_dict() if doc.exists else None


def mark_invoice_used(payload_token: str):
    get_db().collection("stars_invoices").document(payload_token).update(
        {"used": True, "used_at": _now()})


# ---------- payments (Telegram Stars) ----------

def get_payment(payment_id: str) -> dict | None:
    """Idempotency check — returns existing payment record if any."""
    doc = get_db().collection("payments").document(payment_id).get()
    return doc.to_dict() if doc.exists else None


def record_payment(payment_id: str, user_id: str, amount_stars: int,
                   status: str, affiliate_code: str | None = None):
    db = get_db()
    ref = db.collection("payments").document(payment_id)
    snap = ref.get()
    data = {
        "payment_id": payment_id,
        "user_id": str(user_id),
        "amount_stars": int(amount_stars),
        "status": status,
        "affiliate_code": affiliate_code,
        "created_at": _now(),
    }
    if snap.exists:
        ref.update(data)
    else:
        ref.set(data)

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

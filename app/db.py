"""SQLite storage. All timestamps stored as UTC ISO strings.
Write operations that must be atomic use BEGIN IMMEDIATE to serialize.
"""
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

_lock = threading.RLock()  # reentrant: helpers call get_conn() inside locked sections
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS tutors (
    id            INTEGER PRIMARY KEY,
    tg_user_id    INTEGER UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    timezone      TEXT NOT NULL DEFAULT 'Asia/Kolkata',
    price_stars   INTEGER NOT NULL DEFAULT 200,
    link_code     TEXT UNIQUE NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS availability_rules (
    id         INTEGER PRIMARY KEY,
    tutor_id   INTEGER NOT NULL REFERENCES tutors(id) ON DELETE CASCADE,
    weekday    INTEGER NOT NULL,              -- 0=Monday .. 6=Sunday
    start_min  INTEGER NOT NULL,              -- minutes since midnight, tutor-local
    end_min    INTEGER NOT NULL,
    slot_min   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    id             INTEGER PRIMARY KEY,
    tutor_id       INTEGER NOT NULL REFERENCES tutors(id) ON DELETE CASCADE,
    starts_at_utc  TEXT NOT NULL,
    ends_at_utc    TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open',  -- open|held|booked|blocked
    UNIQUE(tutor_id, starts_at_utc)
);

CREATE TABLE IF NOT EXISTS bookings (
    id               INTEGER PRIMARY KEY,
    tutor_id         INTEGER NOT NULL REFERENCES tutors(id) ON DELETE CASCADE,
    slot_id          INTEGER NOT NULL UNIQUE REFERENCES slots(id) ON DELETE CASCADE,
    student_tg_id    INTEGER NOT NULL,
    student_name     TEXT,
    status           TEXT NOT NULL DEFAULT 'held',  -- held|paid|cancelled|completed|expired
    hold_expires_at  TEXT,
    idempotency_key  TEXT UNIQUE NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    id                  INTEGER PRIMARY KEY,
    booking_id          INTEGER NOT NULL UNIQUE REFERENCES bookings(id) ON DELETE CASCADE,
    telegram_charge_id  TEXT UNIQUE NOT NULL,
    amount_stars        INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'completed',  -- completed|refunded
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id             INTEGER PRIMARY KEY,
    booking_id     INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,   -- 24h|1h
    remind_at_utc  TEXT NOT NULL,
    sent_at        TEXT,
    UNIQUE(booking_id, kind)
);

CREATE TABLE IF NOT EXISTS onboarding_sessions (
    tg_user_id  INTEGER PRIMARY KEY,
    data        TEXT NOT NULL,   -- JSON: survives bot restarts (unlike RAM)
    updated_at  TEXT NOT NULL
);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL;")
            _conn.execute("PRAGMA foreign_keys=ON;")
            _conn.executescript(SCHEMA)
            _conn.commit()
        return _conn


def new_link_code() -> str:
    return secrets.token_urlsafe(6)


# ---------------------------------------------------------------- tutors ---
def create_tutor(tg_user_id: int, name: str, timezone: str) -> dict:
    c = get_conn()
    with _lock:
        cur = c.execute(
            "INSERT INTO tutors (tg_user_id, name, timezone, link_code, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (tg_user_id, name, timezone, new_link_code(), utcnow_iso()),
        )
        c.commit()
        return get_tutor_by_id(cur.lastrowid)


def get_tutor_by_id(tutor_id: int) -> dict | None:
    row = get_conn().execute("SELECT * FROM tutors WHERE id=?", (tutor_id,)).fetchone()
    return dict(row) if row else None


def get_tutor_by_tg(tg_user_id: int) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM tutors WHERE tg_user_id=?", (tg_user_id,)
    ).fetchone()
    return dict(row) if row else None


def get_tutor_by_link(link_code: str) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM tutors WHERE link_code=?", (link_code,)
    ).fetchone()
    return dict(row) if row else None


def update_tutor(tutor_id: int, **fields) -> None:
    allowed = {"name", "timezone", "price_stars"}
    sets = ", ".join(f"{k}=?" for k in fields if k in allowed)
    vals = [v for k, v in fields.items() if k in allowed]
    if not sets:
        return
    with _lock:
        c = get_conn()
        c.execute(f"UPDATE tutors SET {sets} WHERE id=?", (*vals, tutor_id))
        c.commit()


# ------------------------------------------------------- availability ------
def set_availability_rules(tutor_id: int, rules: list[dict]) -> None:
    """Replace all rules. Each rule: weekday, start_min, end_min, slot_min."""
    with _lock:
        c = get_conn()
        c.execute("DELETE FROM availability_rules WHERE tutor_id=?", (tutor_id,))
        for r in rules:
            c.execute(
                "INSERT INTO availability_rules (tutor_id, weekday, start_min, end_min, slot_min)"
                " VALUES (?, ?, ?, ?, ?)",
                (tutor_id, r["weekday"], r["start_min"], r["end_min"], r["slot_min"]),
            )
        c.commit()


def get_availability_rules(tutor_id: int) -> list[dict]:
    rows = get_conn().execute(
        "SELECT * FROM availability_rules WHERE tutor_id=? ORDER BY weekday, start_min",
        (tutor_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------- slots -----
def upsert_slots(tutor_id: int, slots: list[tuple[str, str]]) -> int:
    """Insert (starts_at_utc, ends_at_utc) pairs, ignoring existing. Returns new count."""
    with _lock:
        c = get_conn()
        n = 0
        for s, e in slots:
            cur = c.execute(
                "INSERT OR IGNORE INTO slots (tutor_id, starts_at_utc, ends_at_utc)"
                " VALUES (?, ?, ?)",
                (tutor_id, s, e),
            )
            n += cur.rowcount
        c.commit()
        return n


def open_slots(tutor_id: int, limit: int = 40) -> list[dict]:
    rows = get_conn().execute(
        "SELECT * FROM slots WHERE tutor_id=? AND status='open'"
        " AND starts_at_utc > ? ORDER BY starts_at_utc LIMIT ?",
        (tutor_id, utcnow_iso(), limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_slot(slot_id: int) -> dict | None:
    row = get_conn().execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------ bookings -----
def hold_slot(tutor_id: int, slot_id: int, student_tg_id: int,
              student_name: str | None, hold_minutes: int) -> dict | None:
    """Atomically hold an open slot. Returns booking or None if unavailable."""
    from datetime import timedelta
    with _lock:
        c = get_conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            slot = c.execute(
                "SELECT * FROM slots WHERE id=? AND tutor_id=?", (slot_id, tutor_id)
            ).fetchone()
            if not slot or slot["status"] != "open":
                c.execute("ROLLBACK")
                return None
            if slot["starts_at_utc"] <= utcnow_iso():
                c.execute("ROLLBACK")
                return None
            c.execute("UPDATE slots SET status='held' WHERE id=?", (slot_id,))
            expires = (datetime.now(timezone.utc)
                       + timedelta(minutes=hold_minutes)).isoformat()
            cur = c.execute(
                "INSERT INTO bookings (tutor_id, slot_id, student_tg_id, student_name,"
                " status, hold_expires_at, idempotency_key, created_at)"
                " VALUES (?, ?, ?, ?, 'held', ?, ?, ?)",
                (tutor_id, slot_id, student_tg_id, student_name, expires,
                 secrets.token_hex(16), utcnow_iso()),
            )
            c.execute("COMMIT")
            return get_booking(cur.lastrowid)
        except Exception:
            c.execute("ROLLBACK")
            raise


def get_booking(booking_id: int) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM bookings WHERE id=?", (booking_id,)
    ).fetchone()
    return dict(row) if row else None


def get_booking_by_idem(idempotency_key: str) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM bookings WHERE idempotency_key=?", (idempotency_key,)
    ).fetchone()
    return dict(row) if row else None


def release_expired_holds() -> int:
    """Return expired 'held' bookings to open slots. Returns count released."""
    with _lock:
        c = get_conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            rows = c.execute(
                "SELECT id, slot_id FROM bookings"
                " WHERE status='held' AND hold_expires_at <= ?",
                (utcnow_iso(),),
            ).fetchall()
            for r in rows:
                c.execute("UPDATE bookings SET status='expired' WHERE id=?", (r["id"],))
                c.execute(
                    "UPDATE slots SET status='open' WHERE id=? AND status='held'",
                    (r["slot_id"],),
                )
            c.execute("COMMIT")
            return len(rows)
        except Exception:
            c.execute("ROLLBACK")
            raise


def mark_booking_paid(booking_id: int, telegram_charge_id: str,
                      amount_stars: int) -> bool:
    """Idempotent: returns True if this call completed the payment."""
    with _lock:
        c = get_conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            # Double-payment guard: charge id already seen?
            if c.execute(
                "SELECT 1 FROM payments WHERE telegram_charge_id=?",
                (telegram_charge_id,),
            ).fetchone():
                c.execute("ROLLBACK")
                return False
            b = c.execute(
                "SELECT * FROM bookings WHERE id=?", (booking_id,)
            ).fetchone()
            if not b or b["status"] != "held":
                c.execute("ROLLBACK")
                return False
            c.execute(
                "INSERT INTO payments (booking_id, telegram_charge_id, amount_stars, created_at)"
                " VALUES (?, ?, ?, ?)",
                (booking_id, telegram_charge_id, amount_stars, utcnow_iso()),
            )
            c.execute("UPDATE bookings SET status='paid' WHERE id=?", (booking_id,))
            c.execute(
                "UPDATE slots SET status='booked' WHERE id=?", (b["slot_id"],)
            )
            c.execute("COMMIT")
            return True
        except Exception:
            c.execute("ROLLBACK")
            raise


def cancel_booking(booking_id: int) -> dict | None:
    """Cancel a paid/held booking, free the slot. Returns booking or None."""
    with _lock:
        c = get_conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            b = c.execute(
                "SELECT * FROM bookings WHERE id=?", (booking_id,)
            ).fetchone()
            if not b or b["status"] not in ("held", "paid"):
                c.execute("ROLLBACK")
                return None
            c.execute(
                "UPDATE bookings SET status='cancelled' WHERE id=?", (booking_id,)
            )
            c.execute(
                "UPDATE slots SET status='open' WHERE id=?", (b["slot_id"],)
            )
            c.execute("COMMIT")
            return get_booking(booking_id)
        except Exception:
            c.execute("ROLLBACK")
            raise


def mark_payment_refunded(booking_id: int) -> None:
    with _lock:
        c = get_conn()
        c.execute(
            "UPDATE payments SET status='refunded' WHERE booking_id=?", (booking_id,)
        )
        c.commit()


def get_payment_by_booking(booking_id: int) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM payments WHERE booking_id=?", (booking_id,)
    ).fetchone()
    return dict(row) if row else None


def upcoming_bookings(tutor_id: int) -> list[dict]:
    rows = get_conn().execute(
        """SELECT b.*, s.starts_at_utc, s.ends_at_utc
           FROM bookings b JOIN slots s ON s.id = b.slot_id
           WHERE b.tutor_id=? AND b.status='paid' AND s.starts_at_utc > ?
           ORDER BY s.starts_at_utc""",
        (tutor_id, utcnow_iso()),
    ).fetchall()
    return [dict(r) for r in rows]


def earnings_stars(tutor_id: int) -> int:
    row = get_conn().execute(
        """SELECT COALESCE(SUM(p.amount_stars),0) AS total
           FROM payments p JOIN bookings b ON b.id=p.booking_id
           WHERE b.tutor_id=? AND p.status='completed'""",
        (tutor_id,),
    ).fetchone()
    return int(row["total"])


# ------------------------------------------------------------ reminders -----
def schedule_reminders(booking_id: int, starts_at_utc: str) -> None:
    start = datetime.fromisoformat(starts_at_utc)
    with _lock:
        c = get_conn()
        for kind, mins in (("24h", 24 * 60), ("1h", 60)):
            remind_at = (start - timedelta(minutes=mins)).isoformat()
            # Don't schedule reminders in the past
            if remind_at > utcnow_iso():
                c.execute(
                    "INSERT OR IGNORE INTO reminders (booking_id, kind, remind_at_utc)"
                    " VALUES (?, ?, ?)",
                    (booking_id, kind, remind_at),
                )
        c.commit()


def due_reminders() -> list[dict]:
    rows = get_conn().execute(
        """SELECT r.*, b.student_tg_id, b.tutor_id, s.starts_at_utc
           FROM reminders r
           JOIN bookings b ON b.id = r.booking_id
           JOIN slots s ON s.id = b.slot_id
           WHERE r.sent_at IS NULL AND r.remind_at_utc <= ? AND b.status='paid'""",
        (utcnow_iso(),),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_reminder_sent(reminder_id: int) -> None:
    with _lock:
        c = get_conn()
        c.execute(
            "UPDATE reminders SET sent_at=? WHERE id=?", (utcnow_iso(), reminder_id)
        )
        c.commit()


# --------------------------------------------------- onboarding sessions ---
def save_onboarding_session(tg_user_id: int, data: dict) -> None:
    import json
    with _lock:
        c = get_conn()
        c.execute(
            "INSERT INTO onboarding_sessions (tg_user_id, data, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(tg_user_id) DO UPDATE SET"
            " data=excluded.data, updated_at=excluded.updated_at",
            (tg_user_id, json.dumps(data), utcnow_iso()),
        )
        c.commit()


def load_onboarding_session(tg_user_id: int) -> dict | None:
    import json
    row = get_conn().execute(
        "SELECT data FROM onboarding_sessions WHERE tg_user_id=?", (tg_user_id,)
    ).fetchone()
    return json.loads(row["data"]) if row else None


def clear_onboarding_session(tg_user_id: int) -> None:
    with _lock:
        c = get_conn()
        c.execute("DELETE FROM onboarding_sessions WHERE tg_user_id=?",
                  (tg_user_id,))
        c.commit()

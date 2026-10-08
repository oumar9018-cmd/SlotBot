"""Slot generation from availability rules. Tutor-local times -> UTC storage."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config, db

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def generate_slots_for_tutor(tutor_id: int,
                             horizon_days: int | None = None) -> int:
    """Generate concrete slots for the next N days from rules. Returns new count."""
    tutor = db.get_tutor_by_id(tutor_id)
    if not tutor:
        return 0
    rules = db.get_availability_rules(tutor_id)
    if not rules:
        return 0
    tz = ZoneInfo(tutor["timezone"])
    horizon = horizon_days or config.SLOT_HORIZON_DAYS
    now_local = datetime.now(tz).date()
    pairs: list[tuple[str, str]] = []
    for d in range(horizon):
        day = now_local + timedelta(days=d)
        weekday = day.weekday()
        for r in rules:
            if r["weekday"] != weekday:
                continue
            start_local = datetime(day.year, day.month, day.day, tzinfo=tz).replace(
                hour=r["start_min"] // 60, minute=r["start_min"] % 60)
            end_local = datetime(day.year, day.month, day.day, tzinfo=tz).replace(
                hour=r["end_min"] // 60, minute=r["end_min"] % 60)
            # Skip slots that already started
            now_utc = datetime.now(timezone.utc)
            cur = start_local
            while cur + timedelta(minutes=r["slot_min"]) <= end_local:
                s_utc = cur.astimezone(timezone.utc)
                e_utc = (cur + timedelta(minutes=r["slot_min"])).astimezone(timezone.utc)
                if s_utc > now_utc:
                    pairs.append((s_utc.isoformat(), e_utc.isoformat()))
                cur += timedelta(minutes=r["slot_min"])
    return db.upsert_slots(tutor_id, pairs)


def fmt_slot(slot: dict, tz_name: str) -> str:
    """Human label, e.g. 'Mon 12 Oct, 6:30 PM'. Timezone-aware."""
    tz = ZoneInfo(tz_name)
    s = datetime.fromisoformat(slot["starts_at_utc"]).astimezone(tz)
    return s.strftime("%a %d %b, %-I:%M %p")


def parse_time_range(text: str) -> tuple[int, int] | None:
    """Parse '18:00-21:00' -> (1080, 1260) minutes. Returns None if invalid."""
    try:
        parts = text.strip().replace(" ", "").split("-")
        if len(parts) != 2:
            return None
        def to_min(t: str) -> int:
            h, m = t.split(":")
            h, m = int(h), int(m)
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError
            return h * 60 + m
        start, end = to_min(parts[0]), to_min(parts[1])
        if end <= start:
            return None
        return start, end
    except Exception:
        return None


COMMON_TIMEZONES = [
    "Asia/Kolkata", "Asia/Dubai", "Asia/Singapore", "Asia/Jakarta",
    "Europe/London", "Europe/Berlin", "America/New_York", "America/Chicago",
]

"""Reminder scheduler: 24h + 1h before each paid session.

Runs inside the main process via APScheduler. Also releases expired holds.
"""
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from zoneinfo import ZoneInfo

from . import availability as av
from . import db, telegram as tg

log = logging.getLogger("slotbot.reminders")
_scheduler: AsyncIOScheduler | None = None


async def _send_due_reminders():
    # Housekeeping: free expired holds
    try:
        released = db.release_expired_holds()
        if released:
            log.info("released %d expired holds", released)
    except Exception:
        log.exception("release_expired_holds failed")

    for r in db.due_reminders():
        try:
            tutor = db.get_tutor_by_id(r["tutor_id"])
            slot = {"starts_at_utc": r["starts_at_utc"]}
            label = av.fmt_slot(slot, tutor["timezone"])
            when = "tomorrow" if r["kind"] == "24h" else "in 1 hour"
            text = (
                f"⏰ Reminder: your session is {when}!\n\n"
                f"👩‍🏫 Tutor: {tutor['name']}\n"
                f"🕐 {label} ({tutor['timezone']})"
            )
            await tg.send_message(r["student_tg_id"], text)
            student_name = db.get_booking(r["booking_id"])["student_name"]
            await tg.send_message(
                tutor["tg_user_id"],
                f"⏰ Reminder: session {when} with {student_name}\n🕐 {label}",
            )
            db.mark_reminder_sent(r["id"])
            log.info("reminder sent: booking=%d kind=%s", r["booking_id"], r["kind"])
        except Exception:
            log.exception("reminder failed: %s", r["id"])


def start():
    global _scheduler
    if _scheduler:
        return
    _scheduler = AsyncIOScheduler(timezone=ZoneInfo("UTC"))
    _scheduler.add_job(_send_due_reminders, "interval", seconds=60,
                       max_instances=1, coalesce=True)
    _scheduler.start()
    log.info("reminder scheduler started (60s interval)")


def stop():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None

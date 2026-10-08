"""DB layer tests: schema, double-booking prevention, idempotent payments."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use a throwaway DB for tests
os.environ["DB_PATH"] = "/tmp/slotbot_test.db"
for f in ("/tmp/slotbot_test.db", "/tmp/slotbot_test.db-wal", "/tmp/slotbot_test.db-shm"):
    if os.path.exists(f):
        os.remove(f)

from app import db


def main():
    t = db.create_tutor(111, "Asha", "Asia/Kolkata")
    assert t["name"] == "Asha" and t["link_code"], "tutor create"

    db.set_availability_rules(t["id"], [
        {"weekday": 0, "start_min": 1080, "end_min": 1260, "slot_min": 60},
    ])
    assert len(db.get_availability_rules(t["id"])) == 1, "rules"

    n = db.upsert_slots(t["id"], [("2026-10-12T12:30:00+00:00", "2026-10-12T13:30:00+00:00")])
    assert n == 1, "slot insert"
    n2 = db.upsert_slots(t["id"], [("2026-10-12T12:30:00+00:00", "2026-10-12T13:30:00+00:00")])
    assert n2 == 0, "slot dedup"
    slots = db.open_slots(t["id"])
    assert len(slots) == 1, "open_slots"

    b1 = db.hold_slot(t["id"], slots[0]["id"], 222, "Ravi", 10)
    assert b1 and b1["status"] == "held", "hold"
    b2 = db.hold_slot(t["id"], slots[0]["id"], 333, "Sneha", 10)
    assert b2 is None, "DOUBLE BOOKING NOT PREVENTED"

    assert db.mark_booking_paid(b1["id"], "charge_abc", 200) is True, "mark paid"
    assert db.mark_booking_paid(b1["id"], "charge_abc", 200) is False, "DUP PAYMENT"
    assert db.mark_booking_paid(b1["id"], "charge_xyz", 200) is False, "2nd pay on paid"

    assert db.earnings_stars(t["id"]) == 200, "earnings"
    db.schedule_reminders(b1["id"], "2026-10-12T12:30:00+00:00")
    assert len(db.upcoming_bookings(t["id"])) == 1, "upcoming"

    # expired hold release
    b3slot = db.upsert_slots(t["id"], [("2026-10-13T12:30:00+00:00", "2026-10-13T13:30:00+00:00")])
    s2 = [s for s in db.open_slots(t["id"], 10) if s["starts_at_utc"].startswith("2026-10-13")][0]
    b3 = db.hold_slot(t["id"], s2["id"], 444, "Meena", hold_minutes=-1)  # already expired
    assert b3 is not None
    released = db.release_expired_holds()
    assert released >= 1, "expired release"
    assert db.get_slot(s2["id"])["status"] == "open", "slot reopened"

    print("ALL DB TESTS PASSED")


if __name__ == "__main__":
    main()

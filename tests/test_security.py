"""Security + regression tests: tutor isolation, concurrent double-booking,
pre-checkout reject path, API auth.
"""
import asyncio
import hashlib
import hmac
import json
import os
import sys
import threading
import time
from urllib.parse import urlencode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DB_PATH"] = "/tmp/slotbot_sec.db"
for f in ("/tmp/slotbot_sec.db", "/tmp/slotbot_sec.db-wal", "/tmp/slotbot_sec.db-shm"):
    if os.path.exists(f):
        os.remove(f)

from app import bot, config, db, security
from app import telegram as tg

config.BOT_TOKEN = "123456:TESTTOKEN"
CALLS = {"precheckout": []}


async def fake_answer_precheckout(qid, ok, error_message=None):
    CALLS["precheckout"].append({"qid": qid, "ok": ok, "err": error_message})
    return {"ok": True}


async def fake_answer_callback(cqid, text=None, show_alert=False):
    return {"ok": True}


async def fake_send_message(chat_id, text, reply_markup=None, parse_mode=None):
    return {"ok": True}


tg.answer_precheckout = fake_answer_precheckout
tg.answer_callback = fake_answer_callback
tg.send_message = fake_send_message


def make_init_data(user_id: int) -> str:
    user = {"id": user_id, "first_name": "T"}
    pairs = {"auth_date": str(int(time.time())),
             "user": json.dumps(user, separators=(",", ":"))}
    dcs = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    sk = hmac.new(b"WebAppData", config.BOT_TOKEN.encode(), hashlib.sha256).digest()
    pairs["hash"] = hmac.new(sk, dcs.encode(), hashlib.sha256).hexdigest()
    return urlencode(pairs)


async def main():
    # ---- two tutors, isolation ----
    t1 = db.create_tutor(101, "TutorA", "Asia/Kolkata")
    t2 = db.create_tutor(102, "TutorB", "Asia/Kolkata")
    db.set_availability_rules(t1["id"], [
        {"weekday": 0, "start_min": 1080, "end_min": 1200, "slot_min": 60}])
    db.set_availability_rules(t2["id"], [
        {"weekday": 0, "start_min": 1080, "end_min": 1200, "slot_min": 60}])
    from app import availability as av
    av.generate_slots_for_tutor(t1["id"])
    av.generate_slots_for_tutor(t2["id"])

    # Tutor A's initData must resolve to Tutor A only
    from app.miniapp import _tutor_from_initdata
    me_a = _tutor_from_initdata(make_init_data(101))
    me_b = _tutor_from_initdata(make_init_data(102))
    assert me_a["id"] == t1["id"] and me_b["id"] == t2["id"], "isolation fail"
    # Non-tutor gets 403
    try:
        _tutor_from_initdata(make_init_data(999))
        assert False, "non-tutor not rejected"
    except Exception as e:
        assert "403" in str(e), f"wrong rejection: {e}"
    # Tampered initData -> 401
    try:
        _tutor_from_initdata(make_init_data(101) + "x")
        assert False, "tampered not rejected"
    except Exception as e:
        assert "401" in str(e), f"wrong rejection: {e}"
    print("1. tutor isolation + initData auth OK")

    # ---- concurrent double-booking: 10 threads, 1 slot, exactly 1 wins ----
    slot = db.open_slots(t1["id"])[0]
    winners = []

    def grab(i):
        b = db.hold_slot(t1["id"], slot["id"], 500 + i, f"S{i}", 10)
        if b:
            winners.append(b["id"])

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(winners) == 1, f"concurrent double-booking! winners={winners}"
    print("2. concurrent double-booking OK (1 winner)")

    # ---- pre-checkout rejects expired hold ----
    b = db.get_booking(winners[0])
    # force-expire the hold
    c = db.get_conn()
    c.execute("UPDATE bookings SET hold_expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
              (b["id"],))
    c.commit()
    await bot.handle_precheckout({"id": "pqX", "invoice_payload": f"book:{b['id']}"})
    pc = CALLS["precheckout"][-1]
    assert pc["ok"] is False, "expired hold accepted!"
    # slot released by the sweeper inside the reject path
    assert db.get_slot(b["slot_id"])["status"] == "open", "slot not released"
    print("3. pre-checkout reject on expired hold OK")

    # ---- pre-checkout rejects already-paid booking ----
    slot2 = db.open_slots(t1["id"])[1]
    b2 = db.hold_slot(t1["id"], slot2["id"], 600, "Zed", 10)
    assert db.mark_booking_paid(b2["id"], "ch_paid", 100)
    await bot.handle_precheckout({"id": "pqY", "invoice_payload": f"book:{b2['id']}"})
    pc = CALLS["precheckout"][-1]
    assert pc["ok"] is False, "paid booking re-accepted!"
    print("4. pre-checkout reject on paid booking OK")

    # ---- stale onboarding callback -> graceful, no crash ----
    _uid = [99]

    async def fake_send2(chat_id, text, reply_markup=None, parse_mode=None):
        CALLS.setdefault("msgs", []).append(text)
        return {"ok": True}

    tg.send_message = fake_send2
    await bot.handle_update({"update_id": 100, "callback_query":
        {"id": "cqZ", "from": {"id": 777, "first_name": "Ghost"},
         "message": {"message_id": 1}, "data": "ob_dur_60"}})
    assert any("expired" in m for m in CALLS["msgs"]), "no graceful expiry msg"
    print("5. stale callback graceful OK")

    print("ALL SECURITY TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())

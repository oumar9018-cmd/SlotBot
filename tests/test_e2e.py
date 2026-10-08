"""End-to-end test of the core loop with Telegram network mocked.

Tutor onboarding -> availability -> student books slot -> Stars payment
-> confirmation -> (reminders scheduled). Also: double-payment guard,
tutor cancel + refund, initData validation.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DB_PATH"] = "/tmp/slotbot_e2e.db"
for f in ("/tmp/slotbot_e2e.db", "/tmp/slotbot_e2e.db-wal", "/tmp/slotbot_e2e.db-shm"):
    if os.path.exists(f):
        os.remove(f)

from app import availability as av
from app import bot, config, db, security
from app import telegram as tg

config.BOT_TOKEN = "123456:TESTTOKEN"
config.BOT_USERNAME = "slotbot_test"

CALLS = {"send_message": [], "send_invoice": [], "answer_precheckout": [],
         "answer_callback": [], "refund": [], "edit": []}


async def fake_send_message(chat_id, text, reply_markup=None, parse_mode=None):
    CALLS["send_message"].append({"chat_id": chat_id, "text": text,
                                  "reply_markup": reply_markup})
    return {"ok": True}


async def fake_send_invoice(chat_id, title, description, payload, amount_stars):
    CALLS["send_invoice"].append({"chat_id": chat_id, "payload": payload,
                                  "amount": amount_stars})
    return {"ok": True}


async def fake_answer_precheckout(qid, ok, error_message=None):
    CALLS["answer_precheckout"].append({"qid": qid, "ok": ok,
                                        "err": error_message})
    return {"ok": True}


async def fake_answer_callback(cqid, text=None, show_alert=False):
    CALLS["answer_callback"].append({"cqid": cqid, "text": text})
    return {"ok": True}


async def fake_refund(user_id, charge_id):
    CALLS["refund"].append({"user_id": user_id, "charge_id": charge_id})
    return {"ok": True, "result": True}


async def fake_edit(chat_id, message_id, text, reply_markup=None):
    CALLS["edit"].append({"text": text})
    return {"ok": True}


tg.send_message = fake_send_message
tg.send_invoice = fake_send_invoice
tg.answer_precheckout = fake_answer_precheckout
tg.answer_callback = fake_answer_callback
tg.refund_stars = fake_refund
tg.edit_message = fake_edit

TUTOR = 111
STUDENT = 222
_uid = [0]


def msg_update(text, uid, name="Asha"):
    _uid[0] += 1
    return {"update_id": _uid[0],
            "message": {"message_id": _uid[0], "chat": {"id": uid},
                        "from": {"id": uid, "first_name": name}, "text": text}}


def cq_update(uid, data, name="Asha"):
    _uid[0] += 1
    return {"update_id": _uid[0],
            "callback_query": {"id": f"cq{_uid[0]}", "from": {"id": uid, "first_name": name},
                               "message": {"message_id": 9}, "data": data}}


async def main():
    # ---- 1. tutor onboarding ----
    await bot.handle_update(msg_update("/start", TUTOR))
    assert bot._sess_get(TUTOR)["step"] == "name"
    await bot.handle_update(msg_update("Asha Sharma", TUTOR))
    await bot.handle_update(cq_update(TUTOR, "ob_tz_0"))          # Asia/Kolkata
    await bot.handle_update(msg_update("200", TUTOR))             # price
    await bot.handle_update(cq_update(TUTOR, "ob_day_0"))         # Mon
    await bot.handle_update(cq_update(TUTOR, "ob_day_2"))         # Wed
    await bot.handle_update(cq_update(TUTOR, "ob_days_done"))
    await bot.handle_update(msg_update("18:00-21:00", TUTOR))
    await bot.handle_update(cq_update(TUTOR, "ob_dur_60"))
    tutor = db.get_tutor_by_tg(TUTOR)
    assert tutor and tutor["name"] == "Asha Sharma", "tutor not created"
    assert tutor["price_stars"] == 200
    assert len(db.get_availability_rules(tutor["id"])) == 2
    slots = db.open_slots(tutor["id"])
    assert len(slots) > 0, "no slots generated"
    # booking link was sent
    assert any("t.me/" in c["text"] for c in CALLS["send_message"]), "no booking link"
    print(f"1. onboarding OK ({len(slots)} slots, link sent)")

    # ---- 2. student opens booking link, picks slot ----
    await bot.handle_update(msg_update(f"/start book_{tutor['link_code']}", STUDENT, "Ravi"))
    assert any("callback_data" in str(c.get("reply_markup", "")) for c in CALLS["send_message"]), "no slot buttons"
    slot = slots[0]
    await bot.handle_update(cq_update(STUDENT, f"slot_{slot['id']}", "Ravi"))
    inv = CALLS["send_invoice"][-1]
    assert inv["chat_id"] == STUDENT and inv["amount"] == 200, "invoice wrong"
    booking_id = int(inv["payload"].split(":")[1])
    booking = db.get_booking(booking_id)
    assert booking["status"] == "held", "not held"
    print(f"2. hold OK (booking {booking_id}, invoice {inv['amount']} Stars)")

    # ---- 3. pre-checkout (must answer within 10s; we answer fast) ----
    _uid[0] += 1
    await bot.handle_update({"update_id": _uid[0], "pre_checkout_query":
        {"id": "pq1", "invoice_payload": f"book:{booking_id}"}})
    pc = CALLS["answer_precheckout"][-1]
    assert pc["ok"] is True, f"precheckout rejected: {pc}"
    print("3. pre-checkout OK")

    # ---- 4. successful payment -> confirmations + reminders ----
    _uid[0] += 1
    await bot.handle_update({"update_id": _uid[0], "message":
        {"message_id": 50, "chat": {"id": STUDENT},
         "from": {"id": STUDENT, "first_name": "Ravi"},
         "successful_payment": {"invoice_payload": f"book:{booking_id}",
                                "telegram_payment_charge_id": "ch_001",
                                "total_amount": 200}}})
    booking = db.get_booking(booking_id)
    assert booking["status"] == "paid", "not paid"
    assert db.get_slot(slot["id"])["status"] == "booked"
    assert db.earnings_stars(tutor["id"]) == 200
    # both parties notified
    texts = [c["text"] for c in CALLS["send_message"]]
    assert any("Booking confirmed" in t for t in texts), "student not confirmed"
    assert any("New booking" in t for t in texts), "tutor not notified"
    print("4. payment + confirmations OK")

    # ---- 5. duplicate successful_payment ignored ----
    n_before = len(CALLS["send_message"])
    _uid[0] += 1
    await bot.handle_update({"update_id": _uid[0], "message":
        {"message_id": 51, "chat": {"id": STUDENT},
         "from": {"id": STUDENT, "first_name": "Ravi"},
         "successful_payment": {"invoice_payload": f"book:{booking_id}",
                                "telegram_payment_charge_id": "ch_001",
                                "total_amount": 200}}})
    assert db.earnings_stars(tutor["id"]) == 200, "DOUBLE PAYMENT COUNTED"
    assert len(CALLS["send_message"]) == n_before, "duplicate re-notified"
    print("5. duplicate payment guard OK")

    # ---- 6. tutor cancel -> slot freed + refund ----
    await bot.handle_update(cq_update(TUTOR, f"tutor_cancel_{booking_id}"))
    assert CALLS["refund"], "no refund issued"
    assert CALLS["refund"][-1]["charge_id"] == "ch_001"
    assert db.get_booking(booking_id)["status"] == "cancelled"
    assert db.get_slot(slot["id"])["status"] == "open", "slot not freed"
    print("6. tutor cancel + refund OK")

    # ---- 7. initData validation ----
    import hashlib, hmac, json, time
    from urllib.parse import urlencode
    user = {"id": TUTOR, "first_name": "Asha"}
    pairs = {"auth_date": str(int(time.time())), "user": json.dumps(user, separators=(",", ":"))}
    dcs = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    sk = hmac.new(b"WebAppData", config.BOT_TOKEN.encode(), hashlib.sha256).digest()
    pairs["hash"] = hmac.new(sk, dcs.encode(), hashlib.sha256).hexdigest()
    init_data = urlencode(pairs)
    assert security.validate_init_data(init_data)["id"] == TUTOR, "valid initData rejected"
    assert security.validate_init_data(init_data + "tampered") is None, "tampered accepted"
    assert security.verify_webhook_secret({"x-telegram-bot-api-secret-token": "nope"}) is False
    print("7. initData validation OK")

    print("ALL E2E TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())

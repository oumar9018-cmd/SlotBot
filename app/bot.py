"""Telegram update handlers: tutor onboarding, student booking, Stars payments.

V1 scope only. No CRM, no packages, no AI chatbot.
"""
import logging

from . import availability as av
from . import config, db, telegram as tg

log = logging.getLogger("slotbot.bot")

# Onboarding state is persisted in the DB (survives restarts).
# `days` is a set in memory, stored as a sorted list in JSON.
def _sess_get(tg_user_id: int) -> dict | None:
    s = db.load_onboarding_session(tg_user_id)
    if s and isinstance(s.get("days"), list):
        s["days"] = set(s["days"])
    return s


def _sess_save(tg_user_id: int, s: dict) -> None:
    d = dict(s)
    if isinstance(d.get("days"), set):
        d["days"] = sorted(d["days"])
    db.save_onboarding_session(tg_user_id, d)


def _sess_clear(tg_user_id: int) -> None:
    db.clear_onboarding_session(tg_user_id)

TERMS_TEXT = (
    "Terms of Service\n\n"
    "SlotBot lets tutors offer bookable sessions and lets students book and pay "
    "for them with Telegram Stars.\n\n"
    "• Payments are processed by Telegram in Telegram Stars.\n"
    "• If a tutor cancels a paid booking, the payment is refunded in Stars.\n"
    "• SlotBot is a scheduling tool; the tutoring service itself is provided "
    "by the tutor, not by SlotBot.\n"
    "• Support: contact the bot owner via /support."
)

SUPPORT_TEXT = (
    "Need help with a payment or booking? Reply to this message describing "
    "the issue and the bot owner will respond.\n\n"
    "Telegram does not handle purchases made via bots — contact us here."
)


# ------------------------------------------------------------------ helpers
def _btn(text: str, data: str) -> dict:
    return {"text": text, "callback_data": data}


def _dashboard_btn():
    """Web App button only when a real absolute HTTPS URL is configured.

    In local/polling test mode (BASE_URL empty) Telegram rejects relative URLs
    like '/miniapp/' and, worse, rejects the ENTIRE message — so the onboarding
    completion + booking link message never reaches the tutor. Omit the button
    instead of breaking the message.
    """
    base = (config.BASE_URL or "").strip()
    if base.startswith("https://"):
        return tg.webapp_button("📊 Dashboard", config.miniapp_url())
    return None


def _tutor_home_kb(tutor: dict) -> dict:
    rows = [
        [_btn("📅 My bookings", "tutor_bookings"),
         _btn("🔗 My booking link", "tutor_link")],
    ]
    dash = _dashboard_btn()
    if dash:
        rows.append([dash])
    return tg.inline_keyboard(rows)


async def _send_tutor_home(tg_user_id: int):
    tutor = db.get_tutor_by_tg(tg_user_id)
    if not tutor:
        return
    upcoming = db.upcoming_bookings(tutor["id"])
    await tg.send_message(
        tg_user_id,
        f"Welcome back, {tutor['name']}! 👋\n"
        f"You have {len(upcoming)} upcoming booking(s).\n"
        f"Session price: {tutor['price_stars']} Stars.",
        reply_markup=_tutor_home_kb(tutor),
    )


# ------------------------------------------------------------- onboarding
async def start_onboarding(tg_user_id: int, first_name: str):
    _sess_save(tg_user_id, {"step": "name"})
    await tg.send_message(
        tg_user_id,
        f"Hi {first_name}! 👋 Welcome to SlotBot.\n\n"
        "I'll set up your booking page in under 2 minutes.\n\n"
        "What's your name (as students should see it)?",
    )


async def handle_onboarding_text(tg_user_id: int, text: str, first_name: str):
    s = _sess_get(tg_user_id)
    if not s:
        return False
    step = s["step"]

    if step == "name":
        s["name"] = text.strip()[:60] or first_name
        s["step"] = "timezone"
        _sess_save(tg_user_id, s)
        kb = tg.inline_keyboard(
            [[_btn(z, f"ob_tz_{i}")] for i, z in enumerate(av.COMMON_TIMEZONES)]
        )
        await tg.send_message(tg_user_id,
                              "What's your timezone?", reply_markup=kb)
        return True

    if step == "price":
        try:
            price = int(text.strip())
            if not 1 <= price <= 100000:
                raise ValueError
        except ValueError:
            await tg.send_message(tg_user_id,
                                  "Please send a number, e.g. 200 (in Stars).")
            return True
        s["price"] = price
        s["step"] = "days"
        s["days"] = set()
        _sess_save(tg_user_id, s)
        await _send_days_kb(tg_user_id, s)
        return True

    if step == "timerange":
        parsed = av.parse_time_range(text)
        if not parsed:
            await tg.send_message(tg_user_id,
                                  "I didn't get that. Send like: 18:00-21:00")
            return True
        s["start_min"], s["end_min"] = parsed
        s["step"] = "duration"
        _sess_save(tg_user_id, s)
        kb = tg.inline_keyboard([[
            _btn("30 min", "ob_dur_30"), _btn("45 min", "ob_dur_45"),
            _btn("60 min", "ob_dur_60"), _btn("90 min", "ob_dur_90"),
        ]])
        await tg.send_message(tg_user_id,
                              "How long is one session?", reply_markup=kb)
        return True

    return False


async def _send_days_kb(tg_user_id: int, s: dict):
    rows = []
    for i, name in enumerate(av.DAY_NAMES):
        mark = "✅" if i in s["days"] else "⬜"
        rows.append(_btn(f"{mark} {name}", f"ob_day_{i}"))
    kb_rows = [rows[i:i + 4] for i in range(0, 7, 4)]
    kb_rows.append([_btn("Done ✅", "ob_days_done")])
    await tg.send_message(
        tg_user_id,
        "Which days do you teach? Tap to select, then Done.",
        reply_markup=tg.inline_keyboard(kb_rows),
    )


async def handle_onboarding_callback(tg_user_id: int, cq_id: str, data: str,
                                     message_id: int | None) -> bool:
    s = _sess_get(tg_user_id)
    if not s:
        return False

    if data.startswith("ob_tz_"):
        idx = int(data.split("_")[-1])
        s["timezone"] = av.COMMON_TIMEZONES[idx]
        s["step"] = "price"
        _sess_save(tg_user_id, s)
        await tg.answer_callback(cq_id)
        await tg.send_message(
            tg_user_id,
            f"Timezone: {s['timezone']} 🌍\n\n"
            "What is your price per session in Telegram Stars?\n"
            "(~200 Stars ≈ $3. Example: send 200)")
        return True

    if data.startswith("ob_day_"):
        i = int(data.split("_")[-1])
        s["days"].discard(i) if i in s["days"] else s["days"].add(i)
        _sess_save(tg_user_id, s)
        await tg.answer_callback(cq_id)
        # rebuild the keyboard in place
        rows = []
        for j, name in enumerate(av.DAY_NAMES):
            mark = "✅" if j in s["days"] else "⬜"
            rows.append(_btn(f"{mark} {name}", f"ob_day_{j}"))
        kb_rows = [rows[k:k + 4] for k in range(0, 7, 4)]
        kb_rows.append([_btn("Done ✅", "ob_days_done")])
        if message_id:
            await tg.edit_message(tg_user_id, message_id,
                                  "Which days do you teach? Tap to select, then Done.",
                                  reply_markup=tg.inline_keyboard(kb_rows))
        return True

    if data == "ob_days_done":
        if not s["days"]:
            await tg.answer_callback(cq_id, "Pick at least one day!", True)
            return True
        s["step"] = "timerange"
        _sess_save(tg_user_id, s)
        await tg.answer_callback(cq_id)
        await tg.send_message(
            tg_user_id,
            "What hours? Send like: 18:00-21:00")
        return True

    if data.startswith("ob_dur_"):
        mins = int(data.split("_")[-1])
        await tg.answer_callback(cq_id)
        await _finish_onboarding(tg_user_id, s, mins)
        return True

    return False


async def _finish_onboarding(tg_user_id: int, s: dict, slot_min: int):
    rules = [{"weekday": d, "start_min": s["start_min"],
              "end_min": s["end_min"], "slot_min": slot_min}
             for d in sorted(s["days"])]
    tutor = db.create_tutor(tg_user_id, s["name"], s["timezone"])
    db.update_tutor(tutor["id"], price_stars=s["price"])
    db.set_availability_rules(tutor["id"], rules)
    n = av.generate_slots_for_tutor(tutor["id"])
    _sess_clear(tg_user_id)
    tutor = db.get_tutor_by_id(tutor["id"])
    link = config.booking_deeplink(tutor["link_code"])
    await tg.send_message(
        tg_user_id,
        f"🎉 You're live, {tutor['name']}!\n\n"
        f"Generated {n} bookable slots for the next 14 days.\n\n"
        f"Share this link with your students:\n{link}\n\n"
        "They pick a slot, pay in Stars, and you both get confirmations + reminders.",
        reply_markup=_tutor_home_kb(tutor),
    )
    log.info("tutor onboarded: %s (%d slots)", tutor["name"], n)


# ------------------------------------------------------------ student flow
async def start_booking(tg_user_id: int, link_code: str, first_name: str):
    tutor = db.get_tutor_by_link(link_code)
    if not tutor:
        await tg.send_message(tg_user_id,
                              "This booking link is invalid. Ask your tutor for a fresh one.")
        return
    slots = db.open_slots(tutor["id"], limit=12)
    if not slots:
        await tg.send_message(
            tg_user_id,
            f"{tutor['name']} has no open slots right now. Check back soon!")
        return
    rows = [[_btn(av.fmt_slot(sl, tutor["timezone"]), f"slot_{sl['id']}")]
            for sl in slots]
    await tg.send_message(
        tg_user_id,
        f"📚 Book a session with {tutor['name']}\n"
        f"💫 Price: {tutor['price_stars']} Stars per session\n"
        f"🕐 Times shown in {tutor['timezone']}\n\n"
        "Pick a slot:",
        reply_markup=tg.inline_keyboard(rows),
    )


async def handle_slot_pick(tg_user_id: int, cq_id: str, slot_id: int,
                           first_name: str, username: str | None):
    slot = db.get_slot(slot_id)
    if not slot:
        await tg.answer_callback(cq_id, "Slot not found.", True)
        return
    tutor = db.get_tutor_by_id(slot["tutor_id"])
    booking = db.hold_slot(tutor["id"], slot_id, tg_user_id,
                           first_name or username or "Student",
                           config.HOLD_MINUTES)
    if not booking:
        await tg.answer_callback(
            cq_id, "Someone just took this slot! Pick another one.", True)
        return
    await tg.answer_callback(cq_id, "Slot reserved for 10 minutes ⏳")
    label = av.fmt_slot(slot, tutor["timezone"])
    await tg.send_invoice(
        tg_user_id,
        title=f"Session with {tutor['name']}",
        description=f"{label} ({tutor['timezone']})",
        payload=f"book:{booking['id']}",
        amount_stars=tutor["price_stars"],
    )
    log.info("hold created: booking=%d slot=%d", booking["id"], slot_id)


# ---------------------------------------------------------------- payments
async def handle_precheckout(query: dict):
    # NOTE: the update object uses field `id`; the answerPreCheckoutQuery
    # *method* takes parameter `pre_checkout_query_id`. Don't confuse them.
    qid = query["id"]
    payload = query.get("invoice_payload", "")
    if not payload.startswith("book:"):
        await tg.answer_precheckout(qid, False, "Invalid invoice.")
        return
    try:
        booking_id = int(payload.split(":")[1])
    except ValueError:
        await tg.answer_precheckout(qid, False, "Invalid invoice.")
        return
    booking = db.get_booking(booking_id)
    # Must be a live hold: not expired, slot still held for this booking
    if not booking or booking["status"] != "held":
        await tg.answer_precheckout(qid, False,
                                    "This slot is no longer available.")
        return
    from datetime import datetime, timezone
    if booking["hold_expires_at"] <= datetime.now(timezone.utc).isoformat():
        db.release_expired_holds()
        await tg.answer_precheckout(qid, False,
                                    "Reservation expired. Please pick a slot again.")
        return
    await tg.answer_precheckout(qid, True)


async def handle_successful_payment(message: dict):
    from . import reminders as rem_mod
    pay = message["successful_payment"]
    payload = pay.get("invoice_payload", "")
    if not payload.startswith("book:"):
        return
    booking_id = int(payload.split(":")[1])
    charge_id = pay["telegram_payment_charge_id"]
    amount = pay["total_amount"]
    done = db.mark_booking_paid(booking_id, charge_id, amount)
    if not done:
        log.warning("duplicate/unknown successful_payment ignored: %s", charge_id)
        return
    booking = db.get_booking(booking_id)
    slot = db.get_slot(booking["slot_id"])
    tutor = db.get_tutor_by_id(booking["tutor_id"])
    db.schedule_reminders(booking_id, slot["starts_at_utc"])
    label = av.fmt_slot(slot, tutor["timezone"])
    # Student confirmation
    await tg.send_message(
        message["chat"]["id"],
        f"✅ Booking confirmed!\n\n"
        f"👩‍🏫 Tutor: {tutor['name']}\n"
        f"🕐 {label} ({tutor['timezone']})\n"
        f"💫 Paid: {amount} Stars\n\n"
        "You'll get reminders 24h and 1h before. See you there!",
    )
    # Tutor notification
    await tg.send_message(
        tutor["tg_user_id"],
        f"💰 New booking!\n\n"
        f"👤 Student: {booking['student_name']}\n"
        f"🕐 {label}\n"
        f"💫 {amount} Stars received",
        reply_markup=_tutor_home_kb(tutor),
    )
    log.info("booking paid: %d charge=%s", booking_id, charge_id)


# ----------------------------------------------------------- tutor actions
async def handle_tutor_callback(tg_user_id: int, cq_id: str, data: str,
                                message_id: int | None) -> bool:
    tutor = db.get_tutor_by_tg(tg_user_id)
    if not tutor:
        return False

    if data == "tutor_bookings":
        await tg.answer_callback(cq_id)
        upcoming = db.upcoming_bookings(tutor["id"])
        if not upcoming:
            await tg.send_message(tg_user_id, "No upcoming bookings yet.")
            return True
        for b in upcoming[:10]:
            kb = tg.inline_keyboard(
                [[_btn("❌ Cancel + refund", f"tutor_cancel_{b['id']}")]])
            await tg.send_message(
                tg_user_id,
                f"👤 {b['student_name']}\n🕐 {av.fmt_slot(b, tutor['timezone'])}",
                reply_markup=kb,
            )
        return True

    if data == "tutor_link":
        await tg.answer_callback(cq_id)
        await tg.send_message(tg_user_id,
                              f"Your booking link:\n{config.booking_deeplink(tutor['link_code'])}")
        return True

    if data.startswith("tutor_cancel_"):
        booking_id = int(data.split("_")[-1])
        booking = db.get_booking(booking_id)
        if not booking or booking["tutor_id"] != tutor["id"]:
            await tg.answer_callback(cq_id, "Booking not found.", True)
            return True
        cancelled = db.cancel_booking(booking_id)
        if not cancelled:
            await tg.answer_callback(cq_id, "Already cancelled.", True)
            return True
        # Refund the Stars payment if one exists
        payment = db.get_payment_by_booking(booking_id)
        if payment and payment["status"] == "completed":
            r = await tg.refund_stars(booking["student_tg_id"],
                                      payment["telegram_charge_id"])
            if r.get("ok"):
                db.mark_payment_refunded(booking_id)
        await tg.answer_callback(cq_id, "Cancelled + refunded ✅")
        await tg.send_message(
            booking["student_tg_id"],
            "⚠️ Your session was cancelled by the tutor. "
            "Your Stars have been refunded.",
        )
        if message_id:
            await tg.edit_message(tg_user_id, message_id,
                                  "❌ Cancelled + refunded.")
        return True

    return False


# ------------------------------------------------------------------ router
async def handle_update(update: dict):
    try:
        if "pre_checkout_query" in update:
            await handle_precheckout(update["pre_checkout_query"])
            return

        if "callback_query" in update:
            cq = update["callback_query"]
            data = cq.get("data", "")
            from_id = cq["from"]["id"]
            msg_id = cq.get("message", {}).get("message_id")
            if data.startswith("ob_"):
                if await handle_onboarding_callback(from_id, cq["id"], data, msg_id):
                    return
                # Session gone (e.g., never started) — don't show "Unknown action"
                await tg.answer_callback(cq["id"])
                await tg.send_message(
                    from_id,
                    "Your setup session expired. Send /start to begin again 🔄")
                return
            if await handle_onboarding_callback(from_id, cq["id"], data, msg_id):
                return
            if data.startswith("slot_"):
                await handle_slot_pick(from_id, cq["id"], int(data[5:]),
                                       cq["from"].get("first_name", ""),
                                       cq["from"].get("username"))
                return
            if await handle_tutor_callback(from_id, cq["id"], data, msg_id):
                return
            await tg.answer_callback(cq["id"], "Unknown action.", True)
            return

        msg = update.get("message")
        if not msg:
            return
        chat_id = msg["chat"]["id"]
        from_id = msg["from"]["id"]
        first_name = msg["from"].get("first_name", "")

        if "successful_payment" in msg:
            await handle_successful_payment(msg)
            return

        text = msg.get("text", "")
        if text.startswith("/start"):
            parts = text.split(maxsplit=1)
            arg = parts[1] if len(parts) > 1 else ""
            if arg.startswith("book_"):
                await start_booking(from_id, arg[5:], first_name)
            elif db.get_tutor_by_tg(from_id):
                await _send_tutor_home(from_id)
            else:
                await start_onboarding(from_id, first_name or "there")
            return

        if text == "/terms":
            await tg.send_message(chat_id, TERMS_TEXT)
            return
        if text in ("/paysupport", "/support"):
            await tg.send_message(chat_id, SUPPORT_TEXT)
            return
        if text == "/dashboard" or text == "/bookings":
            tutor = db.get_tutor_by_tg(from_id)
            if tutor:
                dash = _dashboard_btn()
                if dash:
                    await tg.send_message(
                        chat_id, "Open your dashboard:",
                        reply_markup=tg.inline_keyboard([[dash]]))
                else:
                    await tg.send_message(
                        chat_id, "Dashboard Mini App needs a public BASE_URL (deploy to Fly.io first). "
                                "Your bookings still work fine from the buttons below.")
            else:
                await tg.send_message(chat_id, "You're not registered as a tutor yet. Send /start.")
            return
        if text == "/mylink":
            tutor = db.get_tutor_by_tg(from_id)
            if tutor:
                await tg.send_message(
                    chat_id,
                    f"Your booking link:\n{config.booking_deeplink(tutor['link_code'])}")
            return

        # Freeform text during onboarding
        if await handle_onboarding_text(from_id, text, first_name or "there"):
            return

        # Fallback
        tutor = db.get_tutor_by_tg(from_id)
        if tutor:
            await _send_tutor_home(from_id)
        else:
            await tg.send_message(chat_id,
                                  "Send /start to set up your tutor booking page 📚")
    except Exception:
        log.exception("handle_update failed")

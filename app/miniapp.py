"""Mini App REST API. Every request carries Telegram initData; user must be a tutor."""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from . import availability as av
from . import config, db, security

router = APIRouter(prefix="/api")


def _tutor_from_initdata(init_data: str | None) -> dict:
    if not init_data:
        raise HTTPException(401, "missing initData")
    user = security.validate_init_data(init_data)
    if not user:
        raise HTTPException(401, "invalid initData")
    tutor = db.get_tutor_by_tg(user["id"])
    if not tutor:
        raise HTTPException(403, "not a tutor")
    return tutor


@router.get("/me")
def me(x_init_data: str | None = Header(default=None, alias="X-Init-Data")):
    t = _tutor_from_initdata(x_init_data)
    return {
        "name": t["name"],
        "timezone": t["timezone"],
        "price_stars": t["price_stars"],
        "booking_link": config.booking_deeplink(t["link_code"]),
    }


@router.get("/bookings")
def bookings(x_init_data: str | None = Header(default=None, alias="X-Init-Data")):
    t = _tutor_from_initdata(x_init_data)
    out = []
    for b in db.upcoming_bookings(t["id"]):
        out.append({
            "id": b["id"],
            "student": b["student_name"],
            "starts_at": b["starts_at_utc"],
            "label": av.fmt_slot(b, t["timezone"]),
        })
    return {"bookings": out}


@router.get("/earnings")
def earnings(x_init_data: str | None = Header(default=None, alias="X-Init-Data")):
    t = _tutor_from_initdata(x_init_data)
    total = db.earnings_stars(t["id"])
    return {"total_stars": total, "approx_usd": round(total * 0.015, 2)}


@router.get("/slots")
def slots(x_init_data: str | None = Header(default=None, alias="X-Init-Data")):
    t = _tutor_from_initdata(x_init_data)
    out = []
    for s in db.open_slots(t["id"], limit=60):
        out.append({"id": s["id"], "label": av.fmt_slot(s, t["timezone"]),
                    "starts_at": s["starts_at_utc"]})
    return {"slots": out, "count": len(out)}


class AvailabilityIn(BaseModel):
    days: list = []
    start: str = "18:00"
    end: str = "21:00"
    duration: int = 60


@router.post("/availability")
def save_availability(body: AvailabilityIn,
                      x_init_data: str | None = Header(default=None, alias="X-Init-Data")):
    t = _tutor_from_initdata(x_init_data)
    days = [d for d in body.days if isinstance(d, int) and 0 <= d <= 6]
    if not days:
        raise HTTPException(400, "pick at least one day")
    try:
        sh, sm = map(int, body.start.split(":"))
        eh, em = map(int, body.end.split(":"))
    except Exception:
        raise HTTPException(400, "bad time format")
    if body.duration not in (15, 30, 45, 60, 90, 120):
        raise HTTPException(400, "bad duration")
    rules = [{"weekday": d, "start_min": sh * 60 + sm,
              "end_min": eh * 60 + em, "slot_min": body.duration} for d in days]
    db.set_availability_rules(t["id"], rules)
    av.generate_slots_for_tutor(t["id"])
    return {"ok": True}


@router.post("/bookings/{booking_id}/cancel")
async def cancel_booking(booking_id: int,
                         x_init_data: str | None = Header(default=None, alias="X-Init-Data")):
    from . import telegram as tg
    t = _tutor_from_initdata(x_init_data)
    b = db.get_booking(booking_id)
    if not b or b["tutor_id"] != t["id"]:
        raise HTTPException(404, "booking not found")
    if b["status"] != "paid":
        raise HTTPException(400, "only paid bookings can be cancelled")
    cancelled = db.cancel_booking(booking_id)
    if not cancelled:
        raise HTTPException(400, "already cancelled")
    payment = db.get_payment_by_booking(booking_id)
    if payment and payment["status"] == "completed":
        r = await tg.refund_stars(b["student_tg_id"], payment["telegram_charge_id"])
        if r.get("ok"):
            db.mark_payment_refunded(booking_id)
    return {"ok": True, "refunded": True}

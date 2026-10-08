"""Mini App REST API. Every request carries Telegram initData; user must be a tutor."""
from fastapi import APIRouter, Header, HTTPException

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

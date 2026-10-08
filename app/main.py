"""SlotBot server: Telegram webhook + Mini App API + static dashboard + reminders.

Run locally (polling, no public URL needed):
    .venv/bin/python -m app.main --polling
Run production (webhook):
    .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import argparse
import asyncio
import logging
import sys

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import bot as bot_handlers
from . import config, db, reminders, security, telegram as tg
from .miniapp import router as miniapp_router

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
# Never leak the bot token: httpx logs full request URLs (token is in the path)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("slotbot")

app = FastAPI(title="SlotBot")
app.include_router(miniapp_router)
app.mount("/miniapp", StaticFiles(directory="miniapp", html=True), name="miniapp")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/webhook")
async def webhook(request: Request):
    if not security.verify_webhook_secret(request.headers):
        return Response(status_code=403)
    update = await request.json()
    await bot_handlers.handle_update(update)
    return {"ok": True}


@app.on_event("startup")
async def on_startup():
    problems = config.validate_startup()
    if problems:
        for p in problems:
            log.error("CONFIG: %s", p)
    db.get_conn()  # init schema
    reminders.start()
    if not config.POLLING and config.BASE_URL and config.WEBHOOK_SECRET:
        r = await tg.set_webhook()
        log.info("setWebhook: %s", r.get("ok"))


@app.on_event("shutdown")
async def on_shutdown():
    reminders.stop()


# ---------------------------------------------------------- polling mode ---
async def polling_loop():
    log.info("polling mode: fetching updates")
    await tg.delete_webhook()
    offset = None
    while True:
        try:
            data = await tg.get_updates(offset=offset, timeout=25)
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                await bot_handlers.handle_update(upd)
        except asyncio.CancelledError:
            break
        except Exception:
            log.exception("polling error")
            await asyncio.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polling", action="store_true",
                    help="run Telegram in polling mode (local dev)")
    args = ap.parse_args()
    if args.polling:
        config.POLLING = True
        problems = config.validate_startup()
        if problems:
            for p in problems:
                print("CONFIG:", p)
            sys.exit(1)
        db.get_conn()
        reminders.start()
        try:
            asyncio.run(polling_loop())
        except KeyboardInterrupt:
            pass
        finally:
            reminders.stop()
    else:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

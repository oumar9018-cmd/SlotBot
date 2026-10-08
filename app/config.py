"""Configuration from environment. No secrets in code."""
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
BOT_USERNAME = os.environ.get("BOT_USERNAME", "slotbot")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
DB_PATH = os.environ.get("DB_PATH", "./data/slotbot.db")
POLLING = os.environ.get("POLLING", "false").lower() == "true"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Booking hold: student has this long to complete Stars payment
HOLD_MINUTES = 10
# How far ahead slots are generated for a tutor
SLOT_HORIZON_DAYS = 14
# Reminders before session start
REMINDER_OFFSETS_MIN = (24 * 60, 60)

TELEGRAM_API = "https://api.telegram.org"


def validate_startup() -> list[str]:
    """Return list of config problems (empty = ok)."""
    problems = []
    if not BOT_TOKEN or ":" not in BOT_TOKEN:
        problems.append("BOT_TOKEN missing or invalid. Create a bot via @BotFather.")
    if not WEBHOOK_SECRET and not POLLING:
        problems.append("WEBHOOK_SECRET missing (required for webhook mode).")
    if not BASE_URL and not POLLING:
        problems.append("BASE_URL missing (required for webhook mode).")
    return problems


def miniapp_url() -> str:
    return f"{BASE_URL}/miniapp/"


def booking_deeplink(tutor_link_code: str) -> str:
    return f"https://t.me/{BOT_USERNAME}?start=book_{tutor_link_code}"

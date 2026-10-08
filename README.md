# SlotBot — Telegram booking + Stars-payment bot for solo tutors

V1 scope (locked): tutor onboarding → availability → shareable booking link →
student picks slot → Telegram Stars payment → confirmations → auto-reminders
→ tutor Mini App dashboard (upcoming bookings, earnings, booking link).

## 0. One-time Telegram setup (founder does this on phone)

1. Chat with [@BotFather](https://t.me/BotFather) → `/newbot` → name it (e.g. `SlotBot`)
   → copy the **bot token**.
2. BotFather → `/mybots` → your bot → **Bot Settings → Configure Mini App** →
   enable, set URL to `https://<your-domain>/miniapp/`.
3. Generate a webhook secret: `openssl rand -hex 32`.
4. In the bot: nothing else needed for Stars — digital-goods invoices work
   out of the box; test in Telegram's test environment first.

## 1. Local run (polling — no public URL needed)

```bash
cd ~/workspace/slotbot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill BOT_TOKEN (WEBHOOK_SECRET/BASE_URL not needed for polling)
.venv/bin/python -m app.main --polling
```

Talk to your bot on Telegram: `/start` → onboarding → booking link.

## 2. Tests

```bash
.venv/bin/python -u tests/test_db.py    # schema, double-booking + duplicate-payment guards
.venv/bin/python -u tests/test_e2e.py   # full core loop with Telegram mocked
```

Both must print `ALL ... TESTS PASSED`.

## 3. Deploy (Fly.io — free tier + persistent volume for SQLite)

```bash
# install flyctl: https://fly.io/docs/hands-on/install-flyctl/
fly auth login
fly launch --no-deploy          # accept defaults; app name e.g. slotbot
fly volumes create slotbot_data --region bom --size 1
fly secrets set BOT_TOKEN="<token>" WEBHOOK_SECRET="<hex>" BASE_URL="https://slotbot.fly.dev"
fly deploy
```

On startup the server calls `setWebhook` automatically and starts the
reminder scheduler. Health check: `https://slotbot.fly.dev/health`.

## 4. Project structure

```
slotbot/
├── app/
│   ├── main.py          # FastAPI: /webhook, /api/*, /miniapp static, /health
│   ├── bot.py           # all Telegram handlers (onboarding, booking, payments)
│   ├── availability.py  # slot generation (tutor-local -> UTC), time parsing
│   ├── booking.py       # (logic lives in db.py transactions; see below)
│   ├── db.py            # SQLite schema + atomic helpers (holds, payments)
│   ├── payments.py      # (Stars flow lives in bot.py; thin by design)
│   ├── reminders.py     # APScheduler: expired-hold sweeper + 24h/1h reminders
│   ├── miniapp.py       # dashboard REST API (initData-authenticated)
│   ├── security.py      # webhook secret + initData HMAC validation
│   ├── telegram.py      # thin Bot API client (httpx)
│   └── config.py        # env config
├── miniapp/index.html   # tutor dashboard (vanilla JS + Telegram WebApp SDK)
├── tests/test_db.py     # DB correctness tests
├── tests/test_e2e.py    # full core-loop test (Telegram mocked)
├── Dockerfile  fly.toml  requirements.txt  .env.example
```

## 5. Security properties (implemented)

- Bot token + webhook secret **only** from env, never in code/logs.
- Webhook verified via `X-Telegram-Bot-Api-Secret-Token`.
- Mini App `initData` validated with HMAC-SHA256 (`compare_digest`).
- Slots held atomically (`BEGIN IMMEDIATE`): two students can't take one slot.
- Payments idempotent: `telegram_payment_charge_id` UNIQUE — duplicate
  `successful_payment` updates are ignored, never double-counted.
- `pre_checkout_query` re-validates the hold server-side (expired/gone → reject).
- All slot times stored UTC; converted per tutor timezone for display.
- Tutor cancel → slot freed + `refundStarPayment` issued automatically.

## 6. Money flow (V1 pilot)

Student pays **Stars** via bot invoice → Stars land in the **bot owner's**
Telegram balance → withdraw via Fragment (TON) → settle tutors manually
(founder-led, fine for 10 pilot tutors). Tutor SaaS fee (₹249/$5 pilot)
collected manually by founder (UPI/bank) — no in-bot subscription billing
in V1 scope.

## 7. Known limitations / blockers

- **Bot token required**: founder must create the bot via @BotFather (can't be
  done from here).
- **Stars → tutor payout is manual** in V1 (documented above).
- **Tutoring via Stars**: sessions are a digital service; this fits Telegram's
  digital-goods Stars flow, but keep `/terms` + `/paysupport` (implemented).
- **Single process**: in-memory onboarding sessions reset on restart;
  acceptable for pilot, move to DB/Redis if it ever matters.
- **Free-tier hosting**: Fly keeps one machine always-on (reminders work).
  Render free would sleep (reminders delayed) and has no persistent disk —
  don't use Render free with SQLite.

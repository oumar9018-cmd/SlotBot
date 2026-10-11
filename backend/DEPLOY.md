# Strict English Voice Test — Backend Deploy Guide

## Architecture recap
```
Telegram Mini App --(wss)--> FastAPI /ws/voice/{session_id} --(wss)--> OpenAI Realtime API
                          --(https)--> Firestore (sessions, users, payments)
                          --(https)--> Telegram Bot API (Stars invoices, payment webhook)
```
The OpenAI API key lives ONLY on this server. The Mini App talks to our
WebSocket; we relay to OpenAI. Ephemeral tokens from `/session/start` are
short-lived and scoped to one Realtime session.

Payments are 100% Telegram Stars (XTR) — 1-click native purchase inside
Telegram. No Razorpay, no KYC forms, no checkout pages.

Voice choice: **OpenAI Realtime API** (`onyx` voice — deep, stern, fits the
examiner persona). Single API, lowest latency, no second vendor.

---

## 1. Firebase setup (5 min)

1. Firebase Console → use existing project (or create new).
2. **Firestore** → Create database → production mode.
3. **Service account:** Project Settings → Service accounts → Generate new private key → save JSON as `firebase-service-account.json`.
4. **Rules** (Firestore → Rules tab) — paste:
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Backend uses Admin SDK (bypasses rules). Lock everything from clients.
    match /{document=**} { allow read, write: if false; }
  }
}
```
All reads/writes go through this backend via the Admin SDK, so client rules stay deny-all.

## 2. Telegram bot + Stars setup (10 min)

1. Talk to **@BotFather** → `/newbot` (or use your existing bot) → copy the **bot token** → `TELEGRAM_BOT_TOKEN`.
2. Stars payments require no extra BotFather config — any bot can issue XTR invoices via `createInvoiceLink`.
3. **Set the payment webhook** so Telegram delivers `pre_checkout_query` and `successful_payment` updates:
```bash
curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://YOUR_DOMAIN/payment/telegram-webhook",
       "secret_token": "pick-a-random-string",
       "allowed_updates": ["message", "pre_checkout_query"]}'
```
   Put the same random string in `TELEGRAM_WEBHOOK_SECRET` (optional but recommended).
4. Price configured in code: **750 Stars** (`FULL_TEST_PRICE_STARS` in `main.py`).

### Stars payment flow (how it works)
1. Mini App calls `POST /payment/stars-invoice` → backend calls `createInvoiceLink` → returns `invoice_url`.
2. Mini App opens the invoice URL (Telegram shows the native 1-click Stars pay sheet).
3. Telegram sends `pre_checkout_query` → backend answers `ok=True`.
4. On success Telegram sends `message.successful_payment` → backend verifies the payload, marks the user paid (idempotent via charge id), credits the affiliate.
5. Mini App polls `GET /affiliate/stats/...` — or simpler: after payment, user taps "I've paid" → frontend calls `/session/start` with `plan=paid`, which now succeeds.

## 3. OpenAI

1. [platform.openai.com](https://platform.openai.com) → API key with **Realtime API** access.
2. Fund the account — Realtime audio is ~$0.06/min input+output blended; a 15-min session ≈ **$0.90**. Price sessions accordingly (750 Stars ≈ $9.75 face value → healthy margin even after platform fees).

## 4. Deploy the server

### Option A — Railway (fastest)
1. Push `backend/` to a GitHub repo.
2. Railway → New Project → Deploy from GitHub.
3. Add env vars (from `.env.example`): `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` (optional).
4. Add the service-account JSON as a Railway **volume/file** at `/secrets/firebase-service-account.json`, set `FIREBASE_CRED_PATH` to that path. (Alternative: base64 the JSON into an env var and decode at boot — modify `firestore_db.py` if you prefer.)
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Copy the public URL → `https://YOUR_DOMAIN`.
7. Run the `setWebhook` command from section 2 against your public URL.

### Option B — VM (this machine pattern)
```bash
cd ~/workspace/visaenglish/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in values
uvicorn main:app --host 0.0.0.0 --port 8000
```
Put Caddy/Nginx in front for TLS (Telegram WebApps and webhooks require HTTPS + WSS).

## 5. Wire the frontend

The Mini App needs:
- `POST https://YOUR_DOMAIN/session/start` with `{"user_id","test_type":"academic|business|general","plan":"free_trial|paid"}` → `{session_id, ephemeral_token, ws_url_path, max_duration_sec}`
- Open `wss://YOUR_DOMAIN/ws/voice/{session_id}`
  - send mic as **binary PCM16 16kHz mono** chunks
  - play received **binary PCM16 24kHz mono** chunks
  - send `{"type":"interrupt"}` on barge-in button
  - status event now carries `"test_type"` (not `visa_type`)
  - on `{"type":"timeup"}` → call `POST /session/end` → show score + paywall (if free trial)
- Stars purchase: `POST /payment/stars-invoice` → open `invoice_url` → Telegram handles the rest natively.

## 6. Smoke test

```bash
curl https://YOUR_DOMAIN/health
# {"ok": true, ...}

# start a session (use a test telegram id)
curl -X POST https://YOUR_DOMAIN/session/start \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"test123","test_type":"academic","plan":"free_trial"}'
# -> session_id + ephemeral_token (2-min limit)

# create a Stars invoice (needs TELEGRAM_BOT_TOKEN)
curl -X POST https://YOUR_DOMAIN/payment/stars-invoice \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"test123"}'
# -> {"invoice_url": "https://t.me/$...", "payload": "test123:full:..."}

# after a voice session, score it
curl -X POST https://YOUR_DOMAIN/session/end \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<id>"}'
# -> {clarity, ties, financial, consistency, composure, total, verdict, weaknesses, feedback}
# verdict is one of: PASS / NEEDS_WORK / FAIL
```

## 7. Kill-gate instrumentation

The 30-day gate (500 paid sessions) is measurable from Firestore:
- `payments` where `status == "captured"` → count.
- Affiliate dashboard: `GET /affiliate/stats/{code}` (earnings now tracked in Stars).

## Cost sanity per session
| Item | 2-min trial | 15-min paid |
|---|---|---|
| Realtime audio (~$0.06/min) | ~$0.12 | ~$0.90 |
| Whisper transcription | ~$0.01 | ~$0.05 |
| GPT-4o scoring | ~$0.02 | ~$0.03 |
| **Total** | **~$0.15** | **~$1.00** |
| Revenue | 0 | 750 Stars (~$9.75 face) |

Free-trial abuse guard: one trial per `user_id` (Telegram id) enforced in `/session/start` + `free_trial_used` flag set on `/session/end`.

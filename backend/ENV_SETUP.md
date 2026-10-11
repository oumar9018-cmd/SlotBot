# VoiceGrill Backend — Environment Variables

All env vars are read at import/startup. Set them in Render →
`visaenglish-backend` → **Environment**. Redeploy after changing them.

Missing vars are logged at startup (see `check_required_env` in `main.py`).

---

## Required

### `TELEGRAM_BOT_TOKEN`
- **What:** The bot token for `@Cripercore_bot`, issued by @BotFather.
- **Where to get it:** Telegram → @BotFather → `/mybots` → select the bot → API Token.
  If lost, `/revoke` the old token and paste the new one.
- **What breaks without it:** everything payment-related — `POST /payment/stars-invoice`
  returns **500 "Payment not configured: TELEGRAM_BOT_TOKEN missing on server"**,
  the `/start` welcome message can't be sent, the webhook can't answer
  pre-checkout queries, and `/api/auth/telegram` HMAC verification fails.

### `GROQ_API_KEY`
- **What:** API key for Groq (LLM inference). Used by `groq_reply()` in
  `realtime_proxy.py` and `scoring.py`.
- **Where to get it:** https://console.groq.com → API Keys → Create API Key.
- **Models used** (see `GROQ_MODEL*` overrides below — both must exist on this key):
  - `openai/gpt-oss-20b` (primary)
  - `qwen/qwen3.8-27b` (fallback)
  Verify with `GET /test-keys` — it lists `Groq_Available_Models` for the key.
- **What breaks without it:** every examiner reply fails — the WebSocket sends
  "Examiner lost train of thought" and session scoring fails.

### `DEEPGRAM_API_KEY`
- **What:** API key for Deepgram — streaming STT (nova-3) and TTS
  (aura-asteria-en strict / aura-luna-en friendly).
- **Where to get it:** https://console.deepgram.com → API Keys → Create a new key.
- **What breaks without it:** no speech recognition and no voice output —
  `voice_proxy` closes the socket with "Voice engine not configured".

## Model overrides (optional, have sane defaults)

### `GROQ_MODEL` (default: `openai/gpt-oss-20b`)
Primary chat model for the examiner dialogue.

### `GROQ_MODEL_FALLBACK` (default: `qwen/qwen3.8-27b`)
Used automatically if the primary model 404s/errors. `groq_reply` tries each in order.

### `GROQ_SCORING_MODEL` (default: `openai/gpt-oss-20b`)
Model used by `scoring.py` for the 0–100 verdict. Can be set to a cheaper model.

## Optional

### `FIREBASE_CRED_PATH`
- **What:** Path to the Firebase service-account JSON for project `slotbot-9d27f`
  (`/users`, `/sessions` persistence).
- **Where to get it:** Firebase Console → Project settings → Service accounts →
  Generate new private key. Mount the file on the server and set the path.
- **What breaks without it:** nothing — `firestore_db.py` falls back to an
  in-memory store. Data is lost on restart/redeploy, so set this for production.

### `TELEGRAM_WEBHOOK_SECRET`
- **What:** Shared secret Telegram sends as `X-Telegram-Bot-Api-Secret-Token` so the
  backend can reject forged webhook calls.
- **Where to get it:** you invent it (random string). Set the same value in
  @BotFather → `/setwebhook` (or in the `setWebhook` call's `secret_token` param).
- **What breaks without it:** the secret check is skipped — webhook still works,
  but anyone who knows the URL could POST fake payment updates.

---

## Quick Render checklist

| Var | Set? | Test |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ required | `/api/auth/telegram` from the Mini App; Stars invoice creates |
| `GROQ_API_KEY` | ✅ required | `GET /test-keys` → `Groq_Status: 200 OK` |
| `DEEPGRAM_API_KEY` | ✅ required | `GET /test-keys` → `Deepgram_Status: 200 OK` |
| `GROQ_MODEL*` | defaults OK | — |
| `FIREBASE_CRED_PATH` | optional | sessions persist across restarts |
| `TELEGRAM_WEBHOOK_SECRET` | optional | webhook rejects bad-secret POSTs |

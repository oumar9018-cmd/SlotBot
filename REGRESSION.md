# SlotBot — Fly.io Deploy + Final Live Regression Runbook

Banaya: 2026-10-07 (~21:20 IST). User order (15:12Z chat): "Fly.io deploy karo,
Mini App ko live Telegram WebView mein test karo, deployment ke baad final live
regression report do." Deploy abhi **blocked hai — fresh Fly.io token ka intezaar**
(last user question 15:26Z: token type kya rakhe; assistant ne 15:27Z jawab diya).
Yeh runbook deploy ke turant baad execute karna hai taaki regression report
turant ban sake. Pre-deploy verification researcher ne kar li hai (neeche §0).

Test bot: @Cripercore_bot (polling mode, local). Deploy ke baad webhook mode.

---

## §0 Pre-deploy verification (researcher ne check kiya — OK)

- `fly.toml`: app `slotbot`, region `bom` (Mumbai), `internal_port=8000` =
  Dockerfile CMD port, volume mount `slotbot_data → /data`, `DB_PATH=/data/slotbot.db`
  dono jagah match, `auto_stop_machines=false` + `min_machines_running=1`
  (reminders ke liye zaroori — APScheduler in-process hai).
- `app/main.py`: non-polling mode mein startup par `set_webhook()` hota hai
  (secret_token + allowed_updates me `pre_checkout_query` shamil). Polling loop
  alag (`--polling` flag) — local test bot se conflict nahi hoga jab tak webhook
  set hai. ⚠️ Deploy se pehle local polling bot **rok dena** (same token par
  polling + webhook ladenge).
- `app/bot.py` `_dashboard_btn()`: webapp button sirf tab judta hai jab BASE_URL
  real `https://` ho — local test mein message-reject wala bug dobara nahi hoga.
- ⚠️ **Ek asli catch:** `fly.toml` mein `[[mounts]]` declared hai lekin `fly deploy`
  volume **auto-create nahi karta**. Pehle yeh chahiye:
  `fly volumes create slotbot_data --region bom --size 1`
  (bina iske deploy "volume not found" par fail hoga).
- ⚠️ **Secret order (chicken-and-egg):** `BASE_URL` deploy se pehle pata nahi hota.
  Order: app create → volume create → secrets (BOT_TOKEN, WEBHOOK_SECRET,
  BOT_USERNAME=Cripercore_bot) → `fly deploy` → URL mile
  (https://slotbot.fly.dev) → `fly secrets set BASE_URL=<url>` → dobara deploy
  ya machine restart (taaki startup hook `setWebhook` chalaye).
- ⚠️ App naam `slotbot` Fly.io par taken ho sakta hai → `fly apps create
  slotbot-<suffix>` aur `fly.toml` mein app naam update karna.
- ⚠️ Fly.io signup par credit card maang sakta hai (unverified heads-up —
  is chhote app ka bill ~$0 rahega).

## §0.5 Token pre-check (added 2026-10-08 ~13:20 IST — kal ka pasted token 401 de
gaya tha, comma-joined/double paste ki wajah se; dobara mid-deploy fail na ho)

Token validate karne ka sabse tez tarika — GraphQL `viewer` query (flyctl install
ki zaroorat nahi, Fly.io ka documented auth path):

```bash
T='<pasted token, single line — koi comma, space, ya newline nahi>'
curl -s -X POST https://api.fly.io/graphql \
  -H "Authorization: Bearer $T" \
  -H 'Content-Type: application/json' \
  -d '{"query":"query { viewer { email } }"}' | head -c 300; echo
# ✅ Valid:  {"data":{"viewer":{"email":"<tumhara email>"}}}
# ❌ Invalid: {"errors":[...]} ya 401 — to token galat paste hua hai, naya banao
```

Fresh token banane ka sahi path: Fly.io Dashboard → Account → Settings →
Access Tokens → Create Access Token. Token **ek hi line** mein copy-paste karo;
do token comma se chipke hon ya line-break ho to 401 aayega (kal yahi hua tha).
flyctl installed ho to equivalent check: `FLY_API_TOKEN='<token>' fly auth whoami`.

## §1 Deploy commands (token milte hi)

```bash
export FLY_API_TOKEN='<fresh-token>'   # ek hi token, bina comma ke
cd ~/workspace/slotbot
fly volumes create slotbot_data --region bom --size 1
fly secrets set BOT_TOKEN='<bot-token>' \
  WEBHOOK_SECRET="$(openssl rand -hex 32)" \
  BOT_USERNAME=Cripercore_bot
fly deploy
# URL note karo, phir:
fly secrets set BASE_URL='https://slotbot.fly.dev'
fly deploy   # ya: fly machine restart  (setWebhook startup par chalega)
```

## §2 Post-deploy verification (har step ka expected result)

1. `curl https://slotbot.fly.dev/health` → `{"ok":true}`
2. `curl https://api.telegram.org/bot<TOKEN>/getWebhookInfo` →
   `url` = `https://slotbot.fly.dev/webhook`, `pending_update_count` = 0
3. `curl -sI https://slotbot.fly.dev/miniapp/` → `200`
4. `curl -s https://slotbot.fly.dev/api/me` → `401` (bina initData — security OK)
5. Bot mein `/start` → tutor home keyboard mein **📊 Dashboard** webapp button
   dikhe (yeh button BASE_URL set hone ke baad hi aata hai — iska dikhna
   prove karta hai BASE_URL sahi set hai).
6. **Mini App live WebView test:** Dashboard button dabao → Telegram ke andar
   khule, bookings/earnings/slots dikhein (tutor isolation: sirf apna data).
7. **Chat-level Open button** (user ne manga: "bot ke saath attach karo, open
   button ya link dikhna chahiye"):
   ```bash
   curl -s -X POST https://api.telegram.org/bot<TOKEN>/setChatMenuButton \
     -H 'Content-Type: application/json' \
     -d '{"menu_button":{"type":"web_app","text":"Dashboard",
         "web_app":{"url":"https://slotbot.fly.dev/miniapp/"}}}'
   ```
   Phir chat ke neeche menu button → "Dashboard" → Mini App khulna chahiye.
8. **E2E payment regression (live):** naya slot → booking link → student hold →
   Stars invoice → pre-checkout (fixed bug — `query["id"]`) → pay → dono taraf
   confirmation. Pehle wali live payment (6 Stars, booking #5 refund test)
   isi code path se guzar chuki hai; deploy ke baad ek fresh booking se
   re-verify karo.
9. **Reminder:** booking ka slot 1h door rakho → reminder aaye (code path
   24h wala same hai; pehle 14:48 UTC par live fire ho chuka hai).
10. **Persistence:** `fly machine restart` → data intact (SQLite volume par hai).
11. **Logs:** `fly logs` mein koi token leak nahi, koi exception nahi.

## §3 Final live regression report (user ko dena hai)

Report mein yeh table bharo — har row PASS/FAIL + evidence (timestamp/message):

| Check | Status | Evidence |
|---|---|---|
| /health | | |
| getWebhookInfo | | |
| Mini App static 200 | | |
| /api/me 401 (security) | | |
| /start → 📊 Dashboard button | | |
| Mini App WebView (Telegram ke andar) | | |
| Chat menu Open button | | |
| Fresh E2E Stars payment | | |
| Reminder fired | | |
| Restart → data intact | | |
| Logs clean (no leak) | | |

Saare PASS → V1 production-ready declare karo. Koi FAIL → fix + re-test, report mein note.

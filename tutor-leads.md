# SlotBot — Pilot Prep: Tutor Sourcing Shortlist + Settlement Formula

**Prepared: 2026-10-08 ~11:20 IST.** Use after Fly.io deploy. Do not outreach before then
(hard rule #1 in outreach-pack.md: core loop must work end-to-end on a live bot first).

---

## A. Where the 30 conversations come from (priority order)

1. **Warm network first** (outreach-pack §5: warm intros convert ~10× better). User action:
   list 10 tutor/teacher contacts from personal network before touching any group.
2. **Telegram — primary channel** (tutors here are already Telegram-native, which is the
   whole product thesis):
   - **@onlinetutorjobs** — **3.5K members VERIFIED 2026-10-08**, "Online teaching jobs -
     part time teaching jobs - all over india", online tuition jobs class 1–12. **Best fit**
     (India tutors actively looking for work): https://telegram.im/@onlinetutorjobs
   - **@tuitionwork** — **766 subscribers VERIFIED 2026-10-08** (was 1,595 in earlier note —
     corrected down), "Home Tutor Jobs & Online Tutor Jobs": https://t.me/tuitionwork.
     **Heads-up: "subscribers" wording means this is a CHANNEL (broadcast), not a discussion
     group** — you can't post in it. Use as a lead source / DM channel admins only, not a
     posting venue. Secondary fit.
   - **Teachers Support Network** — teacher community group VERIFIED 2026-10-08 (invite link
     live, description matches: home tuition + online teaching opportunities, peer support):
     https://t.me/+dsOcEBcSzpdkODk1 — good fit. Member count not visible without login;
     confirm in-app after joining.
   - Directory for more: https://groupda.com/education/tuition-telegram-group-link/
     (mostly student-facing tuition groups — lower priority, verify before joining).
3. **Reddit / Facebook — secondary only.** Verify any specific group exists and allows
   promo posts before using. Not the priority; Telegram-first.

**Method (from outreach-pack §5, unchanged):** join 8–12 groups → observe/contribute 2 days →
post once per group (pack §2) → DM only people who reply/react (pack §3–4). No spam,
never post twice in the same group.

---

## B. Stars → INR settlement formula (numbers locked 2026-10-08)

Settlement-audit.md required this **before real tutor money flows**. Today's research:

- **Developer payout: flat $0.013 / Star.** Taken from Telegram's own payments table
  (e.g. 1,000 Stars bought on mobile = $19.99 → $6.00 store cut + $0.99 Telegram admin +
  $13.00 developer). The app-store 30% inflates the *buyer's* price; it never touches the
  developer's $0.013/Star. Source: https://dev.to/landstrider/telegram-stars-payouts-the-app-store-cut-hits-the-buyer-not-you-2bk3 (Oct 2026, cites Telegram's payments table).
- **Withdrawal path:** Fragment → TON → exchange/P2P → INR. **Minimum 1,000 Stars (~$13)** —
  below that nothing moves. **21-day rolling hold** from each payment before it becomes
  withdrawable. Payout triggered by the bot owner.
- **USD/INR today:** ≈ 96.77 (Finnhub, 2026-10-08).
- **Proposed pilot formula:**
  `settlement INR = Stars_earned × 0.013 × 96.77 × 0.96 (≈4% conversion buffer)`
  ≈ **₹1.20 per Star**, fixed for the pilot. Review monthly against the live Fragment rate.
- **Cash-flow catch (new):** settlement-tracker.md promises "every Monday, settle last
  week's paid bookings — no settlement older than 7 days." The **21-day hold makes that
  impossible from withdrawn Stars** — the founder must front settlements from pocket for
  the first ~3 weeks. At pilot scale (≤10 tutors, a few classes/week) that's a few thousand
  rupees of float — feasible, but go in knowing it. Also: sub-1,000-Star balances can't be
  withdrawn at all, so batch Fragment withdrawals monthly and log the exact rate used each
  week in settlement-tracker.md (template already has the column).

Example: tutor earns 500 Stars in a week → settle ₹600 that Monday from founder's pocket;
the Stars become withdrawable ~3 weeks later and accumulate toward the 1,000-Star minimum.

---

## C. Week-1 execution plan (post-deploy)

- **Day 1:** paste fresh Fly.io token → deploy per REGRESSION.md §1 (~30 min). Set chat menu
  button (§2.7), verify /health + webhook + Dashboard button (§2.1–5).
- **Day 1–2:** join the 3 Telegram groups above, observe; write the 10-name warm list.
- **Day 3–5:** group post (pack §2) → DM responders (pack §3–4) → **collect ₹249 BEFORE
  any setup call** (pack hard rule #2 — no pay, no setup).
- **Ongoing:** log every genuine 1:1 conversation in settlement-tracker.md; 30 conversations
  → count paid pilots → ≥2 = CONTINUE, <2 = KILL. Compliments don't count.

---

## D. State check (2026-10-08 ~11:20 IST)

- Local test bot (@Cripercore_bot, polling) is **not running** — no slotbot/uvicorn process
  found. Root `slotbot.db` is 0 bytes (fresh); `data/slotbot.db` holds the Oct-7 test data
  (WAL present). Nothing live to poke until restart or deploy.
- Deploy still blocked on the **fresh Fly.io token** (user action — morning nudge already queued).
- Pre-live founder checklist (from settlement-audit.md, still open): enable **2-step
  verification** on the Telegram account that owns the bot; confirm the live Fragment
  Stars→TON rate once before first real tutor payout.

## E. Tutor sourcing — field revision (2026-10-08 ~15:25 IST)

The §A / §C "join 8–12 groups → observe 2 days → post once" plan does NOT survive
contact with reality. Live checks today: India's tutor Telegram scene is dominated by
job-BROADCAST channels, not discussion communities. Only **2 postable groups** exist;
everything else is channels (lead source / admin-DM only). Revised plan:

1. **Warm 10-name personal list FIRST** (outreach-pack §5) — converts ~10× better
   than any group.
2. Join the 2 verified postable groups → observe/contribute 2 days → post once each
   (pack §2) → DM only responders:
   - `@onlinetutorjobs` — 3.5K members, "Online tuition jobs class 1 to 12th", supergroup
   - Teachers Support Network — invite live, home-tuition + online-teaching group
   (After joining, check pinned rules for promo-post policy before posting.)
3. Telegram in-app search with the pack's term list (English + Hinglish + city names)
   — the scalable leg. DM tutors in reply-context, no spam.
4. Channel-admin DMs: ask `@tuitionwork` and TUITION CORNER admins for a pinned pilot
   post or tutor referrals. Both are broadcast channels (TUITION CORNER: Kolkata-focused,
   multiple posts/day, 500–2,200 views/post = real audience) — can't post in them, but
   their admins reach 1–3K tutors.
5. Reddit secondary: r/tutors, r/OnlineTeaching — verify promo rules before posting.
   (@tutorexjobs = Singapore, wrong fit; tdirectory teacher-ed group = stale Feb-2026.)

## F. Fragment rate cross-check (2026-10-08 ~15:25 IST, live)

Live https://fragment.com/stars/buy today: 50 Stars = $0.75, 1,000 Stars = $15.00,
100,000 Stars = $1,500 → retail $0.015/Star. The §B formula's $0.013 wholesale
assumption sits correctly just below retail — **₹1.20/Star holds, no revision**.
First real settlement: log the actual rate in settlement-tracker.md's rate column;
the 4% buffer exists for exactly this.

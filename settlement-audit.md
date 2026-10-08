# Settlement Model Audit — Stars → Manual UPI (Single-Bot Architecture)

**Model:** Student pays Stars → Stars land in SlotBot's Telegram balance →
founder withdraws via Fragment (TON → fiat) → settles tutors weekly via UPI/bank
with per-booking ledger. Tutor SaaS fee (₹249/₹499) collected separately by founder.

**Audited against:** Telegram Bot Payments docs (payments-stars), Bot Platform
Developer Terms of Service §6.2 (Digital Goods and Services), App Store/Play Store
digital-goods rules referenced therein. Read 2026-10-07.

## Verdict: ACCEPTABLE for pilot, with 3 conditions

### 1. Selling tutoring bookings via Stars — ALLOWED ✅
Stars are for "digital goods and services." A tutoring session is a service
delivered digitally (video call). This fits. Telegram imposes no product-category
ban; the constraint is using Stars (not external payments) for in-Telegram
digital sales — which we do.

### 2. Manual UPI settlement to tutors — NOT PROHIBITED ✅
Telegram's payment terms govern the Stars transaction (student → bot). What the
bot owner does with withdrawn Stars (Fragment → TON → fiat → UPI to tutors) is
outside Telegram's payment terms — it is a normal business payout to a supplier.
No Telegram rule prohibits it. The ledger (per-tutor earnings, per-payment
charge IDs) makes every rupee traceable.

### 3. Compliance items — IMPLEMENTED ✅
- `/terms` command: implemented (required before live).
- `/paysupport` command: implemented (required; bot owner handles disputes).
- Bot owner dispute responsibility: founder-led in pilot (per ToS §6.2.1).
- 2-step verification on the bot owner's Telegram account: DO BEFORE LIVE.

## Conditions / risks (not blockers for pilot)

1. **Star economics spread.** Users buy Stars at retail (Apple/Google take a cut);
   bot owners withdraw via Fragment at Telegram's wholesale rate. 200 Stars paid
   by a student ≠ 200 Stars of withdrawable value. **Founder must check the live
   Fragment rate and set the settlement formula BEFORE real tutor money flows.**
   Document the rate used each week in settlement-tracker.md.
2. **Dispute liability sits with the founder.** If a student disputes a charge,
   Telegram points them to the bot (/paysupport) — the founder must handle it.
   Keep the refund path (tutor cancel → auto-refund) working; it is the main
   dispute fire-escape.
3. **Pilot trust model.** Tutors are trusting the founder with their class fees
   until weekly settlement. This works for 10 founder-led pilots; it does not
   scale without automated payouts. Revisit before 50+ tutors.
4. **Policy gray zone (low risk).** If Telegram ever reclassifies live tutoring
   as a non-digital service, Stars could be questioned. Mitigation: sessions ARE
   delivered digitally; keep /terms accurate; no action needed now.

## What was verified in code
- Every payment stores `telegram_payment_charge_id` (refund/dispute traceability).
- Per-tutor earnings queryable; per-payment ledger complete.
- Tutor-cancel triggers `refundStarPayment` automatically (tested live below).
- No tutor can see another tutor's earnings (API scoped by validated initData).

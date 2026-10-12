/**
 * Firestore access via REST API (no Firebase JS SDK — saves ~570KB bundle).
 *
 * Project: slotbot-9d27f (user's existing project)
 * API key is public by design for Firebase web apps — loaded from env.
 */
import { getTelegramUser, getInitDataRaw } from './telegram-auth.js'
import { authWithBackend, api } from './api.js'

const PROJECT_ID = 'slotbot-9d27f'
const API_KEY = import.meta.env.VITE_FIREBASE_API_KEY || ''
const BASE = `https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents`

// VITE_MOCK=true → demo mode allowed outside Telegram (for UI testing only)
const MOCK = import.meta.env.VITE_MOCK === 'true'

let _user = null
let _demoMode = false

function hasApiKey() {
  if (!API_KEY) {
    console.error('[firebase] VITE_FIREBASE_API_KEY missing — running without Firestore')
    return false
  }
  return true
}

/** Marker for server timestamp — replaced with client ISO time on write. */
function serverTimestamp() {
  return { _serverTimestamp: true }
}

/**
 * fetch() with a 15s timeout. A hung Firestore must never hang the UI —
 * every caller already catches errors and degrades honestly, so a timeout
 * surfaces as a visible retry state instead of an eternal spinner.
 */
async function fetchTimeout(url, options = {}, ms = 15000) {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), ms)
  try {
    return await fetch(url, { ...options, signal: ctrl.signal })
  } catch (e) {
    if (e && e.name === 'AbortError') throw new Error('Firestore request timed out')
    throw e
  } finally {
    clearTimeout(t)
  }
}

// ---------- Firestore REST value conversion ----------

function toFirestoreValue(v) {
  if (v === null || v === undefined) return { nullValue: null }
  if (v && v._serverTimestamp) return { timestampValue: new Date().toISOString() }
  if (typeof v === 'string') return { stringValue: v }
  if (typeof v === 'boolean') return { booleanValue: v }
  if (typeof v === 'number') {
    return Number.isInteger(v)
      ? { integerValue: String(v) }
      : { doubleValue: v }
  }
  if (Array.isArray(v)) {
    return { arrayValue: { values: v.map(toFirestoreValue) } }
  }
  if (typeof v === 'object') {
    return { mapValue: { fields: toFirestoreFields(v) } }
  }
  return { stringValue: String(v) }
}

function toFirestoreFields(obj) {
  const fields = {}
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined) continue
    fields[k] = toFirestoreValue(v)
  }
  return fields
}

function fromFirestoreValue(v) {
  if (!v || typeof v !== 'object') return null
  if (v.stringValue !== undefined) return v.stringValue
  if (v.integerValue !== undefined) return parseInt(v.integerValue, 10)
  if (v.doubleValue !== undefined) return parseFloat(v.doubleValue)
  if (v.booleanValue !== undefined) return v.booleanValue
  if (v.timestampValue !== undefined) return v.timestampValue
  if (v.nullValue !== undefined) return null
  if (v.arrayValue) return (v.arrayValue.values || []).map(fromFirestoreValue)
  if (v.mapValue) return fromFirestoreFields(v.mapValue.fields)
  return null
}

function fromFirestoreFields(fields = {}) {
  const obj = {}
  for (const [k, v] of Object.entries(fields)) {
    obj[k] = fromFirestoreValue(v)
  }
  return obj
}

// ---------- Firestore REST document operations ----------

const docPath = (telegram_id) => `${BASE}/users/${encodeURIComponent(String(telegram_id))}`

/** GET users/{id} → parsed object, or null on 404 / error. */
async function getUserDoc(telegram_id) {
  if (!hasApiKey()) return null
  try {
    const res = await fetchTimeout(`${docPath(telegram_id)}?key=${API_KEY}`)
    if (res.status === 404) return null
    if (!res.ok) {
      console.error('[firebase] getUserDoc failed:', res.status)
      return null
    }
    const data = await res.json()
    return fromFirestoreFields(data.fields)
  } catch (e) {
    console.error('[firebase] getUserDoc error:', e.message)
    return null
  }
}

/** POST users?documentId={id} → create document. Returns true on success. */
async function setUserDoc(telegram_id, data) {
  if (!hasApiKey()) return false
  try {
    const res = await fetchTimeout(
      `${BASE}/users?documentId=${encodeURIComponent(String(telegram_id))}&key=${API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fields: toFirestoreFields(data) }),
      }
    )
    if (!res.ok) {
      // Document may already exist (409) — fall back to update
      if (res.status === 409) return updateUserDoc(telegram_id, data)
      console.error('[firebase] setUserDoc failed:', res.status)
      return false
    }
    return true
  } catch (e) {
    console.error('[firebase] setUserDoc error:', e.message)
    return false
  }
}

/** PATCH users/{id} → update fields. Returns true on success. */
async function updateUserDoc(telegram_id, data) {
  if (!hasApiKey()) return false
  try {
    const fields = toFirestoreFields(data)
    const mask = Object.keys(fields)
      .map((k) => `updateMask.fieldPaths=${encodeURIComponent(k)}`)
      .join('&')
    const res = await fetchTimeout(
      `${docPath(telegram_id)}?key=${API_KEY}&${mask}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fields }),
      }
    )
    if (!res.ok) {
      console.error('[firebase] updateUserDoc failed:', res.status)
      return false
    }
    return true
  } catch (e) {
    console.error('[firebase] updateUserDoc error:', e.message)
    return false
  }
}

// ---------- Public API (unchanged) ----------

/**
 * Auto-login via Telegram identity. No login screen, no manual auth.
 *
 * 4-step flow:
 *  1. Instant UI: extract user from initDataUnsafe / launch params (client-side)
 *  2. Backend verify: send RAW initData to FastAPI → HMAC-SHA256 signature check
 *  3. Firestore check: backend checks/creates the user doc
 *  4. Verified user: returned with verified=true (or Firestore fallback if
 *     the backend is not configured / unreachable)
 *
 * @param {object|null} tgUserHint - pre-extracted Telegram user (from retry
 *   logic in useTelegram). If provided, skips re-extraction.
 * @param {string|null} initDataRawHint - pre-fetched raw initData string.
 * Returns the user object, or NULL when outside Telegram in production
 * (caller shows the "Open in Telegram" screen). In MOCK mode it falls
 * back to a demo user for browser testing.
 */
export async function ensureUser(tgUserHint = null, initDataRawHint = null) {
  if (_user) return _user

  // Step 1: Get raw initData + parsed user (instant UI)
  const initDataRaw = initDataRawHint || getInitDataRaw()
  const tgUser = tgUserHint || getTelegramUser()

  // Step 2: Try backend verification (secure)
  if (initDataRaw) {
    const verified = await authWithBackend(initDataRaw)
    if (verified) {
      _user = {
        telegram_id: verified.telegram_id,
        first_name: verified.first_name || '',
        last_name: verified.last_name || '',
        username: verified.username || '',
        language_code: tgUser?.language_code || '',
        photo_url: tgUser?.photo_url || '',
        free_trial_used: !!verified.free_trial_used,
        paid: !!verified.paid,
        is_new: !!verified.is_new,
        verified: true, // backend-verified!
        auth_source: 'backend',
        created_at: Date.now(),
        // Credit/trial fields: the verified payload may not carry them.
        // Default them here so the 3-day trial actually works for
        // backend-verified users (refreshCredits() later overwrites with
        // the real Firestore values when the doc exists).
        interview_credits: Number(verified.interview_credits || 0),
        trial_credits: Number(verified.trial_credits ?? 1),
        trial_started_at: verified.trial_started_at || Date.now(),
        free_mini_used: !!verified.free_mini_used,
        credited_charges: Array.isArray(verified.credited_charges)
          ? verified.credited_charges
          : [],
      }
      return _user
    }
    // Backend rejected/unreachable — fall through to direct Firestore
  }

  // Step 3: Fallback to direct Firestore (REST)
  if (!tgUser) {
    if (MOCK) {
      _demoMode = true
      _user = {
        telegram_id: 999999,
        first_name: 'Demo',
        username: 'demo_candidate',
        free_trial_used: false,
        paid: false,
        demo: true,
        auth_source: 'demo',
        created_at: Date.now(),
      }
      return _user
    }
    // Production outside Telegram: no user — caller shows "Open in Telegram"
    _user = null
    return null
  }

  const telegram_id = tgUser.id
  const profile = {
    telegram_id,
    first_name: tgUser.first_name || '',
    last_name: tgUser.last_name || '',
    username: tgUser.username || '',
    language_code: tgUser.language_code || '',
    photo_url: tgUser.photo_url || '',
    auth_source: tgUser.source || 'unknown',
    updated_at: serverTimestamp(),
  }

  if (!hasApiKey()) {
    // No Firestore configured — local-only user
    _user = { ...profile, free_trial_used: false, paid: false, created_at: Date.now() }
    return _user
  }

  try {
    const existing = await getUserDoc(telegram_id)
    if (existing) {
      await updateUserDoc(telegram_id, { ...profile })
      _user = { ...existing, ...profile, telegram_id }
      // Backfill trial fields for users created before credits existed
      if (_user.trial_started_at == null) {
        await initTrialForNewUser()
      } else {
        _user.interview_credits = Number(_user.interview_credits || 0)
        _user.trial_credits = Number(_user.trial_credits ?? 1)
        _user.free_mini_used = !!_user.free_mini_used
        _user.credited_charges = _user.credited_charges || []
      }
    } else {
      const fresh = {
        ...profile,
        free_trial_used: false,
        paid: false,
        created_at: serverTimestamp(),
      }
      await setUserDoc(telegram_id, fresh)
      _user = { ...fresh, telegram_id, created_at: Date.now() }
      await initTrialForNewUser()
    }
  } catch (e) {
    console.error('[firebase] ensureUser failed, local-only:', e.message)
    _user = { ...profile, free_trial_used: false, paid: false, created_at: Date.now() }
  }
  return _user
}

export function getUser() {
  return _user
}

export function isDemoMode() {
  return _demoMode
}

export async function markTrialUsed() {
  if (_user) _user.free_trial_used = true
  if (!_user || _user.demo) return
  try {
    await updateUserDoc(String(_user.telegram_id), {
      free_trial_used: true,
      updated_at: serverTimestamp(),
    })
  } catch (e) {
    console.error('[firebase] markTrialUsed failed:', e.message)
  }
}

export async function markPaid() {
  if (_user) _user.paid = true
  if (!_user || _user.demo) return
  try {
    await updateUserDoc(String(_user.telegram_id), {
      paid: true,
      paid_at: serverTimestamp(),
      updated_at: serverTimestamp(),
    })
  } catch (e) {
    console.error('[firebase] markPaid failed:', e.message)
  }
}

/** Append a finished test to the user's history (best-effort). */
export async function saveTestResult({ score, verdict }) {
  if (!_user || _user.demo) return
  try {
    const existing = await getUserDoc(String(_user.telegram_id))
    const history = existing?.history || []
    history.unshift({ score, verdict, at: Date.now() })
    await updateUserDoc(String(_user.telegram_id), {
      history: history.slice(0, 20),
      updated_at: serverTimestamp(),
    })
  } catch (e) {
    console.error('[firebase] saveTestResult failed:', e.message)
  }
}

// ---------- Interview credits (pay-per-use bundles) ----------

/**
 * Credit model (users/{id} doc):
 *  interview_credits: number  — paid bundle credits
 *  trial_credits: number      — 1 free trial interview (3-day expiry)
 *  trial_started_at: number   — ms timestamp when trial began
 *  free_mini_used: bool       — first 3-question mini interview hook
 *  credited_charges: string[] — Telegram charge ids already credited (idempotency)
 */

export function getTrialDaysLeft() {
  // Number() coercion: a legacy string timestamp (or any garbage) must
  // yield 0, never NaN — a NaN day-count would poison the credit math.
  const started = Number(_user?.trial_started_at)
  if (!Number.isFinite(started) || started <= 0) return 0
  const elapsed = Date.now() - started
  const left = 3 - Math.floor(elapsed / (24 * 60 * 60 * 1000))
  return Math.max(0, left)
}

/** Refresh credit fields from Firestore into the local user object. */
export async function refreshCredits() {
  if (!_user || _user.demo) return _user
  try {
    const doc = await getUserDoc(String(_user.telegram_id))
    if (doc) {
      _user.interview_credits = Number(doc.interview_credits || 0)
      _user.trial_credits = Number(doc.trial_credits ?? 1)
      _user.trial_started_at = doc.trial_started_at || _user.trial_started_at || Date.now()
      _user.free_mini_used = !!doc.free_mini_used
      _user.credited_charges = doc.credited_charges || []
    } else {
      // No doc (backend in-memory fallback, or deleted): PERSIST the local
      // credit state (create-or-update). Two reasons this must write, not
      // just default locally:
      //  1. A missing doc makes every later PATCH (credit purchases, trial
      //     consumption) 404 and lose data.
      //  2. trial_started_at must be anchored once — resetting it to "now"
      //     on every app open would silently re-grant the trial forever.
      const now = Date.now()
      if (!_user.trial_started_at) {
        _user.trial_started_at = now
        _user.trial_credits = 1
        _user.free_mini_used = false
      }
      _user.interview_credits = Number(_user.interview_credits || 0)
      _user.credited_charges = _user.credited_charges || []
      await setUserDoc(String(_user.telegram_id), {
        telegram_id: _user.telegram_id,
        first_name: _user.first_name || '',
        last_name: _user.last_name || '',
        username: _user.username || '',
        language_code: _user.language_code || '',
        photo_url: _user.photo_url || '',
        interview_credits: _user.interview_credits,
        trial_credits: Number(_user.trial_credits ?? 1),
        trial_started_at: _user.trial_started_at,
        free_mini_used: !!_user.free_mini_used,
        free_trial_used: !!_user.free_trial_used,
        paid: !!_user.paid,
        credited_charges: _user.credited_charges,
        created_at: serverTimestamp(),
        updated_at: serverTimestamp(),
      })
    }
  } catch (e) {
    console.error('[firebase] refreshCredits failed:', e.message)
  }
  return _user
}

/** Total usable interviews: paid credits + trial (if within 3 days). */
export function getUsableInterviews() {
  if (!_user) return 0
  const paid = Number(_user.interview_credits || 0)
  const trial = getTrialDaysLeft() > 0 ? Number(_user.trial_credits || 0) : 0
  return paid + trial
}

/**
 * Credit a bundle purchase. Idempotent per charge_id.
 * Returns the new paid-credit total.
 */
export async function creditBundlePurchase({ interviews, chargeId }) {
  if (!_user || _user.demo) return 0
  const charges = _user.credited_charges || []
  if (chargeId && charges.includes(chargeId)) return Number(_user.interview_credits || 0)
  const next = Number(_user.interview_credits || 0) + Number(interviews || 0)
  _user.interview_credits = next
  if (chargeId) charges.push(chargeId)
  _user.credited_charges = charges
  try {
    await updateUserDoc(String(_user.telegram_id), {
      interview_credits: next,
      credited_charges: charges.slice(-50),
      updated_at: serverTimestamp(),
    })
  } catch (e) {
    console.error('[firebase] creditBundlePurchase failed:', e.message)
  }
  return next
}

/**
 * Check which interview credit would be used (without consuming).
 * Order: free mini hook → trial → paid credits.
 * Returns { ok, used } where used = 'free_mini' | 'trial' | 'paid'.
 * The backend actually consumes the credit atomically with session creation.
 */
export function peekInterview() {
  if (!_user || _user.demo) return { ok: true, used: 'demo' }
  if (!_user.free_mini_used) return { ok: true, used: 'free_mini' }
  if (getTrialDaysLeft() > 0 && Number(_user.trial_credits || 0) > 0) {
    return { ok: true, used: 'trial' }
  }
  if (Number(_user.interview_credits || 0) > 0) {
    return { ok: true, used: 'paid' }
  }
  return { ok: false, used: 'none' }
}

/**
 * Consume one interview. Order: free mini hook → trial → paid credits.
 * Returns { ok, used } where used = 'free_mini' | 'trial' | 'paid'.
 * DEPRECATED: Backend now consumes credits atomically. Use peekInterview() instead.
 */
export async function consumeInterview() {
  if (!_user || _user.demo) return { ok: true, used: 'demo' }
  // 1. Free 3-question mini hook
  if (!_user.free_mini_used) {
    _user.free_mini_used = true
    try {
      await updateUserDoc(String(_user.telegram_id), {
        free_mini_used: true,
        updated_at: serverTimestamp(),
      })
    } catch (e) { console.error('[firebase] consumeInterview mini failed:', e.message) }
    return { ok: true, used: 'free_mini' }
  }
  // 2. Trial credit (within 3 days)
  if (getTrialDaysLeft() > 0 && Number(_user.trial_credits || 0) > 0) {
    _user.trial_credits = Number(_user.trial_credits) - 1
    try {
      await updateUserDoc(String(_user.telegram_id), {
        trial_credits: _user.trial_credits,
        updated_at: serverTimestamp(),
      })
    } catch (e) { console.error('[firebase] consumeInterview trial failed:', e.message) }
    return { ok: true, used: 'trial' }
  }
  // 3. Paid credits
  if (Number(_user.interview_credits || 0) > 0) {
    _user.interview_credits = Number(_user.interview_credits) - 1
    try {
      await updateUserDoc(String(_user.telegram_id), {
        interview_credits: _user.interview_credits,
        updated_at: serverTimestamp(),
      })
    } catch (e) { console.error('[firebase] consumeInterview paid failed:', e.message) }
    return { ok: true, used: 'paid' }
  }
  return { ok: false, used: 'none' }
}

/** Mark the 3-day trial as initialized for a brand-new user. */
export async function initTrialForNewUser() {
  if (!_user || _user.demo || _user.trial_started_at) return
  _user.trial_started_at = Date.now()
  _user.trial_credits = 1
  _user.free_mini_used = false
  _user.interview_credits = _user.interview_credits || 0
  _user.credited_charges = []
  try {
    await updateUserDoc(String(_user.telegram_id), {
      trial_started_at: _user.trial_started_at,
      trial_credits: 1,
      free_mini_used: false,
      interview_credits: _user.interview_credits,
      credited_charges: [],
      updated_at: serverTimestamp(),
    })
  } catch (e) {
    console.error('[firebase] initTrialForNewUser failed:', e.message)
  }
}

/* ---------- referrals ---------- */

/**
 * Claim the referral bonus: +1 free interview for this user AND +1 for the
 * referrer. Backend enforces one claim per user ever (400 on
 * self-referral / already-claimed / unknown referrer).
 * Returns { ok, ... } — never throws.
 */
export async function claimReferral(referrerId) {
  if (!_user || _user.demo) return { ok: false, reason: 'demo' }
  const uid = String(_user.telegram_id || '')
  const rid = String(referrerId || '').trim()
  if (!uid || !rid) return { ok: false, reason: 'missing' }
  if (uid === rid) return { ok: false, reason: 'self' }
  try {
    const res = await api.claimReferral({ userId: uid, referrerId: rid })
    // Sync the local credit count so the UI updates without a refresh.
    if (res && typeof res.user_credits === 'number') {
      _user.interview_credits = res.user_credits
    }
    return { ok: true, ...res }
  } catch (e) {
    return { ok: false, reason: e?.message || 'failed' }
  }
}

/**
 * Referral stats for the invite UI.
 * Returns { count, earned } — friends joined + free interviews earned.
 */
export async function getReferralStats() {
  if (!_user || _user.demo) return { count: 0, earned: 0 }
  const uid = String(_user.telegram_id || '')
  if (!uid) return { count: 0, earned: 0 }
  try {
    const s = await api.getReferralStats(uid)
    return {
      count: Number(s.referral_count || 0),
      earned: Number(s.referral_earned || 0),
    }
  } catch {
    return { count: 0, earned: 0 }
  }
}

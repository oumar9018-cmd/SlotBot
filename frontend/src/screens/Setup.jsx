import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '../store.js'
import { useTelegram } from '../hooks/useTelegram.js'
import { useBackendWake } from '../hooks/useBackendWake.js'
import { markTrialUsed, peekInterview, getUsableInterviews } from '../firebase.js'
import { api, preconnectVoice } from '../api.js'
import { INTERVIEW_STYLES } from '../config/interviewStyles.jsx'
import ProfileChip from '../components/ProfileChip.jsx'
import BackendWakeSplash from '../components/BackendWakeSplash.jsx'
import Grilly from '../components/Grilly.jsx'
import { IconFlame } from '../components/icons.jsx'

const FIELDS = [
  { key: 'name', label: 'YOUR NAME', placeholder: 'e.g. Rahul Sharma', bg: 'bg-peach/60' },
  { key: 'background', label: 'YOUR BACKGROUND', placeholder: 'e.g. B.Tech student', bg: 'bg-lav/60' },
  { key: 'goal', label: 'YOUR GOAL', placeholder: 'e.g. Score 80+ in 3 weeks', bg: 'bg-mint/60' },
]

// Feature 1 (DS-160 Attacker): optional intel so Grilly knows WHERE to attack.
// These ride into the background summary; the backend picks ONE vector per session.
const INTEL_FIELDS = [
  { key: 'university', label: 'UNIVERSITY (optional)', placeholder: 'e.g. Arizona State University', bg: 'bg-sky/60' },
  { key: 'sponsor', label: 'SPONSOR (optional)', placeholder: 'e.g. Father — businessman', bg: 'bg-peach/60' },
  { key: 'pastRejection', label: 'PAST REJECTION (optional)', placeholder: 'e.g. None / 214(b) in 2024', bg: 'bg-rose/60' },
]

export default function Setup() {
  const store = useStore()
  const { name, background, goal, testType, set, go } = store
  const university = store.university || ''
  const sponsor = store.sponsor || ''
  const pastRejection = store.pastRejection || ''
  const { user, haptic } = useTelegram()
  const { awake, showSplash, retryCount, timedOut, activate } = useBackendWake()
  const [waitingBackend, setWaitingBackend] = useState(false)
  const [splashVisible, setSplashVisible] = useState(false)
  // Busy + error state for the begin flow. `starting` blocks double-taps
  // (which would consume TWO credits). `startError` surfaces a failed
  // session start instead of the old silent fallback, which dumped users
  // into a broken room after burning a credit.
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState(null)
  const ready = name.trim() && background.trim() && goal.trim()

  const doBegin = async ({ skipConsume = false } = {}) => {
    if (starting) return
    setStarting(true)
    setStartError(null)
    // Pay-per-use gate: check which interview credit is available
    // (free mini -> trial -> paid). The BACKEND consumes the credit atomically
    // with session creation, so a failed session start no longer burns credits.
    // No credits left -> send to the store.
    let consumeRes = null
    if (!skipConsume) {
      // peekInterview is synchronous (no Firestore write) - just checks availability
      consumeRes = peekInterview()
      if (!api.isMock) {
        if (!consumeRes?.ok) {
          setStarting(false)
          store.refreshCreditState()
          go('bundles')
          return
        }
        store.refreshCreditState()
      }
    }
    // The interview style chosen in TestRoom maps to the backend's test_type,
    // so Grok's persona matches. Style-specific persona block goes into the
    // background summary so the backend includes it in the system prompt.
    // Strictness also rides along as a tone directive — this is what makes
    // the Settings "Gentle / Brutal" toggle actually do something.
    const style = INTERVIEW_STYLES[store.interviewStyle]
    const backendTestType = style?.backendTestType || testType
    const toneDirective = store.strictness === 'normal'
      ? ' [EXAMINER TONE: firm but encouraging coach — demanding, yet wants the candidate to succeed. Interrupt gently.]'
      : ' [EXAMINER TONE: brutal — zero mercy, cold, intimidating.]'
    const styledBackground = style
      ? `[INTERVIEW STYLE: ${style.name}] ${style.persona}${toneDirective} Applicant background: ${background}`
      : `${background}${toneDirective}`
    // Feature 1 (DS-160 Attacker): append intel so the backend knows the
    // candidate's university / sponsor / past rejection for targeted attacks.
    const intelBits = []
    if (university.trim()) intelBits.push(`University: ${university.trim()}`)
    if (sponsor.trim()) intelBits.push(`Sponsor: ${sponsor.trim()}`)
    if (pastRejection.trim()) intelBits.push(`Past rejection: ${pastRejection.trim()}`)
    const fullBackground = intelBits.length
      ? `${styledBackground} [DS-160 INTEL: ${intelBits.join(' | ')}]`
      : styledBackground
    try {
      // Plan routing: paid credits must hit the backend 'paid' branch (900s,
      // gated on user.paid). Sending 'free' for a paid credit would make the
      // backend take the free_trial branch and 402 (trial already used) —
      // burning the user's paid credit with no session.
      const sessionPlan = consumeRes?.used === 'paid' ? 'paid' : 'free'
      const s = await api.startSession({
        telegramId: user?.telegram_id || user?.id,
        name, background: fullBackground, goal, testType: backendTestType, plan: sessionPlan,
      })
      if (!s?.session_id) throw new Error('Backend returned no session id')
      set({ sessionId: s.session_id, maxSeconds: s.max_seconds || 120, plan: sessionPlan, transcript: [] })
      markTrialUsed()
      // Latency: pre-connect the voice WebSocket NOW so TestRoom doesn't pay
      // the handshake + Deepgram STT connect cost on mount.
      preconnectVoice(s.session_id)
      go('test')
    } catch (e) {
      // Session start failed: backend consumes credit atomically with session
      // creation, so NO credit was deducted. Safe to retry normally.
      console.error('[setup] startSession failed:', e?.message || e)
      setStartError(
        "Grilly couldn't start your grilling — the server didn't answer. " +
        'Your credit is safe. Check your internet and try again.'
      )
      haptic('error')
    } finally {
      setStarting(false)
    }
  }

  const begin = () => {
    if (!ready || waitingBackend || starting) return
    haptic('success')
    // Wake gate: if the backend is asleep (Render cold start), show the cute
    // splash and auto-proceed once it's awake instead of failing silently.
    if (!api.isMock && !awake) {
      activate()
      setWaitingBackend(true)
      return
    }
    doBegin()
  }

  // Backend woke up (or gave up after 60s) → proceed with session start
  useEffect(() => {
    if (waitingBackend && (awake || timedOut)) {
      setWaitingBackend(false)
      doBegin()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [waitingBackend, awake, timedOut])

  // Delay the overlay slightly so fast backends never flash the splash
  useEffect(() => {
    if (!waitingBackend) { setSplashVisible(false); return }
    const t = setTimeout(() => setSplashVisible(true), 900)
    return () => clearTimeout(t)
  }, [waitingBackend])

  return (
    <div className="h-full overflow-y-auto dot-bg px-5 pt-5 flex flex-col" style={{ paddingBottom: '110px' }}>
      <div className="flex items-center justify-between mb-4">
        <motion.button
          whileTap={{ scale: 0.88, rotate: -8 }}
          onClick={() => go('home')}
          className="w-11 h-11 rounded-full bg-white border-2 border-cardline shadow-cute flex items-center justify-center text-ink font-extrabold text-lg"
        >
          ←
        </motion.button>
        <ProfileChip onOpenSettings={() => go('settings')} />
      </div>

      {/* cute step header */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-11 h-11 rounded-full candy-mesh text-white font-display font-bold flex items-center justify-center text-lg shadow-cute">1</div>
        <div className="flex-1">
          <div className="font-display font-bold text-ink text-lg leading-tight">Quick intro</div>
          <div className="text-dim text-xs font-bold">Grilly reads this before grilling you</div>
        </div>
        <Grilly mood="happy" size={56} />
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ type: 'spring', stiffness: 140, damping: 16 }}>
        <h2 className="font-display font-bold text-[30px] tracking-tight text-ink leading-tight">
          Who's getting <span className="candy-text">grilled?</span>
        </h2>
        <p className="text-body mt-1.5 text-[15px] font-semibold">Be honest — vague answers get sniffed out in seconds.</p>
      </motion.div>

      <div className="space-y-3.5 mt-5 flex-1">
        {FIELDS.map((f, i) => (
          <motion.div
            key={f.key}
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: 0.08 * i, type: 'spring', stiffness: 180, damping: 17 }}
            className={`${f.bg} border-2 border-white rounded-[26px] p-4 shadow-cute focus-within:border-pinkdeep transition-colors`}
          >
            <label className="text-[11px] font-extrabold tracking-[0.18em] text-body">{f.label}</label>
            <input
              value={store[f.key]}
              onChange={(e) => set({ [f.key]: e.target.value })}
              placeholder={f.placeholder}
              className="mt-1.5 w-full bg-transparent text-ink placeholder:text-dim/60 outline-none font-display font-bold text-lg"
            />
          </motion.div>
        ))}

        {/* DS-160 intel — optional, but Grilly attacks what you reveal */}
        <div className="pt-1">
          <div className="text-[11px] font-extrabold tracking-[0.18em] text-dim mb-2 px-1">
            VISA INTEL <span className="text-pinkdeep">(optional — Grilly WILL probe these)</span>
          </div>
        </div>
        {INTEL_FIELDS.map((f, i) => (
          <motion.div
            key={f.key}
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: 0.24 + 0.08 * i, type: 'spring', stiffness: 180, damping: 17 }}
            className={`${f.bg} border-2 border-white rounded-[26px] p-4 shadow-cute focus-within:border-pinkdeep transition-colors`}
          >
            <label className="text-[11px] font-extrabold tracking-[0.18em] text-body">{f.label}</label>
            <input
              value={store[f.key] || ''}
              onChange={(e) => set({ [f.key]: e.target.value })}
              placeholder={f.placeholder}
              className="mt-1.5 w-full bg-transparent text-ink placeholder:text-dim/60 outline-none font-display font-bold text-lg"
            />
          </motion.div>
        ))}

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="rounded-[26px] bg-sun/70 border-2 border-white p-4 shadow-cute flex gap-3"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2.4" strokeLinecap="round" className="shrink-0 mt-0.5">
            <path d="M12 9v4m0 4h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
          </svg>
          <span className="text-[14px] text-ink font-semibold leading-relaxed">
            <b>Heads up, cutie:</b> Grilly interrupts vague answers and scores without mercy. The free grilling is 2 minutes. The clock does not stop.
          </span>
        </motion.div>
      </div>

      {/* session-start error: visible, honest, retry reuses the consumed credit */}
      {startError && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-5 rounded-[24px] bg-rose-50 border-2 border-rose-300 p-4 flex items-start gap-3"
          role="alert"
        >
          <span className="w-9 h-9 rounded-full bg-rose-500 text-white font-display font-bold flex items-center justify-center shrink-0">
            !
          </span>
          <div className="flex-1">
            <p className="text-[13.5px] font-semibold text-rose-900 leading-snug">{startError}</p>
            <button
              onClick={() => doBegin()}
              disabled={starting}
              className="mt-2.5 px-5 py-2.5 rounded-full candy-mesh text-white text-sm font-display font-bold active:scale-95 transition-transform disabled:opacity-60"
            >
              {starting ? 'RETRYING…' : 'TRY AGAIN'}
            </button>
          </div>
          <button
            onClick={() => setStartError(null)}
            className="shrink-0 w-7 h-7 rounded-full bg-white border border-rose-300 text-rose-700 font-bold text-sm leading-none"
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </motion.div>
      )}

      <motion.button
        whileTap={ready && !starting ? { scale: 0.94 } : {}}
        onClick={begin}
        disabled={!ready || starting}
        className={`w-full py-5 mt-6 rounded-full font-display font-bold text-lg transition-all ${
          ready && !starting ? 'btn-candy text-white candy-mesh' : 'bg-white border-2 border-cardline text-dim'
        }`}
      >
        {starting ? (
          <span className="inline-flex items-center gap-2">
            <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" opacity="0.25" />
              <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
            </svg>
            WAKING GRILLY…
          </span>
        ) : ready ? "LET'S GO — GRILL ME" : 'FILL ALL THREE FIELDS'}
      </motion.button>

      {/* Starting overlay: Grilly popup the moment you tap GO */}
      {starting && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 backdrop-blur-sm">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="bg-white rounded-[32px] border-2 border-cardline shadow-cute p-8 mx-6 text-center max-w-[320px]"
          >
            <motion.div
              animate={{ rotate: [0, -8, 8, 0] }}
              transition={{ repeat: Infinity, duration: 1.2 }}
            >
              <Grilly mood="happy" size={96} className="mx-auto mb-3" />
            </motion.div>
            <div className="font-display font-bold text-ink text-xl mb-1">
              Grilly is waking up…
            </div>
            <div className="text-dim text-sm font-semibold flex items-center justify-center gap-1.5">
              Sharpening questions for you <IconFlame size={18} />
            </div>
            <div className="mt-4 flex justify-center gap-1.5">
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="w-2.5 h-2.5 rounded-full bg-pinkdeep"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ repeat: Infinity, duration: 1, delay: i * 0.2 }}
                />
              ))}
            </div>
          </motion.div>
        </div>
      )}

      {/* Backend wake overlay: cute splash while Render cold-starts */}
      {splashVisible && showSplash && !timedOut && (
        <div className="fixed inset-0 z-50">
          <BackendWakeSplash retryCount={retryCount} />
        </div>
      )}
    </div>
  )
}

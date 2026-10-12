import { useCallback, useEffect, useRef, useState } from 'react'
import { api, wsUrl, takePreconnectedVoice } from '../api.js'
import { useStore } from '../store.js'
// Inline worklet source as a string (Vite ?raw) → Blob URL at runtime.
// Avoids worklet file-path issues on static hosts (GitHub Pages).
import pcm16WorkletCode from '../audio/pcm16-worklet.js?raw'

/**
 * Voice hook: mic capture + WebSocket to backend + examiner audio playback.
 * In MOCK mode, simulates examiner turns locally.
 *
 * Live flow:
 *  - WS connects on mount → backend auto-sends opening examiner line (text + PCM16 audio)
 *  - Binary WS frames = PCM16 24kHz mono examiner voice → played via Web Audio API
 *  - User taps circle → barge-in (stop playback, send interrupt) + mic capture starts
 *  - User taps again → mic stops, sends user_done → backend transcribes & replies
 */
export function useVoice({ sessionId, onTranscript, onInterrupted, onEnded, onDenied, onThinkingTimeout, styleId, enabled = true }) {
  const [listening, setListening] = useState(false)
  const [speaking, setSpeaking] = useState(false) // examiner audio actually playing
  const [thinking, setThinking] = useState(false) // examiner generating reply
  const [interrupted, setInterrupted] = useState(false)
  const [micLevel, setMicLevel] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0) // examiner voice loudness 0..1
  // Connection state for the live voice socket.
  // 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'failed'
  const [wsStatus, setWsStatus] = useState('idle')
  // Human-readable mic permission error (null when OK).
  const [micError, setMicError] = useState(null)
  // Confirmed user transcript ("heard" state): backend sends
  // {type:'transcript_user',text} when speech is finalized, preceded by
  // {type:'status',state:'heard'}. Graceful no-op when never sent.
  const [heardText, setHeardText] = useState('')
  const [heard, setHeard] = useState(false)
  const wsRef = useRef(null)
  const mediaRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)
  const mockTimer = useRef(null)
  const turnRef = useRef(0)
  // REBUILD 2026-10-12: HTMLAudioElement for examiner playback (industry standard).
  // Web Audio API bypasses browser AEC → mic hears speaker → echo/double sound.
  // HTMLAudioElement stays in browser pipeline → AEC works → no echo.
  const audioElRef = useRef(null)
  const audioUrlRef = useRef(null) // current blob URL (for proper revocation)
  const pcmAccumRef = useRef([]) // accumulated PCM16 chunks for current turn
  const audioTurnRef = useRef(0) // turn ID for stale audio rejection
  const workletCtxRef = useRef(null) // AudioContext for PCM16 capture
  const workletNodeRef = useRef(null) // AudioWorkletNode for PCM16 capture
  const styleRef = useRef(styleId)
  styleRef.current = styleId

  // ---- Examiner audio playback (PCM16 24kHz mono) ----
  const playCtxRef = useRef(null) // AudioContext for playback (separate from capture)
  const playQueueRef = useRef([]) // queued BufferSources, in order
  const playHeadRef = useRef(0) // next scheduled start time (ctx.currentTime)
  const audioActiveRef = useRef(false) // true while any examiner audio scheduled/playing
  const thinkingTimeoutRef = useRef(null) // FIX 2026-10-12: timeout for stuck "thinking"

  /** Clear the thinking-timeout (called when examiner responds or errors). */
  const clearThinkingTimeout = () => {
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current)
      thinkingTimeoutRef.current = null
    }
  }

  /** Start a 30s timeout: if backend never responds, unstick the UI.
   *  FIX 2026-10-12: "kabhi response nahi" — frontend waited forever on
   *  "thinking" when backend thinker crashed or Groq timed out. */
  const armThinkingTimeout = () => {
    clearThinkingTimeout()
    thinkingTimeoutRef.current = setTimeout(() => {
      thinkingTimeoutRef.current = null
      // Only fire if still genuinely stuck thinking with no audio
      if (!audioActiveRef.current) {
        console.error('[voice] thinking timeout — backend did not respond in 30s')
        setThinking(false)
        // Notify TestRoom so it can show a retry hint
        if (typeof onThinkingTimeout === 'function') {
          try { onThinkingTimeout() } catch { /* noop */ }
        }
      }
    }, 30000)
  }
  const levelRafRef = useRef(null)

  const pushTranscript = useCallback((role, text) => {
    onTranscript && onTranscript({ role, text, at: Date.now() })
  }, [onTranscript])

  // ---- Streaming examiner text (live mode): reveal word-by-word so the
  // transcript feels alive instead of popping in all at once. ----
  const streamRef = useRef(null)
  const streamExaminerText = useCallback((fullText) => {
    if (streamRef.current) clearInterval(streamRef.current)
    const updateLast = useStore.getState().updateLastTranscript
    pushTranscript('examiner', '')
    const words = fullText.split(' ')
    let i = 0
    streamRef.current = setInterval(() => {
      i += 3
      if (i >= words.length) {
        clearInterval(streamRef.current)
        streamRef.current = null
        updateLast({ text: fullText })
      } else {
        updateLast({ text: words.slice(0, i).join(' ') })
      }
    }, 90)
  }, [pushTranscript])

  const cancelStream = useCallback(() => {
    if (streamRef.current) {
      clearInterval(streamRef.current)
      streamRef.current = null
    }
  }, [])

  function getPlayCtx() {
    if (!playCtxRef.current) {
      const Ctx = window.AudioContext || window.webkitAudioContext
      playCtxRef.current = new Ctx({ sampleRate: 24000 })
    }
    return playCtxRef.current
  }

  /** Ensure playback AudioContext is running before scheduling audio.
   *  FIX 2026-10-12: getPlayCtx() called ctx.resume() without awaiting —
   *  the first audio chunk was scheduled on a suspended context and got
   *  silently dropped (user heard voice "cut at start"). Now we await. */
  async function ensurePlayCtxRunning() {
    const ctx = getPlayCtx()
    if (ctx.state === 'suspended') {
      try { await ctx.resume() } catch { /* noop */ }
    }
    return ctx
  }

  // FIX 2026-10-12: Serial playback queue — the async ensurePlayCtxRunning()
  // introduced a race: two chunks arriving together both read the same
  // playHeadRef and scheduled overlapping audio ("double voice").
  // Now chunks chain strictly in order via a promise tail.
  const playTailRef = useRef(Promise.resolve())

  /** Stop all examiner playback immediately (barge-in / unmount). */
  const stopPlayback = useCallback(() => {
    // New turn: invalidate any queued chunks from the old turn.
    audioTurnRef.current += 1
    // REBUILD: stop HTMLAudioElement
    if (audioElRef.current) {
      try { audioElRef.current.pause() } catch { /* noop */ }
      try { audioElRef.current.currentTime = 0 } catch { /* noop */ }
    }
    pcmAccumRef.current = []
    // Legacy Web Audio cleanup (kept for safety)
    for (const src of playQueueRef.current) {
      try { src.onended = null } catch { /* noop */ }
      try { src.stop() } catch { /* noop */ }
      try { src.disconnect() } catch { /* noop */ }
    }
    playQueueRef.current = []
    playHeadRef.current = 0
    audioActiveRef.current = false
    // Reset the serial chain so stale queued chunks don't play after barge-in
    playTailRef.current = Promise.resolve()
    cancelAnimationFrame(levelRafRef.current)
    setSpeaking(false)
    setAudioLevel(0)
  }, [])

  /** REBUILD 2026-10-12: Accumulate PCM16 chunks, play via HTMLAudioElement.
   *  Why: Web Audio API bypasses browser AEC → mic hears speaker → echo.
   *  HTMLAudioElement stays in browser pipeline → AEC cancels echo.
   *  Chunks accumulate; when turn completes (status=listening), we build a
   *  WAV blob and play it. Turn IDs reject stale audio. */
  const playPcm16Chunk = useCallback((arrayBuffer) => {
    if (!arrayBuffer || arrayBuffer.byteLength === 0) return
    // Accumulate for current turn
    pcmAccumRef.current.push(new Uint8Array(arrayBuffer))
    // Update speaking state and audio level from chunk energy
    setSpeaking(true)
    const data = new Int16Array(arrayBuffer)
    let sum = 0
    for (let i = 0; i < data.length; i += 16) sum += Math.abs(data[i])
    const avg = sum / (data.length / 16) / 32768
    setAudioLevel(Math.min(1, avg * 3))
  }, [])

  /** Convert accumulated PCM16 24kHz mono to WAV blob and play via HTMLAudioElement. */
  const flushAndPlayAudio = useCallback(() => {
    const chunks = pcmAccumRef.current
    pcmAccumRef.current = []
    if (chunks.length === 0) {
      setSpeaking(false)
      setAudioLevel(0)
      return
    }
    const turn = audioTurnRef.current
    // Combine chunks
    let totalLen = 0
    for (const c of chunks) totalLen += c.length
    const pcmData = new Uint8Array(totalLen)
    let offset = 0
    for (const c of chunks) {
      pcmData.set(c, offset)
      offset += c.length
    }
    // Build WAV (16-bit PCM, 24kHz, mono)
    const wavBuffer = new ArrayBuffer(44 + pcmData.length)
    const view = new DataView(wavBuffer)
    // RIFF header
    const writeStr = (off, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i))
    }
    writeStr(0, 'RIFF')
    view.setUint32(4, 36 + pcmData.length, true)
    writeStr(8, 'WAVE')
    writeStr(12, 'fmt ')
    view.setUint32(16, 16, true) // fmt chunk size
    view.setUint16(20, 1, true) // PCM
    view.setUint16(22, 1, true) // mono
    view.setUint32(24, 24000, true) // sample rate
    view.setUint32(28, 24000 * 2, true) // byte rate
    view.setUint16(32, 2, true) // block align
    view.setUint16(34, 16, true) // bits per sample
    writeStr(36, 'data')
    view.setUint32(40, pcmData.length, true)
    // PCM data (with 1.8x gain)
    const pcm16 = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.length / 2)
    const outView = new DataView(wavBuffer, 44)
    for (let i = 0; i < pcm16.length; i++) {
      let s = (pcm16[i] / 32768) * 1.8
      if (s > 1) s = 1 - 1 / (1 + (s - 1) * 4) * 0.25
      else if (s < -1) s = -1 + 1 / (1 + (-s - 1) * 4) * 0.25
      outView.setInt16(i * 2, s * 32767, true)
    }
    // Play via HTMLAudioElement (browser AEC works!)
    const blob = new Blob([wavBuffer], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    if (!audioElRef.current) {
      audioElRef.current = new Audio()
      // BUGFIX: revoke via ref (not closure) so every turn's URL is freed
      audioElRef.current.onended = () => {
        setSpeaking(false)
        setAudioLevel(0)
        audioActiveRef.current = false
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current)
          audioUrlRef.current = null
        }
      }
      audioElRef.current.onerror = () => {
        setSpeaking(false)
        setAudioLevel(0)
        audioActiveRef.current = false
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current)
          audioUrlRef.current = null
        }
      }
    }
    const audioEl = audioElRef.current
    // Stale turn? Don't play.
    if (turn !== audioTurnRef.current) {
      URL.revokeObjectURL(url)
      return
    }
    // BUGFIX: revoke previous URL before assigning new one
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
    }
    audioUrlRef.current = url
    audioEl.src = url
    // BUGFIX: track real playback state so barge-in works
    audioActiveRef.current = true
    audioEl.play().then(() => {
      // playing
    }).catch((err) => {
      console.error('[voice] audio element play failed', err)
      setSpeaking(false)
      audioActiveRef.current = false
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
        audioUrlRef.current = null
      }
    })
  }, [])

  /** Internal: schedule a PCM16 chunk on a running AudioContext. */
  function schedulePcm16Chunk(ctx, arrayBuffer) {
    if (!arrayBuffer || arrayBuffer.byteLength === 0) return
    const data = new Int16Array(arrayBuffer)
    if (data.length === 0) return

      // RMS loudness for the visual wave (sample every 8th to save CPU)
      let sum = 0, n = 0
      for (let i = 0; i < data.length; i += 8) { sum += data[i] * data[i]; n++ }
      const rms = n > 0 ? Math.sqrt(sum / n) / 32768 : 0
      // Smooth the level meter toward the new RMS
      setAudioLevel((prev) => Math.min(1, Math.max(prev * 0.6, Math.min(1, rms * 4))))

      const floatData = new Float32Array(data.length)
      // GAIN: measured safe boost. Backend RMS ~6.3%, target ~12% for clear
      // mobile playback. 1.8x gain with soft limiter — no harsh clipping.
      // (Old 2.5x was too aggressive per GPT review.)
      for (let i = 0; i < data.length; i++) {
        let s = (data[i] / 32768) * 1.8
        // Soft limiter: gentle saturation instead of hard clip
        if (s > 1) s = 1 - 1 / (1 + (s - 1) * 4) * 0.25
        else if (s < -1) s = -1 + 1 / (1 + (-s - 1) * 4) * 0.25
        floatData[i] = s
      }
      // FIX 2026-10-12: Resample 24kHz TTS to context's native rate.
      // Mobile browsers often ignore {sampleRate: 24000} and run at 44100/48000.
      // The old code created a 24kHz buffer and relied on browser resampling,
      // which cuts/glitches on mobile ("har sentence adha sunai deta hai").
      // Manual linear resampling is bulletproof.
      const srcRate = 24000
      const dstRate = ctx.sampleRate
      let outData = floatData
      let outRate = srcRate
      if (dstRate !== srcRate) {
        const ratio = srcRate / dstRate
        const outLen = Math.floor(floatData.length / ratio)
        outData = new Float32Array(outLen)
        for (let i = 0; i < outLen; i++) {
          const srcIdx = i * ratio
          const i0 = Math.floor(srcIdx)
          const i1 = Math.min(i0 + 1, floatData.length - 1)
          const frac = srcIdx - i0
          outData[i] = floatData[i0] * (1 - frac) + floatData[i1] * frac
        }
        outRate = dstRate
      }
      const buffer = ctx.createBuffer(1, outData.length, outRate)
      buffer.getChannelData(0).set(outData)
      const source = ctx.createBufferSource()
      source.buffer = buffer
      source.connect(ctx.destination)

      // Schedule back-to-back so chunks play seamlessly in order.
      // 250ms initial buffer absorbs mobile network jitter (was 150ms — cutting
      // on mobile); clamp chunks that would start >0.8s late (long stall silences).
      const now = ctx.currentTime
      if (playHeadRef.current < now + 0.25) playHeadRef.current = now + 0.25
      else if (playHeadRef.current > now + 0.8) playHeadRef.current = now + 0.25
      source.start(playHeadRef.current)
      playHeadRef.current += buffer.duration

      playQueueRef.current.push(source)
      if (!audioActiveRef.current) {
        audioActiveRef.current = true
        setSpeaking(true)
      }
      source.onended = () => {
        const idx = playQueueRef.current.indexOf(source)
        if (idx >= 0) playQueueRef.current.splice(idx, 1)
        try { source.disconnect() } catch { /* noop */ }
        if (playQueueRef.current.length === 0) {
          audioActiveRef.current = false
          // Grace period: backend may still be streaming more chunks
          setTimeout(() => {
            if (!audioActiveRef.current && playQueueRef.current.length === 0) {
              setSpeaking(false)
              setAudioLevel(0)
            }
          }, 350)
        }
      }
  }

  // ---- Mock examiner simulation ----
  const mockExaminerTurn = useCallback(() => {
    setThinking(true)
    // Simulate "thinking" delay before speaking
    setTimeout(() => {
      setThinking(false)
      setSpeaking(true)
      const line = api.mockExaminerLine(turnRef.current++, styleRef.current)
      // Simulate streaming words
      const words = line.split(' ')
      let i = 0
      mockTimer.current = setInterval(() => {
        i += 2
        if (i >= words.length) {
          clearInterval(mockTimer.current)
          pushTranscript('examiner', line)
          setSpeaking(false)
        }
      }, 260)
    }, 900)
  }, [pushTranscript])

  const simulateInterruption = useCallback(() => {
    setInterrupted(true)
    onInterrupted && onInterrupted()
    setTimeout(() => setInterrupted(false), 1200)
    pushTranscript('examiner', 'Stop. Answer the question I asked. No rambling.')
  }, [onInterrupted, pushTranscript])

  // ---- Mic level meter ----
  const startMeter = useCallback((stream) => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext
      const ctx = new Ctx()
      const src = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 256
      src.connect(analyser)
      analyserRef.current = { ctx, analyser }
      const data = new Uint8Array(analyser.frequencyBinCount)
      const tick = () => {
        analyser.getByteFrequencyData(data)
        const avg = data.reduce((a, b) => a + b, 0) / data.length
        setMicLevel(Math.min(1, avg / 90))
        rafRef.current = requestAnimationFrame(tick)
      }
      tick()
    } catch { /* meter optional */ }
  }, [])

  const stopMeter = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    try { analyserRef.current?.ctx.close() } catch { /* noop */ }
    setMicLevel(0)
  }, [])

  // ---- WS message handler (stable via ref; always calls latest callbacks) ----
  const handlerRef = useRef(null)
  handlerRef.current = (ev) => {
    const data = ev.data
    // Binary examiner audio?
    if (data instanceof ArrayBuffer) {
      playPcm16Chunk(data)
      return
    }
    if (typeof Blob !== 'undefined' && data instanceof Blob) {
      data.arrayBuffer().then(playPcm16Chunk).catch(() => {})
      return
    }
    // JSON control messages
    let msg
    try {
      msg = JSON.parse(data)
    } catch { return /* ignore unparseable */ }
    if (msg.type === 'transcript_user') {
      // Backend finalized the user's speech — confirms the mic worked.
      const text = msg.text || ''
      setHeardText(text)
      setHeard(true)
      pushTranscript('user', text)
    } else if (msg.type === 'transcript_examiner') {
      clearThinkingTimeout()
      setThinking(false)
      // New examiner turn: increment turn ID so stale audio chunks are dropped.
      audioTurnRef.current += 1
      // Stream the text word-by-word for a live feel (audio plays in parallel)
      streamExaminerText(msg.text)
      // speaking state is driven by actual audio playback; this is a fallback
      // in case audio frames fail, so the UI doesn't hang on "thinking"
      setTimeout(() => {
        if (!audioActiveRef.current) setThinking(false)
      }, 4000)
    } else if (msg.type === 'status') {
      if (msg.state === 'heard') {
        // Backend heard the user — brief "Got it!" confirmation before thinking.
        setHeard(true)
        setThinking(false)
      } else if (msg.state === 'thinking') {
        // Transition to thinking — clear the "heard" confirmation.
        setHeard(false)
        setThinking(true)
        armThinkingTimeout() // FIX: unstick UI if backend never responds
      } else if (msg.state === 'speaking') {
        setThinking(false)
        // audio chunks incoming; speaking=true set by playPcm16Chunk
      } else if (msg.state === 'listening') {
        setThinking(false)
        // REBUILD: turn complete — flush accumulated audio via HTMLAudioElement
        flushAndPlayAudio()
      } else if (msg.state === 'connecting') {
        setThinking(true)
      }
    } else if (msg.type === 'interrupted') {
      stopPlayback(); cancelStream()
      simulateInterruption()
    } else if (msg.type === 'session_ended') {
      stopPlayback()
      onEnded && onEnded()
    } else if (msg.type === 'visa_denied') {
      // Feature 4: instant cut-off — the officer ended the interview.
      stopPlayback()
      if (onDenied) onDenied(msg.reason || 'Over-explaining / Vague Answers', msg.over_explain_count || 3)
      else onEnded && onEnded()
    } else if (msg.type === 'overexplain_cutoff') {
      // Feature 4: warning — the officer cut the candidate off (not fatal yet).
      // Surface as a brief interruption flash; TestRoom can show a toast.
      if (onInterrupted) onInterrupted()
    } else if (msg.type === 'error') {
      console.error('[voice] backend error:', msg.message)
      clearThinkingTimeout()
      setThinking(false)
    }
  }

  // ---- Connect WS on mount (live mode). Backend auto-sends the opening line. ----
  // Gated on `enabled` so callers (TestRoom backend-wake splash) can hold the
  // connection until the backend is confirmed awake.
  // Latency: prefer a socket pre-connected during Setup (see preconnectVoice).
  //
  // Resilience (v2): Render's proxy kills idle WebSockets, and mobile networks
  // drop sockets on network switches. Strategy:
  //  - Keepalive ping every 20s (backend responds with pong)
  //  - Exponential backoff reconnect: 1s, 2s, 4s, 8s, 16s (5 attempts)
  //  - Only show the scary "reconnecting" overlay after 2 failed attempts;
  //    quick blips reconnect silently in the background
  //  - Validate the preconnected socket is actually OPEN before adopting it
  const reconnectTimerRef = useRef(null)
  const keepaliveTimerRef = useRef(null)
  const attemptRef = useRef(0)
  const MAX_RECONNECT_ATTEMPTS = 5
  const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000]
  useEffect(() => {
    if (!enabled) return
    if (api.isMock) return
    const url = wsUrl(sessionId)
    if (!url) { setWsStatus('failed'); return }

    let cancelled = false

    // Keepalive: ping every 20s so Render's proxy doesn't kill idle sockets.
    const startKeepalive = (sock) => {
      stopKeepalive()
      keepaliveTimerRef.current = setInterval(() => {
        if (cancelled) return
        try {
          if (sock.readyState === WebSocket.OPEN) {
            sock.send(JSON.stringify({ type: 'ping' }))
          }
        } catch { /* socket dead — onclose will handle */ }
      }, 20000)
    }
    const stopKeepalive = () => {
      if (keepaliveTimerRef.current) {
        clearInterval(keepaliveTimerRef.current)
        keepaliveTimerRef.current = null
      }
    }

    const attach = (sock) => {
      wsRef.current = sock
      attemptRef.current = 0 // reset backoff on successful connect
      setWsStatus('connected')
      startKeepalive(sock)
      // Wrap the message handler to also catch pong (latency probe)
      const baseHandler = (ev) => handlerRef.current && handlerRef.current(ev)
      sock.onmessage = (ev) => {
        // Fast-path pong without bothering the transcript handler
        if (typeof ev.data === 'string') {
          try {
            const m = JSON.parse(ev.data)
            if (m && m.type === 'pong') return
          } catch { /* not JSON — fall through */ }
        }
        baseHandler(ev)
      }
      sock.onerror = (e) => console.error('[voice] ws error', e)
      sock.onclose = () => {
        if (cancelled) return
        stopKeepalive()
        if (wsRef.current === sock) wsRef.current = null
        drop(sock)
      }
    }

    const drop = (sock) => {
      if (cancelled) return
      const attempt = attemptRef.current
      if (attempt >= MAX_RECONNECT_ATTEMPTS) {
        console.error('[voice] ws reconnect exhausted — giving up')
        setWsStatus('failed')
        return
      }
      attemptRef.current = attempt + 1
      const delay = RECONNECT_DELAYS[Math.min(attempt, RECONNECT_DELAYS.length - 1)]
      // Only show the scary overlay after 2 failed attempts — quick blips
      // (network switch, brief proxy hiccup) reconnect silently.
      if (attempt >= 1) setWsStatus('reconnecting')
      console.warn(`[voice] ws dropped — retry ${attempt + 1}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms`)
      reconnectTimerRef.current = setTimeout(() => {
        if (cancelled) return
        let fresh = null
        try {
          fresh = new WebSocket(url)
          fresh.binaryType = 'arraybuffer'
        } catch (err) {
          console.error('[voice] ws reconnect failed', err)
          drop(sock) // count this as a failed attempt, backoff again
          return
        }
        // Only show 'connecting' if we're already showing the overlay
        if (attempt >= 1) setWsStatus('connecting')
        const giveUp = setTimeout(() => {
          if (fresh.readyState !== WebSocket.OPEN) {
            try { fresh.close() } catch { /* noop */ }
            if (!cancelled) drop(sock)
          }
        }, 10000)
        fresh.onopen = () => { clearTimeout(giveUp); attach(fresh) }
        fresh.onclose = () => {
          clearTimeout(giveUp)
          if (!cancelled) {
            if (wsRef.current === fresh) wsRef.current = null
            drop(sock)
          }
        }
        fresh.onerror = () => { /* onclose carries the failure */ }
      }, delay)
    }

    let ws = takePreconnectedVoice(sessionId)
    // Validate the preconnected socket is actually alive — a stale socket
    // from Setup would instantly drop and burn a reconnect attempt.
    if (ws && ws.readyState !== WebSocket.OPEN) {
      try { ws.close() } catch { /* noop */ }
      ws = null
    }
    let adopted = !!ws
    try {
      if (!ws) {
        ws = new WebSocket(url)
        ws.binaryType = 'arraybuffer' // binary frames arrive as ArrayBuffer
      } else {
        // Adopted socket: ensure binary frames arrive as ArrayBuffer
        try { ws.binaryType = 'arraybuffer' } catch { /* noop */ }
      }
    } catch (err) {
      console.error('[voice] ws connect failed', err)
      setWsStatus('failed')
      return
    }

    if (adopted) {
      // Pre-connected during Setup and verified OPEN — adopt it.
      attach(ws)
    } else {
      setWsStatus('connecting')
      const giveUp = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          try { ws.close() } catch { /* noop */ }
          if (!cancelled) setWsStatus('failed')
        }
      }, 15000)
      ws.onopen = () => { clearTimeout(giveUp); attach(ws) }
      ws.onclose = () => {
        clearTimeout(giveUp)
        if (!cancelled) drop(ws)
      }
      ws.onerror = () => { /* onclose carries the failure */ }
    }

    return () => {
      cancelled = true
      clearTimeout(reconnectTimerRef.current)
      stopKeepalive()
      attemptRef.current = 0
      try { ws && ws.close() } catch { /* noop */ }
      if (wsRef.current === ws) wsRef.current = null
      setWsStatus('idle')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, enabled])

  // ---- Start / stop talking (mic only; WS is already connected) ----
  const startTalking = useCallback(async () => {
    // Barge-in: user cutting off the examiner — stop playback AND text stream NOW
    cancelStream()
    setMicError(null) // fresh attempt, clear any previous mic error
    setHeard(false) // new user turn — clear the last confirmed transcript
    setHeardText('')
    if (audioActiveRef.current) {
      stopPlayback()
      try {
        if (wsRef.current && wsRef.current.readyState === 1) {
          wsRef.current.send(JSON.stringify({ type: 'interrupt' }))
        }
      } catch { /* noop */ }
    }
    try {
      // FIX 2026-10-12: Robust mic startup for mobile. The strict constraints
      // (sampleRate 16000, etc.) fail silently on many mobile devices, causing
      // "baar baar bolna padta" — mic appears active but captures nothing.
      // Try ideal first, fall back to basic audio: true.
      let stream = null
      const micAttempts = [
        { echoCancellation: true, noiseSuppression: true, autoGainControl: true,
          sampleRate: 16000, channelCount: 1 },
        { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        true, // last resort: any audio
      ]
      let micErr = null
      for (const constraints of micAttempts) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: constraints })
          // Verify the stream actually has audio tracks
          if (stream && stream.getAudioTracks().length > 0) break
          try { stream.getTracks().forEach(t => t.stop()) } catch { /* noop */ }
          stream = null
        } catch (err) {
          micErr = err
          stream = null
        }
      }
      if (!stream) throw micErr || new Error('No mic stream')
      mediaRef.current = stream
      startMeter(stream)
      setListening(true)

      if (api.isMock) {
        // In mock mode, stop after ~4s and trigger examiner turn
        setTimeout(() => {
          stopTalking()
          pushTranscript('user', '…')
          if (Math.random() < 0.3) simulateInterruption()
          else setTimeout(mockExaminerTurn, 600)
        }, 4000)
        return
      }

      // ---- PCM16 16kHz mono capture via AudioWorklet ----
      // Backend (Deepgram) needs raw PCM16 bytes, not WebM/Opus.
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext
        const audioCtx = new Ctx()
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume().catch(() => {})
        }
        const blob = new Blob([pcm16WorkletCode], { type: 'application/javascript' })
        const workletUrl = URL.createObjectURL(blob)
        try {
          await audioCtx.audioWorklet.addModule(workletUrl)
        } finally {
          URL.revokeObjectURL(workletUrl)
        }
        const source = audioCtx.createMediaStreamSource(stream)
        const workletNode = new AudioWorkletNode(audioCtx, 'pcm16-worklet')
        workletNode.port.onmessage = (e) => {
          // e.data is an ArrayBuffer of PCM16 16kHz mono samples
          const ws = wsRef.current
          if (ws && ws.readyState === 1) ws.send(e.data)
        }
        source.connect(workletNode)
        // NOTE: do NOT connect to destination — no mic feedback.
        workletCtxRef.current = audioCtx
        workletNodeRef.current = workletNode
      } catch (err) {
        console.error('[voice] AudioWorklet PCM16 capture failed', err)
      }
    } catch (err) {
      console.error('mic error', err)
      const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError')
      setMicError(
        denied
          ? 'Microphone blocked. Allow mic access in your browser / Telegram settings, then tap the circle again.'
          : 'Could not start the microphone. Check your connection and tap the circle again.'
      )
    }
  }, [startMeter, pushTranscript, simulateInterruption, mockExaminerTurn, stopPlayback, cancelStream])

  const stopTalking = useCallback(() => {
    setListening(false)
    stopMeter()
    try { mediaRef.current?.getTracks().forEach(t => t.stop()) } catch { /* noop */ }
    // Tear down PCM16 AudioWorklet capture
    try { workletNodeRef.current?.disconnect() } catch { /* noop */ }
    try { workletCtxRef.current?.close() } catch { /* noop */ }
    workletNodeRef.current = null
    workletCtxRef.current = null
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ type: 'user_done' }))
    }
    // Backend will transcribe, think (status: thinking), then reply with audio
    setThinking(true)
  }, [stopMeter])

  useEffect(() => () => {
    clearInterval(mockTimer.current)
    cancelAnimationFrame(levelRafRef.current)
    if (streamRef.current) { clearInterval(streamRef.current); streamRef.current = null }
    clearThinkingTimeout()
    stopTalking()
    stopPlayback()
    try { wsRef.current?.close() } catch { /* noop */ }
    try { playCtxRef.current?.close() } catch { /* noop */ }
  }, [stopTalking, stopPlayback])

  return { listening, speaking, thinking, audioLevel, interrupted, micLevel, startTalking, stopTalking, mockExaminerTurn, wsStatus, micError, clearMicError: () => setMicError(null), heard, heardText }
}

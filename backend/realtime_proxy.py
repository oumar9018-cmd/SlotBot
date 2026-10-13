"""
Turn-based voice pipeline: Mini App <-> FastAPI <-> Deepgram STT <-> Groq (Groq) <-> Deepgram TTS.

Frontend protocol (over this WebSocket):
  client -> server : binary PCM16 16kHz mono audio chunks (mic)
  client -> server : {"type": "interrupt"}            (barge-in)
  client -> server : {"type": "user_done"}            (frontend stop-talk hint; flushes pending utterance)
  server -> client : binary PCM16 24kHz mono audio chunks (examiner voice)
  server -> client : {"type":"status","state":"..."}  (connected/connecting/listening/thinking/speaking/ended)
  server -> client : {"type":"transcript_user","text":"..."}   (echo of flushed user utterance — instant "Heard you ✓")
  server -> client : {"type":"transcript_examiner","text":"..."}  (compat: current frontend build)
  server -> client : {"type":"interrupted"}            (compat: current frontend build)
  server -> client : {"type":"timeup"}                (hard cutoff hit)
  server -> client : {"type":"session_ended"}         (compat: current frontend build)
  server -> client : {"type":"error","message":"..."}

Backend flow:
  1. Mic audio chunks -> Deepgram STT WebSocket (nova-3, endpointing 2000ms)
  2. Final utterance transcript -> Groq chat completions (Groq, OpenAI-compatible REST)
  3. Examiner reply text -> Deepgram TTS STRICT voice (aura-asteria-en, linear16 24kHz)
  4. TTS audio streamed back to frontend in chunks; barge-in cancels mid-stream
  5. Feedback/verdict audio uses FRIENDLY coach voice (aura-luna-en) via /session/feedback-audio

API keys never leave this backend. Transcript accumulates here and is flushed
to Firestore when the socket closes (or the time limit fires).
"""
import asyncio
import json
import logging
import os
import random
import re
import time

from fastapi import WebSocket, WebSocketDisconnect
import httpx
import websockets

import firestore_db as db

log = logging.getLogger("englishtest.proxy")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL_FALLBACK = os.environ.get("GROQ_MODEL_FALLBACK", "qwen/qwen3.8-27b")
GROQ_SCORING_MODEL = os.environ.get("GROQ_SCORING_MODEL", "openai/gpt-oss-20b")

DG_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3&language=en&smart_format=true"
    "&interim_results=true&endpointing=1200"
    "&encoding=linear16&sample_rate=16000&channels=1"
    "&punctuate=true&profanity_filter=false&diarize=false"
    # NOTE: keyterm/keywords removed 2026-10-12 — nova-3 rejects them with
    # HTTP 400, breaking all voice interviews. Indian-name boosting disabled
    # until Deepgram's supported biasing param for nova-3 is confirmed.
    # FIX 2026-10-13: endpointing 4000→1200. 4s was too long — user taps stop
    # but Deepgram hadn't emitted finals yet → "not_heard" → 3x repeat bug.
)
DG_TTS_STRICT_URL = (
    "https://api.deepgram.com/v1/speak"
    "?model=aura-asteria-en&encoding=linear16&sample_rate=24000"
)
DG_TTS_FRIENDLY_URL = (
    "https://api.deepgram.com/v1/speak"
    "?model=aura-luna-en&encoding=linear16&sample_rate=24000"
)
# Back-compat alias (interview default)
DG_TTS_URL = DG_TTS_STRICT_URL
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

TTS_CHUNK = 16384  # ~0.34s of 24kHz 16-bit mono per chunk
WEBM_MAGIC = b"\x1aE\xdf\xa3"  # EBML header — detects wrong audio container

# Load persona once at import
_PERSONA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PERSONA.md")
try:
    with open(_PERSONA_PATH, "r", encoding="utf-8") as f:
        PERSONA_INSTRUCTIONS = f.read()
except FileNotFoundError:
    PERSONA_INSTRUCTIONS = "You are a strict English oral examiner conducting an advanced speaking test. Be stern, interrupt rambling, probe vague answers for specifics."
    log.warning("PERSONA.md not found, using fallback instructions")

TEST_TYPE_HINTS = {
    "academic": "Academic speaking test — probe study plans, academic background, and future goals.",
    "business": "Business speaking test — probe work experience, professional background, and career plans.",
    "general": "General speaking test — probe personal background and everyday topics in depth.",
}

# ---------- Feature 1: Dynamic DS-160 Attacker ----------
# Each session, the officer picks ONE vector to aggressively probe.
# The last-used vector is excluded (stored per user) so the candidate
# can never predict where the attack comes from.
ATTACK_VECTORS = {
    "university": (
        "the candidate's UNIVERSITY choice. Attack it relentlessly: why THIS school and not "
        "a better/worse one? Is it a visa-mill? What do you even know about the campus? "
        "Ask 2-3 sharp follow-ups on the university alone."
    ),
    "sponsor": (
        "the candidate's SPONSOR and funding source. Who exactly is paying? How did they earn "
        "that money? Show me the proof. Attack any vagueness about the sponsor's income, "
        "job, or relationship to the candidate. 2-3 follow-ups minimum."
    ),
    "course": (
        "the candidate's COURSE / field of study. Why this field? What will you actually DO "
        "with this degree? Is this just a cover to immigrate? Demand specifics about "
        "subjects, professors, career plans. 2-3 follow-ups."
    ),
    "funding": (
        "the candidate's FINANCIAL proof. Exact amounts, bank statements, loans, who arranged "
        "the money and how. If numbers don't add up, pounce immediately. 2-3 follow-ups "
        "on money alone."
    ),
    "past_rejection": (
        "any PAST VISA REJECTION or weak travel history. What changed since the refusal? "
        "Why should I believe you now? Attack inconsistencies between then and now. "
        "2-3 follow-ups if a rejection is mentioned or suspected."
    ),
}


def pick_attack_vector(user_id: str) -> str:
    """Pick a random attack vector, excluding the one used last session."""
    try:
        last = db.get_last_attack_vector(user_id)
    except Exception:
        last = None
    pool = [v for v in ATTACK_VECTORS if v != last] or list(ATTACK_VECTORS)
    chosen = random.choice(pool)
    try:
        db.set_last_attack_vector(user_id, chosen)
    except Exception:
        pass
    log.info("attack vector for user %s: %s (last was %s)", user_id, chosen, last)
    return chosen


# ---------- Feature 2: Nervousness & Hesitation Penalty ----------
# Filler words counted per user utterance; ratio -> 0-100 nervousness score.
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "like", "you know", "actually",
    "basically", "hmm", "er", "ah", "ahh", "well",
}
_FILLER_RE = re.compile(
    r"\b(" + "|".join(sorted(FILLER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def count_fillers(text: str) -> int:
    """Count filler-word occurrences in a transcript chunk."""
    return len(_FILLER_RE.findall(text or ""))


def nervousness_from_counts(filler_count: int, word_count: int) -> int:
    """Filler ratio -> 0-100. ~2% is normal speech; 6%+ reads as very nervous."""
    if word_count <= 0:
        return 0
    ratio = filler_count / word_count
    return max(0, min(100, int(ratio * 1000)))


# ---------- Feature 3: "Lie Detector" Cross-Questioning ----------
# Specific numbers / amounts / company names the candidate mentions are
# extracted and held in session state; the officer cross-checks them later.
_FACT_PATTERNS = [
    r"\$\s?[\d,]+(?:\.\d+)?",                                  # $40,000
    r"\b\d[\d,]*\s*(?:lakhs?|crores?)\b",                      # 20 lakh
    r"\b\d+\s*backlogs?\b",                                    # 3 backlogs
    r"\b\d+(?:\.\d+)?\s*(?:cgpa|gpa|percent|%)\b",              # 8.5 CGPA
    r"\b(?:works?|worked|working|employed)\s+(?:at|for|with)\s+([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,2})",  # works at TCS
]


def extract_facts(text: str) -> list[str]:
    """Pull concrete facts (amounts, numbers, company names) from an utterance."""
    facts: list[str] = []
    for pat in _FACT_PATTERNS:
        try:
            for m in re.finditer(pat, text or "", re.IGNORECASE):
                fact = (m.group(1) if m.lastindex else m.group(0)).strip()
                if fact and len(fact) <= 60:
                    facts.append(fact)
        except re.error:
            continue
    # de-dupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for f in facts:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out[:8]


# ---------- Feature 4: Instant Cut-Off ----------
# Real officers don't listen to stories. Answers running past this many
# seconds of audio get aggressively cut off; 3 cut-offs = instant denial.
OVEREXPLAIN_SECONDS = 20
OVEREXPLAIN_LIMIT = 3


def build_system_prompt(test_type: str, background_summary: str | None,
                        attack_vector: str | None = None) -> str:
    prompt = PERSONA_INSTRUCTIONS
    prompt += "\n\nTEST FOCUS: " + TEST_TYPE_HINTS.get(test_type, TEST_TYPE_HINTS["general"])
    if background_summary:
        prompt += (
            "\n\nAPPLICANT BACKGROUND FORM SUMMARY (use for contradiction traps):\n"
            + background_summary.strip()
        )
    prompt += (
        "\n\nBRUTAL OFFICER RULES (apply ruthlessly):"
        "\n1. RED FLAG DETECTOR: If the candidate uses any of these words — 'work', 'settle', "
        "'job', 'try', 'maybe', 'I think', 'probably' — call it out IMMEDIATELY. "
        "Example: \"You said 'work'. On a tourist visa? Explain yourself.\" "
        "Penalize vagueness heavily in your mental scoring."
        "\n2. INTERRUPTION HACK: If the candidate rambles (long, unfocused answer), cut in with "
        "\"Keep it brief. Move to the point.\" or \"Stop. Answer directly.\" Never let a ramble finish."
        "\n3. CONTRADICTION TRAP: Track everything the candidate says. If a new answer contradicts "
        "an earlier one, pounce: \"Thirty seconds ago you said X. Now you say Y. Which is it?\""
        "\n4. STYLE: Cold, clipped, zero pleasantries. 1-2 sentences max per reply. "
        "Never encouraging during the interview. Never explain your reasoning."
        "\n5. LIE DETECTOR: If the candidate mentions ANY specific number, amount, company "
        "name, or concrete fact (e.g. '$40,000 savings', '3 backlogs', 'works at TCS'), "
        "REMEMBER it. Two or three questions later, ambush them: 'Wait — you said $40,000 "
        "earlier. Give me the exact breakdown of how your sponsor arranged that.' "
        "Catch them if the story shifts even slightly."
        "\n6. CUT-OFF RULE: If a candidate rambles (you will be told when an answer ran too "
        "long), cut them off mid-flow: 'Stop. I don't need a story. Get to the point.' "
        "Then ask your next question. Never apologize for interrupting."
        "\n\nKeep every reply SHORT (1-2 sentences). You are speaking, not writing essays. "
        "Never output stage directions or bracketed notes — only the words you say aloud."
    )
    # Feature 1: this session's dedicated attack vector
    if attack_vector and attack_vector in ATTACK_VECTORS:
        prompt += (
            "\n\nTHIS SESSION'S ATTACK VECTOR: " + ATTACK_VECTORS[attack_vector] +
            " Make this the centerpiece of your interrogation."
        )
    return prompt


async def groq_reply(history: list[dict], system_prompt: str) -> str:
    """Call Groq chat completions (OpenAI-compatible). Falls back to older model."""
    messages = [{"role": "system", "content": system_prompt}] + history
    last_err = "unknown"
    for model in (GROQ_MODEL, GROQ_MODEL_FALLBACK):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    GROQ_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 150,  # short, clipped examiner replies
                        "temperature": 0.8,
                    },
                )
            if r.status_code == 200:
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            log.warning("Groq model %s failed: %s", model, last_err)
        except Exception as e:
            last_err = str(e)[:200]
            log.warning("Groq model %s error: %s", model, last_err)
    raise RuntimeError(f"Groq unavailable ({last_err})")


async def deepgram_tts_stream(text: str, voice_mode: str = "strict"):
    """Deepgram TTS as an async generator yielding PCM16 chunks as they're
    generated — first audio arrives in ~300ms instead of waiting for the full
    file (1-3s for typical responses). Retries once on transient failure so a
    single Deepgram hiccup doesn't leave the user in silence."""
    url = DG_TTS_FRIENDLY_URL if voice_mode == "friendly" else DG_TTS_STRICT_URL
    last_err = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "Authorization": f"Token {DEEPGRAM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"text": text},
                ) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        raise RuntimeError(f"Deepgram TTS HTTP {r.status_code}: {body[:200]}")
                    yielded = False
                    async for chunk in r.aiter_bytes(chunk_size=TTS_CHUNK):
                        if chunk:
                            yielded = True
                            yield chunk
                    if yielded:
                        return  # success — don't retry
                    last_err = "empty audio stream"
        except Exception as e:
            last_err = e
            log.warning("Deepgram TTS attempt %d failed: %s", attempt + 1, str(e)[:150])
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Deepgram TTS failed after retry: {str(last_err)[:150]}")


async def deepgram_tts(text: str, voice_mode: str = "strict") -> bytes:
    """Deepgram TTS -> raw PCM16 24kHz mono bytes (non-streaming fallback)."""
    chunks = []
    async for chunk in deepgram_tts_stream(text, voice_mode):
        chunks.append(chunk)
    return b"".join(chunks)


async def connect_deepgram_stt():
    """Open Deepgram streaming STT websocket. Works across websockets 13/14 header kwarg rename."""
    try:
        return await websockets.connect(
            DG_STT_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            max_size=10 * 1024 * 1024,
        )
    except TypeError:
        # websockets < 14 fallback
        return await websockets.connect(
            DG_STT_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            max_size=10 * 1024 * 1024,
        )


async def voice_proxy(ws: WebSocket, session_id: str):
    sess = db.get_session(session_id)
    if not sess:
        # Session not found (e.g. in-memory cleared after restart).
        # Create a temporary session so voice still works instead of closing.
        log.warning("voice_proxy: session %s not found, creating temp session", session_id)
        try:
            sid = db.create_session(
                user_id="temp",
                test_type="general",
                plan="free_trial",
                max_duration_sec=120,
            )
            # Use the newly created session id for this connection
            sess = db.get_session(sid)
            session_id = sid
        except Exception as e:
            log.error("voice_proxy: failed to create temp session: %s", e)
            await ws.close(code=4404, reason="unknown session")
            return
    if not sess:
        await ws.close(code=4404, reason="unknown session")
        return
    max_dur = int(sess.get("max_duration_sec", 120))
    test_type = sess.get("test_type", "general")
    background = sess.get("background_summary")
    user_id = str(sess.get("user_id", "temp"))

    await ws.accept()
    await ws.send_json({"type": "status", "state": "connecting"})

    if not GROQ_API_KEY or not DEEPGRAM_API_KEY:
        await ws.send_json({"type": "error", "message": "Voice engine not configured"})
        await ws.close()
        return

    # Feature 1: pick this session's attack vector (excludes last session's)
    attack_vector = pick_attack_vector(user_id)
    system_prompt = build_system_prompt(test_type, background, attack_vector)
    transcript: list[dict] = list(sess.get("transcript", []))
    history: list[dict] = []  # groq chat history, capped at 20 entries
    # Feature 2/3/4 session analytics
    filler_count = 0
    word_count = 0
    facts_mentioned: list[str] = []
    over_explain_count = 0
    turn_audio_start: float | None = None  # Feature 4: answer-duration tracking
    start_ts = time.time()
    closed = asyncio.Event()
    barge = asyncio.Event()       # set on barge-in; checked during TTS streaming
    flush_req = asyncio.Event()   # set on user_done; flushes pending utterance
    user_turns: asyncio.Queue = asyncio.Queue()
    seq = 0                       # generation counter — invalidates stale turns
    warned_webm = False
    last_audio_ts = time.time()

    async def safe_send(obj: dict):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    # ---- Deepgram STT ----
    try:
        dg = await connect_deepgram_stt()
    except Exception as e:
        log.error("Deepgram STT connect failed: %s", e)
        await safe_send({"type": "error", "message": "Voice engine unavailable"})
        await ws.close()
        return
    log.info("Deepgram STT connected for session %s", session_id)

    # FIX 2026-10-12: Indian name STT corrections. Deepgram nova-3 rejects
    # the keyterm/keywords query param (HTTP 400), so we do post-processing
    # instead. Maps common mishearings to correct spellings.
    NAME_CORRECTIONS = {
        "karzan": "Farzan", "harzan": "Farzan", "farzan": "Farzan",
        "arjun": "Arjun", "arjan": "Arjun",
        "priya": "Priya", "preya": "Priya",
        "rahul": "Rahul", "rahul": "Rahul",
        "sneha": "Sneha", "snehaa": "Sneha",
        "vikram": "Vikram", "vikram": "Vikram",
        "ananya": "Ananya", "anania": "Ananya",
        "rohan": "Rohan", "rohen": "Rohan",
        "amit": "Amit", "ameet": "Amit",
        "kiran": "Kiran", "keeran": "Kiran",
        "divya": "Divya", "divia": "Divya",
        "sanjay": "Sanjay", "sanjai": "Sanjay",
        "meera": "Meera", "mira": "Meera",
        "kabir": "Kabir", "kabeer": "Kabir",
        "ishaan": "Ishaan", "ishan": "Ishaan",
        "neha": "Neha", "nehaa": "Neha",
        "aditya": "Aditya", "aditia": "Aditya",
        "pooja": "Pooja", "puja": "Pooja",
        "suresh": "Suresh", "sures": "Suresh",
        "kavya": "Kavya", "kaviya": "Kavya",
    }

    def correct_names(text: str) -> str:
        """Replace misheard Indian names with correct spellings."""
        import re
        def repl(m):
            word = m.group(0)
            lower = word.lower()
            if lower in NAME_CORRECTIONS:
                correct = NAME_CORRECTIONS[lower]
                # Preserve original capitalization
                if word[0].isupper():
                    return correct
                return correct.lower()
            return word
        return re.sub(r'\b[A-Za-z]+\b', repl, text)

    def flush_utterance(buf: list):
        nonlocal turn_audio_start
        text = " ".join(buf).strip()
        buf.clear()
        # Apply Indian name corrections before processing
        text = correct_names(text)
        # Feature 4: how long did the candidate talk? (>20s = over-explaining)
        duration = (time.time() - turn_audio_start) if turn_audio_start else 0.0
        turn_audio_start = None
        if text:
            if duration > OVEREXPLAIN_SECONDS:
                user_turns.put_nowait(("overexplain", (text, duration)))
            else:
                user_turns.put_nowait(("user", text))

    # ---- Deepgram -> thinker queue ----
    async def dg_listener():
        buf: list[str] = []
        last_interim: str = ""  # FIX 2026-10-13: fallback if finals never arrive
        flush_deadline: float = 0  # when to give up waiting for finals
        try:
            while not closed.is_set():
                try:
                    raw = await asyncio.wait_for(dg.recv(), timeout=0.3)
                except asyncio.TimeoutError:
                    if flush_req.is_set():
                        # FIX 2026-10-13: don't give up after 0.3s. Deepgram
                        # needs time to finalize after user_done. Wait up to
                        # 2.5s for finals, then fall back to interim text.
                        now = time.time()
                        if flush_deadline == 0:
                            flush_deadline = now + 2.5
                        if now >= flush_deadline:
                            flush_req.clear()
                            flush_deadline = 0
                            if buf:
                                flush_utterance(buf)
                            elif last_interim:
                                # Use interim as fallback — better than "not heard"
                                flush_utterance([last_interim])
                            else:
                                user_turns.put_nowait(("not_heard", ""))
                    continue
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("type") != "Results":
                    continue  # ignore Metadata / UtteranceEnd / SpeechStarted
                alt = (msg.get("channel") or {}).get("alternatives", [{}])[0]
                text = (alt.get("transcript") or "").strip()
                if text and msg.get("is_final"):
                    buf.append(text)
                elif text:
                    # Track latest interim for fallback
                    last_interim = text
                if msg.get("speech_final") or flush_req.is_set():
                    # FIX 2026-10-12: "no response" bug — old code had `and buf`,
                    # so if Deepgram heard nothing (empty buf), user_done did
                    # NOTHING and the interview hung forever. Now we always
                    # clear flush_req and let flush_utterance handle empty.
                    # FIX 2026-10-13: don't flush immediately on flush_req —
                    # Deepgram may still be finalizing. The timeout path above
                    # handles the 2.5s wait. Only flush here for speech_final.
                    if msg.get("speech_final"):
                        was_flush_req = flush_req.is_set()
                        flush_req.clear()
                        flush_deadline = 0
                        if buf:
                            flush_utterance(buf)
                        elif was_flush_req and last_interim:
                            flush_utterance([last_interim])
                        elif was_flush_req:
                            # User tapped stop but STT heard nothing — tell them.
                            log.warning("session %s: user_done but no transcript", session_id)
                            # Put a sentinel so the thinker responds with "didn't hear"
                            user_turns.put_nowait(("not_heard", ""))
                    # If only flush_req (not speech_final), let the timeout
                    # path handle it with the 2.5s grace period.
        except websockets.exceptions.ConnectionClosed:
            log.warning("Deepgram STT connection closed for %s", session_id)
        except Exception as e:
            log.exception("dg_listener crashed: %s", e)
        finally:
            closed.set()

    # ---- frontend -> Deepgram (+ control) ----
    async def client_reader():
        nonlocal warned_webm, last_audio_ts, seq, turn_audio_start
        try:
            while not closed.is_set():
                msg = await ws.receive()
                if closed.is_set():
                    break
                if "bytes" in msg and msg["bytes"]:
                    data = msg["bytes"]
                    if not warned_webm and data[:4] == WEBM_MAGIC:
                        warned_webm = True
                        log.warning(
                            "session %s: received WebM audio, expected raw PCM16 16kHz — "
                            "STT will fail until frontend sends PCM16", session_id)
                    try:
                        await dg.send(data)
                    except Exception:
                        break
                    # Feature 4: mark when the candidate starts talking
                    if turn_audio_start is None:
                        turn_audio_start = time.time()
                    last_audio_ts = time.time()
                elif "text" in msg and msg["text"]:
                    try:
                        ctl = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    t = ctl.get("type")
                    if t == "ping":
                        # Keepalive: client pings every ~20s to keep Render's proxy
                        # from killing idle connections. Respond so the client
                        # knows the path is alive.
                        await safe_send({"type": "pong"})
                    elif t == "interrupt":
                        # barge-in: invalidate current turn, stop TTS streaming
                        seq += 1
                        barge.set()
                        await safe_send({"type": "status", "state": "listening"})
                        await safe_send({"type": "interrupted"})
                    elif t == "user_done":
                        flush_req.set()
                        # FIX 2026-10-13: tell Deepgram to finalize immediately.
                        # Without this, Deepgram waits for the full endpointing
                        # window (1.2s) before emitting finals, but our flush
                        # logic gives up after 0.3s → "not_heard" → 3x repeat.
                        try:
                            await dg.send(json.dumps({"type": "Finalize"}))
                        except Exception:
                            pass  # dg may be closed; flush logic handles it
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("client_reader crashed: %s", e)
        finally:
            closed.set()

    # ---- Deepgram keepalive (prevents idle disconnect during silences) ----
    async def keepalive():
        try:
            while not closed.is_set():
                await asyncio.sleep(5)
                if closed.is_set():
                    break
                if time.time() - last_audio_ts > 8:
                    try:
                        await dg.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

    # ---- user turn -> Groq -> TTS -> frontend ----
    async def thinker():
        nonlocal seq, filler_count, word_count, over_explain_count
        try:
            while not closed.is_set():
                kind, payload = await user_turns.get()
                if closed.is_set():
                    break
                seq += 1
                my_seq = seq
                # CRITICAL FIX: barge must only cancel the CURRENT turn's TTS.
                # Clear it at turn start, or one interrupt wedges the session
                # silent forever ("sometimes no response").
                barge.clear()

                if kind == "not_heard":
                    # FIX 2026-10-12: STT heard nothing — tell user to speak up
                    # instead of hanging silently ("response nahi aa raha").
                    await safe_send({"type": "status", "state": "listening"})
                    reply = "I didn't catch that. Please speak louder and clearer."
                    await safe_send({"type": "status", "state": "speaking"})
                    await safe_send({"type": "transcript_examiner", "text": reply})
                    try:
                        async for audio_chunk in deepgram_tts_stream(reply, voice_mode="strict"):
                            if barge.is_set() or my_seq != seq or closed.is_set():
                                break
                            try:
                                await ws.send_bytes(audio_chunk)
                            except Exception:
                                break
                    except Exception as e:
                        log.error("TTS failed for not_heard: %s", e)
                    await safe_send({"type": "status", "state": "listening"})
                    continue

                if kind == "user":
                    # UX win: echo the heard transcript + distinct "heard" state
                    # IMMEDIATELY, before Groq starts — kills the long "thinking" confusion.
                    await safe_send({"type": "transcript_user", "text": payload})
                    await safe_send({"type": "status", "state": "heard"})
                    transcript.append({"role": "user", "text": payload})
                    history.append({"role": "user", "content": payload})
                    # Feature 2: hesitation tracking
                    filler_count += count_fillers(payload)
                    word_count += len(payload.split())
                    # Feature 3: remember concrete facts for cross-questioning
                    for f in extract_facts(payload):
                        if f.lower() not in {x.lower() for x in facts_mentioned}:
                            facts_mentioned.append(f)
                    sys_extra = ""
                    # Feature 3: feed remembered facts into the officer's context
                    if facts_mentioned:
                        sys_extra += (
                            "\n\n[FACTS THE CANDIDATE MENTIONED EARLIER — ambush one soon: "
                            + "; ".join(facts_mentioned[-5:]) + "]"
                        )
                elif kind == "overexplain":
                    # Feature 4: the candidate rambled past the limit.
                    text, duration = payload
                    over_explain_count += 1
                    transcript.append({"role": "user", "text": text})
                    history.append({"role": "user", "content": text})
                    filler_count += count_fillers(text)
                    word_count += len(text.split())
                    for f in extract_facts(text):
                        if f.lower() not in {x.lower() for x in facts_mentioned}:
                            facts_mentioned.append(f)
                    log.info("session %s: over-explain #%d (%.1fs)",
                             session_id, over_explain_count, duration)
                    if over_explain_count >= OVEREXPLAIN_LIMIT:
                        # INSTANT CUT-OFF: 3 strikes — interview over, visa denied.
                        reason = "Over-explaining / Vague Answers"
                        try:
                            db.mark_session_denied(session_id, reason)
                            db.update_session_analytics(session_id, {
                                "nervousness_score": nervousness_from_counts(
                                    filler_count, word_count),
                                "filler_count": filler_count,
                                "word_count": word_count,
                                "facts_mentioned": facts_mentioned,
                                "over_explain_count": over_explain_count,
                            })
                        except Exception:
                            pass
                        await safe_send({"type": "visa_denied", "reason": reason,
                                         "over_explain_count": over_explain_count})
                        await safe_send({"type": "session_ended"})
                        await safe_send({"type": "status", "state": "ended"})
                        closed.set()
                        continue
                    await safe_send({"type": "transcript_user", "text": text})
                    await safe_send({"type": "status", "state": "heard"})
                    await safe_send({"type": "overexplain_cutoff",
                                     "count": over_explain_count,
                                     "limit": OVEREXPLAIN_LIMIT})
                    sys_extra = (
                        f"\n\n[The candidate just rambled for {duration:.0f} seconds "
                        f"(cut-off #{over_explain_count} of {OVEREXPLAIN_LIMIT}). "
                        "CUT THEM OFF aggressively: 'Stop. I don't need a story. "
                        "Get to the point.' Then ask your next question sharply.]"
                    )
                elif kind == "opening":
                    sys_extra = ("\n\n[The candidate just connected. Deliver your opening line now — "
                                 "demand their full name, sternly. Keep it short.]")
                else:  # nudge (e.g. 60s warning)
                    sys_extra = f"\n\n[SYSTEM NOTICE: {payload}]"

                await safe_send({"type": "status", "state": "thinking"})
                try:
                    reply = await groq_reply(history if kind == "user" else list(history),
                                             system_prompt + sys_extra)
                except Exception as e:
                    log.error("Groq failed for %s: %s", session_id, e)
                    await safe_send({"type": "error",
                                     "message": "Examiner lost train of thought — keep going"})
                    await safe_send({"type": "status", "state": "listening"})
                    continue

                reply = (reply or "").strip()
                if not reply and kind == "opening":
                    # Never leave the user in silence on connect — use a
                    # canned stern opener if the LLM returns nothing.
                    reply = "State your full name. Now."
                    log.warning("Groq empty on opening for %s — using fallback", session_id)
                if not reply or my_seq != seq or closed.is_set():
                    await safe_send({"type": "status", "state": "listening"})
                    continue  # interrupted while thinking, or empty

                transcript.append({"role": "examiner", "text": reply})
                history.append({"role": "assistant", "content": reply})
                del history[:-20]

                if my_seq != seq or closed.is_set() or barge.is_set():
                    await safe_send({"type": "status", "state": "listening"})
                    continue

                barge.clear()
                # LATENCY: send transcript + speaking status IMMEDIATELY, before
                # TTS — the UI updates instantly instead of waiting 1-3s for audio.
                await safe_send({"type": "status", "state": "speaking"})
                await safe_send({"type": "transcript_examiner", "text": reply})
                # Stream TTS audio chunks as Deepgram generates them — first
                # audio hits the frontend in ~300ms, not after the full file.
                try:
                    chunk_idx = 0
                    async for audio_chunk in deepgram_tts_stream(reply, voice_mode="strict"):
                        if barge.is_set() or my_seq != seq or closed.is_set():
                            break
                        try:
                            await ws.send_bytes(audio_chunk)
                        except Exception:
                            break
                        if chunk_idx >= 1:
                            await asyncio.sleep(0)  # yield after initial burst
                        chunk_idx += 1
                except Exception as e:
                    log.error("Deepgram TTS stream failed for %s: %s", session_id, e)
                    # text already delivered; audio skipped
                await safe_send({"type": "status", "state": "listening"})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("thinker crashed: %s", e)
        finally:
            closed.set()

    # ---- hard time cutoff ----
    async def enforcer():
        try:
            elapsed = time.time() - start_ts
            remaining = max(0.0, max_dur - elapsed)
            if remaining > 60:
                await asyncio.sleep(remaining - 60)
                if not closed.is_set():
                    await user_turns.put(("nudge",
                        "60 seconds remain. Say 'Last question.', then ask your final question."))
                    await asyncio.sleep(60)
            else:
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            return
        if not closed.is_set():
            # flush transcript NOW so /session/end (triggered by session_ended)
            # doesn't race the socket-close flush
            try:
                db.append_transcript(session_id, transcript)
                db.update_session_analytics(session_id, {
                    "nervousness_score": nervousness_from_counts(
                        filler_count, word_count),
                    "filler_count": filler_count,
                    "word_count": word_count,
                    "facts_mentioned": facts_mentioned,
                    "over_explain_count": over_explain_count,
                    "attack_vector": attack_vector,
                })
            except Exception as e:
                log.exception("timeup transcript flush failed: %s", e)
            await safe_send({"type": "timeup"})
            await safe_send({"type": "session_ended"})
            await safe_send({"type": "status", "state": "ended"})
            closed.set()

    # opening examiner line — no silence at session start
    await user_turns.put(("opening", None))
    await safe_send({"type": "status", "state": "connected",
                     "test_type": test_type, "max_duration_sec": max_dur})

    tasks = [
        asyncio.create_task(dg_listener()),
        asyncio.create_task(client_reader()),
        asyncio.create_task(keepalive()),
        asyncio.create_task(thinker()),
        asyncio.create_task(enforcer()),
    ]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in tasks:
        t.cancel()

    # flush transcript + analytics
    nervousness = nervousness_from_counts(filler_count, word_count)
    try:
        db.append_transcript(session_id, transcript)
        db.mark_session_ended(session_id, duration_sec=int(time.time() - start_ts))
        # Features 2/3/4: persist analytics for scoring + results UI
        db.update_session_analytics(session_id, {
            "nervousness_score": nervousness,
            "filler_count": filler_count,
            "word_count": word_count,
            "facts_mentioned": facts_mentioned,
            "over_explain_count": over_explain_count,
            "attack_vector": attack_vector,
        })
    except Exception as e:
        log.exception("transcript flush failed: %s", e)
    try:
        await dg.send(json.dumps({"type": "CloseStream"}))
    except Exception:
        pass
    try:
        await dg.close()
    except Exception:
        pass
    try:
        await ws.close()
    except Exception:
        pass
    log.info("voice session %s closed, %d transcript turns", session_id, len(transcript))

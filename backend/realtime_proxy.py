"""
Turn-based voice pipeline: Mini App <-> FastAPI <-> Deepgram STT <-> Groq (Groq) <-> Deepgram TTS.

Frontend protocol (over this WebSocket):
  client -> server : binary PCM16 16kHz mono audio chunks (mic)
  client -> server : {"type": "interrupt"}            (barge-in)
  client -> server : {"type": "user_done"}            (frontend stop-talk hint; flushes pending utterance)
  server -> client : binary PCM16 24kHz mono audio chunks (examiner voice)
  server -> client : {"type":"status","state":"..."}  (connected/connecting/listening/thinking/speaking/ended)
  server -> client : {"type":"transcript_examiner","text":"..."}  (compat: current frontend build)
  server -> client : {"type":"interrupted"}            (compat: current frontend build)
  server -> client : {"type":"timeup"}                (hard cutoff hit)
  server -> client : {"type":"session_ended"}         (compat: current frontend build)
  server -> client : {"type":"error","message":"..."}

Backend flow:
  1. Mic audio chunks -> Deepgram STT WebSocket (nova-2, endpointing 300ms)
  2. Final utterance transcript -> Groq chat completions (Groq, OpenAI-compatible REST)
  3. Examiner reply text -> Deepgram TTS (aura-asteria-en, linear16 24kHz)
  4. TTS audio streamed back to frontend in chunks; barge-in cancels mid-stream

API keys never leave this backend. Transcript accumulates here and is flushed
to Firestore when the socket closes (or the time limit fires).
"""
import asyncio
import json
import logging
import os
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
    "?model=nova-2&language=en&smart_format=true"
    "&interim_results=true&endpointing=300"
    "&encoding=linear16&sample_rate=16000&channels=1"
)
DG_TTS_URL = (
    "https://api.deepgram.com/v1/speak"
    "?model=aura-asteria-en&encoding=linear16&sample_rate=24000"
)
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

TTS_CHUNK = 8192  # ~0.17s of 24kHz 16-bit mono per chunk
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


def build_system_prompt(test_type: str, background_summary: str | None) -> str:
    prompt = PERSONA_INSTRUCTIONS
    prompt += "\n\nTEST FOCUS: " + TEST_TYPE_HINTS.get(test_type, TEST_TYPE_HINTS["general"])
    if background_summary:
        prompt += (
            "\n\nAPPLICANT BACKGROUND FORM SUMMARY (use for contradiction traps):\n"
            + background_summary.strip()
        )
    prompt += (
        "\n\nKeep every reply SHORT (1-3 sentences). You are speaking, not writing essays. "
        "Never output stage directions or bracketed notes — only the words you say aloud."
    )
    return prompt


async def groq_reply(history: list[dict], system_prompt: str) -> str:
    """Call Groq chat completions (OpenAI-compatible). Falls back to older model."""
    messages = [{"role": "system", "content": system_prompt}] + history
    last_err = "unknown"
    for model in (GROQ_MODEL, GROQ_MODEL_FALLBACK):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
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


async def deepgram_tts(text: str) -> bytes:
    """Deepgram TTS -> raw PCM16 24kHz mono bytes."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            DG_TTS_URL,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
    if r.status_code != 200:
        raise RuntimeError(f"Deepgram TTS HTTP {r.status_code}: {r.text[:200]}")
    return r.content


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
        await ws.close(code=4404, reason="unknown session")
        return
    max_dur = int(sess.get("max_duration_sec", 120))
    test_type = sess.get("test_type", "general")
    background = sess.get("background_summary")

    await ws.accept()
    await ws.send_json({"type": "status", "state": "connecting"})

    if not GROQ_API_KEY or not DEEPGRAM_API_KEY:
        await ws.send_json({"type": "error", "message": "Voice engine not configured"})
        await ws.close()
        return

    system_prompt = build_system_prompt(test_type, background)
    transcript: list[dict] = list(sess.get("transcript", []))
    history: list[dict] = []  # groq chat history, capped at 20 entries
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

    def flush_utterance(buf: list):
        text = " ".join(buf).strip()
        buf.clear()
        if text:
            user_turns.put_nowait(("user", text))

    # ---- Deepgram -> thinker queue ----
    async def dg_listener():
        buf: list[str] = []
        try:
            while not closed.is_set():
                try:
                    raw = await asyncio.wait_for(dg.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    if flush_req.is_set():
                        flush_req.clear()
                        flush_utterance(buf)
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
                if (msg.get("speech_final") or flush_req.is_set()) and buf:
                    flush_req.clear()
                    flush_utterance(buf)
        except websockets.exceptions.ConnectionClosed:
            log.warning("Deepgram STT connection closed for %s", session_id)
        except Exception as e:
            log.exception("dg_listener crashed: %s", e)
        finally:
            closed.set()

    # ---- frontend -> Deepgram (+ control) ----
    async def client_reader():
        nonlocal warned_webm, last_audio_ts, seq
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
                    last_audio_ts = time.time()
                elif "text" in msg and msg["text"]:
                    try:
                        ctl = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    t = ctl.get("type")
                    if t == "interrupt":
                        # barge-in: invalidate current turn, stop TTS streaming
                        seq += 1
                        barge.set()
                        await safe_send({"type": "status", "state": "listening"})
                        await safe_send({"type": "interrupted"})
                    elif t == "user_done":
                        flush_req.set()
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
        nonlocal seq
        try:
            while not closed.is_set():
                kind, payload = await user_turns.get()
                if closed.is_set():
                    break
                seq += 1
                my_seq = seq

                if kind == "user":
                    transcript.append({"role": "user", "text": payload})
                    history.append({"role": "user", "content": payload})
                    sys_extra = ""
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
                if not reply or my_seq != seq or closed.is_set():
                    continue  # interrupted while thinking, or empty

                transcript.append({"role": "examiner", "text": reply})
                history.append({"role": "assistant", "content": reply})
                del history[:-20]

                try:
                    audio = await deepgram_tts(reply)
                except Exception as e:
                    log.error("Deepgram TTS failed for %s: %s", session_id, e)
                    # text still delivered; audio skipped
                    await safe_send({"type": "transcript_examiner", "text": reply})
                    await safe_send({"type": "status", "state": "listening"})
                    continue

                if my_seq != seq or closed.is_set() or barge.is_set():
                    await safe_send({"type": "status", "state": "listening"})
                    continue

                barge.clear()
                await safe_send({"type": "status", "state": "speaking"})
                await safe_send({"type": "transcript_examiner", "text": reply})
                for i in range(0, len(audio), TTS_CHUNK):
                    if barge.is_set() or my_seq != seq or closed.is_set():
                        break
                    try:
                        await ws.send_bytes(audio[i:i + TTS_CHUNK])
                    except Exception:
                        break
                    await asyncio.sleep(0)  # yield between chunks
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

    # flush transcript
    try:
        db.append_transcript(session_id, transcript)
        db.mark_session_ended(session_id, duration_sec=int(time.time() - start_ts))
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

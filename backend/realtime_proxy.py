"""
Realtime voice proxy: Mini App <-> FastAPI <-> OpenAI Realtime API.

Frontend protocol (over this WebSocket):
  client -> server : binary PCM16 16kHz mono audio chunks (mic)
  client -> server : {"type": "interrupt"}            (barge-in)
  server -> client : binary PCM16 24kHz mono audio chunks (examiner voice)
  server -> client : {"type":"status","state":"..."}  (connected/listening/speaking/ended)
  server -> client : {"type":"timeup"}                (hard cutoff hit)
  server -> client : {"type":"error","message":"..."}

OpenAI key NEVER leaves this backend. We hold the server-side WS to OpenAI
and relay audio + transcripts. Transcript accumulates here and is flushed
to Firestore when the socket closes (or the time limit fires).
"""
import asyncio
import base64
import json
import logging
import os
import time

from fastapi import WebSocket, WebSocketDisconnect
import websockets

import firestore_db as db

log = logging.getLogger("englishtest.proxy")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")

# Load persona once at import
_PERSONA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PERSONA.md")
try:
    with open(_PERSONA_PATH, "r", encoding="utf-8") as f:
        PERSONA_INSTRUCTIONS = f.read()
except FileNotFoundError:
    PERSONA_INSTRUCTIONS = "You are a strict English oral examiner conducting an advanced speaking test. Be stern, interrupt rambling, probe vague answers for specifics."
    log.warning("PERSONA.md not found, using fallback instructions")


async def voice_proxy(ws: WebSocket, session_id: str):
    sess = db.get_session(session_id)
    if not sess:
        await ws.close(code=4404, reason="unknown session")
        return
    max_dur = int(sess.get("max_duration_sec", 120))
    test_type = sess.get("test_type", "general")

    await ws.accept()
    await ws.send_json({"type": "status", "state": "connecting"})

    if not OPENAI_API_KEY:
        await ws.send_json({"type": "error", "message": "Voice engine not configured"})
        await ws.close()
        return

    transcript: list[dict] = list(sess.get("transcript", []))
    start_ts = time.time()
    closed = asyncio.Event()

    # ---- connect to OpenAI Realtime ----
    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    try:
        oai = await websockets.connect(
            url,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            max_size=10 * 1024 * 1024,
        )
    except Exception as e:
        log.error("OpenAI WS connect failed: %s", e)
        await ws.send_json({"type": "error", "message": "Voice engine unavailable"})
        await ws.close()
        return

    async def send_session_update():
        await oai.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": PERSONA_INSTRUCTIONS,
                "voice": "onyx",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"},
                "max_response_output_tokens": 300,
            },
        }))

    # ---- OpenAI -> frontend ----
    async def oai_to_client():
        try:
            async for raw in oai:
                if closed.is_set():
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type", "")

                if t == "session.created":
                    await send_session_update()

                elif t == "conversation.item.input_audio_transcription.completed":
                    text = (msg.get("transcript") or "").strip()
                    if text:
                        transcript.append({"role": "user", "text": text})
                        await ws.send_json({"type": "status", "state": "listening"})

                elif t == "response.audio_transcript.done":
                    text = (msg.get("transcript") or "").strip()
                    if text:
                        transcript.append({"role": "examiner", "text": text})

                elif t == "response.audio.delta":
                    b64 = msg.get("delta", "")
                    if b64:
                        await ws.send_bytes(base64.b64decode(b64))
                        # best-effort speaking state (throttled by client anyway)
                        # we don't spam status messages per chunk

                elif t == "response.done":
                    await ws.send_json({"type": "status", "state": "listening"})

                elif t == "error":
                    log.error("OpenAI realtime error: %s", json.dumps(msg)[:400])
                    await ws.send_json({"type": "error",
                                        "message": "Voice engine hiccup — keep going"})
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            log.exception("oai_to_client crashed: %s", e)
        finally:
            closed.set()

    # ---- frontend -> OpenAI ----
    async def client_to_oai():
        try:
            while not closed.is_set():
                msg = await ws.receive()
                if closed.is_set():
                    break
                if "bytes" in msg and msg["bytes"]:
                    # mic audio chunk -> OpenAI
                    b64 = base64.b64encode(msg["bytes"]).decode()
                    await oai.send(json.dumps({
                        "type": "input_audio_buffer.append", "audio": b64}))
                elif "text" in msg and msg["text"]:
                    try:
                        ctl = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    if ctl.get("type") == "interrupt":
                        # barge-in: cancel current response
                        await oai.send(json.dumps({"type": "response.cancel"}))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("client_to_oai crashed: %s", e)
        finally:
            closed.set()

    # ---- hard time cutoff ----
    async def enforcer():
        elapsed = time.time() - start_ts
        remaining = max(0.0, max_dur - elapsed)
        # warn at 60s remaining (persona says "Last question")
        if remaining > 60:
            await asyncio.sleep(remaining - 60)
            if not closed.is_set():
                await oai.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": "[SYSTEM: 60 seconds left. Say 'Last question.', ask your final question.]"}]},
                }))
                await oai.send(json.dumps({"type": "response.create"}))
                await asyncio.sleep(60)
        else:
            await asyncio.sleep(remaining)
        if not closed.is_set():
            try:
                await ws.send_json({"type": "timeup"})
            except Exception:
                pass
            closed.set()

    await ws.send_json({"type": "status", "state": "connected",
                        "test_type": test_type, "max_duration_sec": max_dur})

    tasks = [
        asyncio.create_task(oai_to_client()),
        asyncio.create_task(client_to_oai()),
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
        await oai.close()
    except Exception:
        pass
    try:
        await ws.close()
    except Exception:
        pass
    log.info("voice session %s closed, %d transcript turns", session_id, len(transcript))

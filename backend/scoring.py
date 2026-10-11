"""
Transcript scoring via Grok (xAI), following the PERSONA.md rubric exactly.

Rubric (total 100):
  clarity     /25  — answer clarity & specificity
  ties        /25  — personal background depth
  financial   /20  — detail credibility
  consistency /15  — zero contradictions
  composure   /15  — composure under pressure

Verdict:
  >= 75          PASS
  60–74          NEEDS_WORK
  < 60           FAIL
"""
import json
import logging
import os

import httpx

log = logging.getLogger("englishtest.scoring")

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_SCORING_MODEL = os.environ.get("XAI_SCORING_MODEL", "grok-3")
XAI_FALLBACK_MODEL = os.environ.get("XAI_MODEL_FALLBACK", "grok-2-1212")
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"

RUBRIC = """You are a senior English speaking-test evaluator scoring a PRACTICE strict English speaking test transcript.

Score these five dimensions. Be brutal and specific — inflated scores hurt the test taker.

1. CLARITY (0-25): Vague/generic answers = near zero. Concrete names, numbers, dates, specific reasons = full marks.
2. BACKGROUND DEPTH (0-25): Concrete, verifiable personal details (studies, work, plans, life situation) — not just claims. Nothing concrete = near zero.
3. DETAIL CREDIBILITY (0-20): Facts, numbers, and reasons that hold up under probing. Fuzzy, evasive, or suspicious details = low.
4. CONSISTENCY (0-15): Zero contradictions = full marks. Caught contradictions or logical holes = heavy deductions.
5. COMPOSURE (0-15): Calm and direct under interruption/pressure = full marks. Rambling, freezing, evasiveness, arguing = deductions.

VERDICT RULES (apply mechanically from total):
- total >= 75  -> "PASS"
- 60 <= total <= 74 -> "NEEDS_WORK"
- total < 60  -> "FAIL"

Return ONLY valid JSON, no markdown fences, exactly this shape:
{"clarity": <0-25>, "ties": <0-25>, "financial": <0-20>, "consistency": <0-15>, "composure": <0-15>, "total": <0-100>, "verdict": "PASS"|"NEEDS_WORK"|"FAIL", "weaknesses": ["<specific weakness 1>", "<specific weakness 2>"], "feedback": "<2-4 sentences of brutally honest coaching: what failed and exactly how to fix it>"}"""


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        role = "TEST_TAKER" if turn.get("role") == "user" else "EXAMINER"
        lines.append(f"{role}: {turn.get('text', '')}")
    return "\n".join(lines)


def _clamp_result(r: dict) -> dict:
    bounds = {"clarity": 25, "ties": 25, "financial": 20,
              "consistency": 15, "composure": 15}
    for k, mx in bounds.items():
        try:
            r[k] = max(0, min(mx, int(r.get(k, 0))))
        except (TypeError, ValueError):
            r[k] = 0
    r["total"] = sum(r[k] for k in bounds)
    t = r["total"]
    r["verdict"] = "PASS" if t >= 75 else ("NEEDS_WORK" if t >= 60 else "FAIL")
    if not isinstance(r.get("weaknesses"), list):
        r["weaknesses"] = []
    r["weaknesses"] = [str(w)[:300] for w in r["weaknesses"][:5]]
    r["feedback"] = str(r.get("feedback", ""))[:1500]
    return r


async def score_transcript(transcript: list[dict], test_type: str = "general") -> dict:
    """Score a session transcript. Returns the rubric JSON dict."""
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY not configured")

    convo = _format_transcript(transcript)
    payload = {
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content":
             f"Test type: {test_type}\n\nTRANSCRIPT:\n{convo}\n\nScore it."},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
    }

    raw = "{}"
    last_err = "unknown"
    for model in (XAI_SCORING_MODEL, XAI_FALLBACK_MODEL):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    XAI_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {XAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, **payload},
                )
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"] or "{}"
                break
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            log.warning("Grok scoring model %s failed: %s", model, last_err)
        except Exception as e:
            last_err = str(e)[:200]
            log.warning("Grok scoring model %s error: %s", model, last_err)
    else:
        raise RuntimeError(f"Scoring engine unavailable ({last_err})")

    # strip accidental markdown fences before parsing
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        log.error("Scoring returned non-JSON: %s", raw[:300])
        raise RuntimeError("Scoring engine returned invalid output")
    return _clamp_result(result)

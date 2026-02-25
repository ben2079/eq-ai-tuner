#!/usr/bin/env python3
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _coerce_20(values: List[float], lo: float, hi: float) -> List[float]:
    out = [0.0] * 20
    for i in range(min(20, len(values))):
        out[i] = _clamp(float(values[i]), lo, hi)
    return out


def _extract_json_text(response_payload: Dict[str, object]) -> Dict[str, object]:
    content = response_payload.get("content", [])
    if not isinstance(content, list):
        raise ValueError("Invalid Claude response: content missing")

    chunks = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                chunks.append(text)

    if not chunks:
        raise ValueError("Invalid Claude response: no text content")

    text = "\n".join(chunks).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Claude response does not contain JSON object")

    return json.loads(text[start : end + 1])


def suggest_with_claude(
    bands_hz: List[float],
    levels_l_db: List[float],
    levels_r_db: List[float],
    style: str,
    max_boost: float,
    max_cut: float,
    history: List[Dict[str, object]],
    timeout_seconds: float = 6.0,
) -> Dict[str, object]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model_name = os.environ.get("CLAUDE_EQ_MODEL", "claude-3-5-sonnet-latest").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    max_cut = float(max_cut)
    max_boost = float(max_boost)

    system_prompt = (
        "You are an audio EQ assistant. "
        "Output only strict JSON and no markdown. "
        "JSON schema: {\"gains_l_db\": number[20], \"gains_r_db\": number[20], \"confidence\": number, \"explanation\": string}. "
        "All gains must be between -max_cut and max_boost. "
        "Prefer smooth neighboring band transitions and avoid abrupt spikes."
    )

    req_payload = {
        "model": model_name,
        "max_tokens": 800,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "task": "Propose stereo 20-band EQ gains",
                                "style": style,
                                "constraints": {"max_boost": max_boost, "max_cut": max_cut},
                                "bands_hz": bands_hz,
                                "levels_l_db": levels_l_db,
                                "levels_r_db": levels_r_db,
                                "recent_history": history[-8:],
                            },
                            separators=(",", ":"),
                        ),
                    }
                ],
            }
        ],
    }

    body = json.dumps(req_payload).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_txt = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Claude HTTP {exc.code}: {err_txt}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Claude connection error: {exc}") from exc

    payload = json.loads(raw)
    parsed = _extract_json_text(payload)

    gains_l = parsed.get("gains_l_db")
    gains_r = parsed.get("gains_r_db")
    confidence = float(parsed.get("confidence", 0.0))

    if not isinstance(gains_l, list) or not isinstance(gains_r, list):
        raise ValueError("Claude JSON missing gains arrays")

    gains_l_20 = _coerce_20(gains_l, -max_cut, max_boost)
    gains_r_20 = _coerce_20(gains_r, -max_cut, max_boost)

    return {
        "gains_l_db": gains_l_20,
        "gains_r_db": gains_r_20,
        "confidence": _clamp(confidence, 0.0, 1.0),
        "explanation": str(parsed.get("explanation", ""))[:400],
        "model": model_name,
    }

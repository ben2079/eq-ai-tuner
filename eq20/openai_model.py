#!/usr/bin/env python3
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _coerce_20(values: List[float], lo: float, hi: float) -> List[float]:
    out = [0.0] * 20
    for i in range(min(20, len(values))):
        out[i] = _clamp(float(values[i]), lo, hi)
    return out


def suggest_with_openai(
    bands_hz: List[float],
    levels_l_db: List[float],
    levels_r_db: List[float],
    style: str,
    max_boost: float,
    max_cut: float,
    history: List[Dict[str, object]],
    timeout_seconds: float = 6.0,
) -> Dict[str, object]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("OPENAI_EQ_MODEL", "gpt-4.1-mini").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    max_cut = float(max_cut)
    max_boost = float(max_boost)

    schema_hint = {
        "gains_l_db": [0.0] * 20,
        "gains_r_db": [0.0] * 20,
        "confidence": 0.5,
        "explanation": "short string",
    }

    system_prompt = (
        "You are an audio EQ assistant for stereo 20-band equalizer tuning. "
        "Return only strict JSON with exactly this shape: "
        + json.dumps(schema_hint, separators=(",", ":"))
        + ". No markdown, no prose outside JSON. "
        "Keep neighboring bands smooth and avoid spikes. "
        "All gain values must stay between -max_cut and max_boost."
    )

    user_payload = {
        "task": "Propose stereo 20-band EQ gains",
        "style": style,
        "constraints": {"max_boost": max_boost, "max_cut": max_cut},
        "bands_hz": bands_hz,
        "levels_l_db": levels_l_db,
        "levels_r_db": levels_r_db,
        "recent_history": history[-8:],
    }

    req_payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
        ],
    }

    body = json.dumps(req_payload).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_API_URL,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_txt = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {err_txt}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI connection error: {exc}") from exc

    payload = json.loads(raw)
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")

    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI response missing message content")

    parsed = json.loads(content)
    gains_l = parsed.get("gains_l_db")
    gains_r = parsed.get("gains_r_db")
    confidence = float(parsed.get("confidence", 0.0))

    if not isinstance(gains_l, list) or not isinstance(gains_r, list):
        raise ValueError("OpenAI JSON missing gains arrays")

    gains_l_20 = _coerce_20(gains_l, -max_cut, max_boost)
    gains_r_20 = _coerce_20(gains_r, -max_cut, max_boost)

    return {
        "gains_l_db": gains_l_20,
        "gains_r_db": gains_r_20,
        "confidence": _clamp(confidence, 0.0, 1.0),
        "explanation": str(parsed.get("explanation", ""))[:400],
        "model": model_name,
    }

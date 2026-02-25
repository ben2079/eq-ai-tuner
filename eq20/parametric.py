#!/usr/bin/env python3
"""Parametric EQ AI suggestions via OpenAI."""
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

VALID_TYPES = {"Bell", "Low Shelf", "High Shelf", "Notch", "High Pass", "Low Pass"}

SYSTEM_PROMPT = """\
You are a professional mastering engineer controlling a parametric EQ.
You analyze per-band RMS levels (dBFS) and return precise parametric EQ corrections.

Rules:
- Use at most 8 bands per channel. Only add a band if it genuinely helps.
- Each band: freq (Hz, 20-20000), gain (dB, negative values only unless boosting presence/air),
  q (0.3-10, higher=narrower), type: Bell | Low Shelf | High Shelf | Notch | High Pass | Low Pass.
- Low Shelf: use for bass tilt (freq 80-250 Hz). High Shelf: use for air (freq 6k-16k Hz).
- Bell: surgical corrections. Notch: resonance removal.
- Total loudness change should be near zero (sum of gains close to 0).
- Respond ONLY with valid JSON, no markdown.
"""

USER_TEMPLATE = """\
Style: {style}
Max boost: {max_boost} dB | Max cut: {max_cut} dB

Left channel RMS per band (Hz: dBFS):
{levels_l}

Right channel RMS per band (Hz: dBFS):
{levels_r}

Return JSON:
{{
  "bands_l": [{{"freq": 200, "gain": -3.0, "q": 1.4, "type": "Bell"}}, ...],
  "bands_r": [...]
}}
"""


@dataclass
class ParamBand:
    freq: float
    gain: float
    q: float = 1.4
    type: str = "Bell"


@dataclass
class ParamResult:
    bands_l: List[ParamBand] = field(default_factory=list)
    bands_r: List[ParamBand] = field(default_factory=list)
    model: str = ""
    error: str = ""


def _format_levels(bands_hz: List[float], levels_db: List[float]) -> str:
    return ", ".join(f"{int(hz)}Hz:{db:.1f}" for hz, db in zip(bands_hz, levels_db))


def _parse_band(d: dict, max_boost: float, max_cut: float) -> Optional[ParamBand]:
    try:
        freq = float(d["freq"])
        gain = float(d["gain"])
        q = float(d.get("q", 1.4))
        btype = str(d.get("type", "Bell")).strip()
        if btype not in VALID_TYPES:
            btype = "Bell"
        freq = max(20.0, min(20000.0, freq))
        gain = max(-abs(max_cut), min(abs(max_boost), gain))
        q = max(0.3, min(10.0, q))
        return ParamBand(freq=freq, gain=gain, q=q, type=btype)
    except (KeyError, ValueError, TypeError):
        return None


def suggest_parametric_rule_based(
    levels_l: List[float],
    levels_r: List[float],
    bands_hz: List[float],
    style: str = "balanced",
    max_boost: float = 6.0,
    max_cut: float = 6.0,
) -> ParamResult:
    """Rule-based parametric EQ — no API required. Finds loudest/quietest regions
    and places up to 6 Bell/Shelf bands to flatten the spectrum."""
    import math

    def _bands_for_channel(levels: List[float]) -> List[ParamBand]:
        if not levels:
            return []
        # Reference: median level
        sorted_l = sorted(levels)
        median = sorted_l[len(sorted_l) // 2]
        bands: List[ParamBand] = []

        # Low Shelf: if bass region (< 200 Hz) is significantly different
        bass_hz = [hz for hz in bands_hz if hz <= 200]
        bass_idxs = [i for i, hz in enumerate(bands_hz) if hz <= 200]
        if bass_idxs:
            bass_avg = sum(levels[i] for i in bass_idxs) / len(bass_idxs)
            bass_diff = median - bass_avg  # positive = bass too quiet → boost
            bass_diff = max(-abs(max_cut), min(abs(max_boost), bass_diff * 0.6))
            if abs(bass_diff) >= 1.0:
                bands.append(ParamBand(freq=120.0, gain=round(bass_diff, 1),
                                       q=0.7, type="Low Shelf"))

        # High Shelf: if treble region (> 8 kHz) is significantly different
        hi_idxs = [i for i, hz in enumerate(bands_hz) if hz >= 8000]
        if hi_idxs:
            hi_avg = sum(levels[i] for i in hi_idxs) / len(hi_idxs)
            hi_diff = median - hi_avg
            hi_diff = max(-abs(max_cut), min(abs(max_boost), hi_diff * 0.5))
            if abs(hi_diff) >= 1.0:
                bands.append(ParamBand(freq=10000.0, gain=round(hi_diff, 1),
                                       q=0.7, type="High Shelf"))

        # Bell bands for mid outliers (200 Hz – 8 kHz)
        mid_idxs = [i for i, hz in enumerate(bands_hz) if 200 < hz < 8000]
        if mid_idxs:
            mid_levels = [(levels[i], i) for i in mid_idxs]
            mid_levels.sort(key=lambda x: abs(x[0] - median), reverse=True)
            added = 0
            for lvl, idx in mid_levels:
                if added >= 4:
                    break
                diff = median - lvl  # positive → boost (too quiet), negative → cut
                diff = max(-abs(max_cut), min(abs(max_boost), diff * 0.7))
                if abs(diff) < 1.5:
                    continue
                hz = bands_hz[idx]
                # Use narrower Q for cuts, wider for boosts
                q = 2.0 if diff < 0 else 1.2
                bands.append(ParamBand(freq=round(hz, 1), gain=round(diff, 1), q=q, type="Bell"))
                added += 1

        return bands

    return ParamResult(
        bands_l=_bands_for_channel(levels_l),
        bands_r=_bands_for_channel(levels_r),
        model="rule-based",
    )


def _call_openai(key: str, model: str, payload: dict, timeout: float) -> dict:
    """Single API call, raises HTTPError on failure."""
    payload = dict(payload, model=model)
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def suggest_parametric_openai(
    levels_l: List[float],
    levels_r: List[float],
    bands_hz: List[float],
    style: str = "balanced",
    max_boost: float = 6.0,
    max_cut: float = 6.0,
    timeout: float = 30.0,
) -> ParamResult:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    primary_model = os.environ.get("OPENAI_EQ_MODEL", "gpt-4.1-mini").strip()
    # Fallback models tried in order if primary is rate-limited
    models_to_try = [primary_model]
    if primary_model != "gpt-4o-mini":
        models_to_try.append("gpt-4o-mini")
    if primary_model != "gpt-3.5-turbo":
        models_to_try.append("gpt-3.5-turbo")

    if not key:
        return ParamResult(error="OPENAI_API_KEY missing")

    user_msg = USER_TEMPLATE.format(
        style=style,
        max_boost=max_boost,
        max_cut=max_cut,
        levels_l=_format_levels(bands_hz, levels_l),
        levels_r=_format_levels(bands_hz, levels_r),
    )
    payload = {
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }

    last_exc: Optional[Exception] = None
    for model in models_to_try:
        for attempt in range(2):
            try:
                data = _call_openai(key, model, payload, timeout)
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                result = ParamResult(model=model)
                for raw in parsed.get("bands_l", [])[:8]:
                    b = _parse_band(raw, max_boost, max_cut)
                    if b:
                        result.bands_l.append(b)
                for raw in parsed.get("bands_r", [])[:8]:
                    b = _parse_band(raw, max_boost, max_cut)
                    if b:
                        result.bands_r.append(b)
                return result
            except HTTPError as exc:
                last_exc = exc
                if exc.code == 429:
                    retry_after = int(exc.headers.get("retry-after", 0) or 0)
                    wait = retry_after if retry_after > 0 else (2 ** attempt * 3)
                    time.sleep(wait)
                    continue
                return ParamResult(error=f"HTTP {exc.code}: {exc.reason}", model=model)
            except Exception as exc:
                return ParamResult(error=str(exc), model=model)
        # Both attempts for this model exhausted, try next model

    # All AI models failed — fall back to rule-based analysis
    return suggest_parametric_rule_based(
        levels_l, levels_r, bands_hz,
        style=style, max_boost=max_boost, max_cut=max_cut,
    )

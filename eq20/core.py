#!/usr/bin/env python3
import json
import math
import os
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .claude_model import suggest_with_claude
from .ml_model import load_model, make_features, predict_and_learn, save_model
from .openai_model import suggest_with_openai

BANDS_20_HZ = [
    31.25,
    44.2,
    62.5,
    88.4,
    125.0,
    176.8,
    250.0,
    353.6,
    500.0,
    707.1,
    1000.0,
    1414.2,
    2000.0,
    2828.4,
    4000.0,
    5656.9,
    8000.0,
    11313.7,
    16000.0,
    20000.0,
]

# Extended 20-band variant for high-res streams. Keeps 20 total bands,
# but allocates more detail above 16 kHz.
BANDS_20_HZ_HIGHRES = [
    31.25,
    44.2,
    62.5,
    88.4,
    125.0,
    176.8,
    250.0,
    353.6,
    500.0,
    707.1,
    1000.0,
    1414.2,
    2000.0,
    2828.4,
    4000.0,
    5656.9,
    8000.0,
    12000.0,
    20000.0,
    28000.0,
]


def select_bands_20(rate: int, high_res: bool) -> List[float]:
    # 26/28 kHz analysis is only valid when Nyquist is well above 28 kHz.
    # At 48 kHz stream rate, Nyquist is 24 kHz and high-res bands are not usable.
    if not high_res or rate < 96000:
        return list(BANDS_20_HZ)

    nyquist_guard = (rate * 0.5) * 0.96
    if nyquist_guard < 28000.0:
        return list(BANDS_20_HZ)
    return list(BANDS_20_HZ_HIGHRES)


@dataclass
class AnalysisResult:
    source: str
    timestamp: float
    bands_hz: List[float]
    levels_l_db: List[float]
    levels_r_db: List[float]
    gains_l_db: List[float]
    gains_r_db: List[float]
    target_l_db: float
    target_r_db: float
    mode: str
    style: str
    model_info: Optional[Dict[str, object]] = None
    rate: int = 48000
    high_res: bool = False


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def get_default_monitor_source() -> str:
    sink_res = run(["pactl", "get-default-sink"])
    if sink_res.returncode != 0:
        raise RuntimeError("Could not read default sink: " + sink_res.stderr.strip())
    sink = sink_res.stdout.strip()
    if not sink:
        raise RuntimeError("Empty sink name returned")
    return sink + ".monitor"


def record_raw_float32(source: str, seconds: int, rate: int, channels: int) -> str:
    raw_fd, raw_path = tempfile.mkstemp(prefix="eq20_", suffix=".raw")
    os.close(raw_fd)
    cmd = [
        "timeout",
        f"{seconds}s",
        "parec",
        "-d",
        source,
        "--format=float32le",
        f"--rate={rate}",
        f"--channels={channels}",
        "--raw",
    ]
    try:
        with open(raw_path, "wb") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    except Exception:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise

    if proc.returncode not in (0, 124):
        err = proc.stderr.strip()
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise RuntimeError(f"Capture failed ({proc.returncode}): {err}")
    return raw_path


def read_stereo_samples(path: str, channels: int) -> Tuple[List[float], List[float]]:
    data = open(path, "rb").read()
    frame_size = 4 * channels
    if len(data) < frame_size * 1024:
        raise RuntimeError("Not enough audio data captured")

    usable = len(data) - (len(data) % frame_size)
    data = data[:usable]

    left: List[float] = []
    right: List[float] = []
    if channels >= 2:
        for frame in struct.iter_unpack("<" + "f" * channels, data):
            left.append(frame[0])
            right.append(frame[1])
    else:
        for (mono,) in struct.iter_unpack("<f", data):
            left.append(mono)
            right.append(mono)
    return left, right


def hanning(n: int) -> List[float]:
    if n <= 1:
        return [1.0] * n
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def band_rms_goertzel(samples: List[float], rate: int, center_hz: float) -> float:
    n = len(samples)
    if n < 64:
        return 0.0

    k = int(round(center_hz * n / rate))
    k = max(1, min(k, n // 2 - 1))
    omega = 2.0 * math.pi * k / n
    coeff = 2.0 * math.cos(omega)

    win = hanning(n)
    s_prev = 0.0
    s_prev2 = 0.0
    for x, w in zip(samples, win):
        s = (x * w) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return math.sqrt(max(power, 0.0) / n)


def dbfs(value: float, floor_db: float = -90.0) -> float:
    if value <= 1e-15:
        return floor_db
    return max(20.0 * math.log10(value), floor_db)


def analyze_levels(left: List[float], right: List[float], rate: int, bands_hz: List[float]) -> Tuple[List[float], List[float]]:
    levels_l = [dbfs(band_rms_goertzel(left, rate, hz)) for hz in bands_hz]
    levels_r = [dbfs(band_rms_goertzel(right, rate, hz)) for hz in bands_hz]
    return levels_l, levels_r


def clamp(x: float, low: float, high: float) -> float:
    return min(max(x, low), high)


def suggest_auto(levels_db: List[float], max_boost: float, max_cut: float) -> Tuple[List[float], float]:
    target = sum(levels_db) / len(levels_db)
    gains = [clamp(target - lvl, -max_cut, max_boost) for lvl in levels_db]
    return gains, target


def _tilt_target(bands_hz: List[float], style: str) -> List[float]:
    # Positive values boost this band relative to mean level.
    style = style.lower()
    targets = []
    for hz in bands_hz:
        bass = 1.0 if hz <= 160 else 0.0
        low_mid = 1.0 if 160 < hz <= 1000 else 0.0
        presence = 1.0 if 1000 < hz <= 6000 else 0.0
        air = 1.0 if hz > 6000 else 0.0

        if style == "bass":
            t = 2.8 * bass + 1.2 * low_mid - 0.6 * presence - 0.2 * air
        elif style == "clarity":
            t = -1.5 * bass - 0.4 * low_mid + 1.7 * presence + 0.8 * air
        elif style == "vocal":
            t = -1.0 * bass + 0.8 * low_mid + 2.0 * presence + 0.2 * air
        else:  # balanced
            t = 0.4 * bass + 0.2 * low_mid + 0.5 * presence + 0.3 * air
        targets.append(t)
    return targets


def _smooth(values: List[float], radius: int = 1) -> List[float]:
    out: List[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def suggest_ai(levels_db: List[float], bands_hz: List[float], style: str, max_boost: float, max_cut: float) -> Tuple[List[float], float]:
    mean = sum(levels_db) / len(levels_db)
    style_tilt = _tilt_target(bands_hz, style)
    desired = [mean + t for t in style_tilt]
    raw = [desired[i] - levels_db[i] for i in range(len(levels_db))]
    smoothed = _smooth(raw, radius=1)
    gains = [clamp(x, -max_cut, max_boost) for x in smoothed]
    return gains, mean


def cross_channel_balance(gains_l: List[float], gains_r: List[float], max_delta: float = 3.0) -> Tuple[List[float], List[float]]:
    out_l: List[float] = []
    out_r: List[float] = []
    for gl, gr in zip(gains_l, gains_r):
        delta = gl - gr
        if abs(delta) > max_delta:
            mid = (gl + gr) * 0.5
            delta = max_delta if delta > 0 else -max_delta
            gl = mid + delta * 0.5
            gr = mid - delta * 0.5
        out_l.append(gl)
        out_r.append(gr)
    return out_l, out_r


def _limit_step_change(gains: List[float], teacher: List[float], max_delta_db: float = 1.2) -> List[float]:
    out: List[float] = []
    for g, t in zip(gains, teacher):
        if g > t + max_delta_db:
            out.append(t + max_delta_db)
        elif g < t - max_delta_db:
            out.append(t - max_delta_db)
        else:
            out.append(g)
    return out


def _run_local_ml(
    levels_l: List[float],
    levels_r: List[float],
    style: str,
    max_boost: float,
    max_cut: float,
    model_path: str,
    bands_hz: List[float],
) -> Tuple[List[float], List[float], float, float, Dict[str, object]]:
    teacher_l, target_l = suggest_ai(levels_l, bands_hz, style, max_boost, max_cut)
    teacher_r, target_r = suggest_ai(levels_r, bands_hz, style, max_boost, max_cut)
    teacher_l, teacher_r = cross_channel_balance(teacher_l, teacher_r)

    feature_count = len(make_features(levels_l[0], bands_hz[0], style))
    model = load_model(path=model_path, feature_count=feature_count, bands=len(bands_hz))
    ml = predict_and_learn(
        model=model,
        bands_hz=bands_hz,
        levels_l_db=levels_l,
        levels_r_db=levels_r,
        style=style,
        target_l_db=teacher_l,
        target_r_db=teacher_r,
        max_boost=max_boost,
        max_cut=max_cut,
    )
    save_model(model_path, model)

    gains_l = _limit_step_change(ml["gains_l_db"], teacher_l, max_delta_db=1.2)
    gains_r = _limit_step_change(ml["gains_r_db"], teacher_r, max_delta_db=1.2)
    gains_l, gains_r = cross_channel_balance(gains_l, gains_r)
    info = {
        "type": "local-online-linear",
        "model_path": model_path,
        "train_step": ml["step"],
        "train_mae": ml["mae"],
        "confidence": ml["confidence"],
        "teacher": "ai-heuristic",
    }
    return gains_l, gains_r, target_l, target_r, info


def analyze_stream(
    seconds: int = 8,
    rate: int = 48000,
    channels: int = 2,
    source: str = "",
    mode: str = "auto",
    style: str = "balanced",
    max_boost: float = 12.0,
    max_cut: float = 12.0,
    manual_l: List[float] = None,
    manual_r: List[float] = None,
    model_path: str = "data/ml_eq_model.json",
    high_res: bool = False,
) -> AnalysisResult:
    source = source or get_default_monitor_source()
    raw_path = record_raw_float32(source, seconds, rate, channels)
    try:
        left, right = read_stereo_samples(raw_path, channels)
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass

    max_n = 65536
    if len(left) > max_n:
        start = (len(left) - max_n) // 2
        left = left[start : start + max_n]
        right = right[start : start + max_n]

    bands_hz = select_bands_20(rate=rate, high_res=high_res)
    levels_l, levels_r = analyze_levels(left, right, rate, bands_hz)

    mode = mode.lower()
    model_info: Optional[Dict[str, object]] = None
    if mode == "manual":
        if manual_l is None or manual_r is None:
            raise ValueError("manual mode requires manual_l and manual_r arrays")
        gains_l = manual_l[: len(bands_hz)]
        gains_r = manual_r[: len(bands_hz)]
        target_l = sum(levels_l) / len(levels_l)
        target_r = sum(levels_r) / len(levels_r)
    elif mode == "ml":
        gains_l, gains_r, target_l, target_r, model_info = _run_local_ml(
            levels_l=levels_l,
            levels_r=levels_r,
            style=style,
            max_boost=max_boost,
            max_cut=max_cut,
            model_path=model_path,
            bands_hz=bands_hz,
        )
    elif mode in ("claude_ml", "hybrid"):
        gains_fallback_l, gains_fallback_r, target_l, target_r, fallback_info = _run_local_ml(
            levels_l=levels_l,
            levels_r=levels_r,
            style=style,
            max_boost=max_boost,
            max_cut=max_cut,
            model_path=model_path,
            bands_hz=bands_hz,
        )

        history_path = "data/claude_history.json"
        history: List[Dict[str, object]] = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded[-20:]
            except (OSError, json.JSONDecodeError):
                history = []

        try:
            claude = suggest_with_claude(
                bands_hz=bands_hz,
                levels_l_db=levels_l,
                levels_r_db=levels_r,
                style=style,
                max_boost=max_boost,
                max_cut=max_cut,
                history=history,
                timeout_seconds=6.0,
            )

            confidence = float(claude.get("confidence", 0.0))
            if confidence < 0.25 and mode == "claude_ml":
                raise RuntimeError(f"Claude confidence too low: {confidence:.2f}")

            # Keep Claude under local safety envelope by limiting deviation from local fallback.
            gains_l = _limit_step_change(claude["gains_l_db"], gains_fallback_l, max_delta_db=1.8)
            gains_r = _limit_step_change(claude["gains_r_db"], gains_fallback_r, max_delta_db=1.8)
            gains_l, gains_r = cross_channel_balance(gains_l, gains_r)

            model_info = {
                "type": "claude-remote",
                "model": claude.get("model", "unknown"),
                "confidence": confidence,
                "explanation": claude.get("explanation", ""),
                "fallback": fallback_info,
            }

            history.append(
                {
                    "ts": time.time(),
                    "style": style,
                    "confidence": confidence,
                    "mode": mode,
                }
            )
            os.makedirs("data", exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history[-30:], f, indent=2)
        except Exception as exc:
            if mode == "claude_ml":
                gains_l = gains_fallback_l
                gains_r = gains_fallback_r
                model_info = {
                    "type": "claude-fallback-local-ml",
                    "error": str(exc),
                    "fallback": fallback_info,
                }
            else:  # hybrid always falls back silently with metadata.
                gains_l = gains_fallback_l
                gains_r = gains_fallback_r
                model_info = {
                    "type": "hybrid-local-ml",
                    "remote_error": str(exc),
                    "fallback": fallback_info,
                }
    elif mode in ("openai_ml", "hybrid_openai"):
        gains_fallback_l, gains_fallback_r, target_l, target_r, fallback_info = _run_local_ml(
            levels_l=levels_l,
            levels_r=levels_r,
            style=style,
            max_boost=max_boost,
            max_cut=max_cut,
            model_path=model_path,
            bands_hz=bands_hz,
        )

        history_path = "data/openai_history.json"
        history: List[Dict[str, object]] = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded[-20:]
            except (OSError, json.JSONDecodeError):
                history = []

        try:
            openai = suggest_with_openai(
                bands_hz=bands_hz,
                levels_l_db=levels_l,
                levels_r_db=levels_r,
                style=style,
                max_boost=max_boost,
                max_cut=max_cut,
                history=history,
                timeout_seconds=6.0,
            )

            confidence = float(openai.get("confidence", 0.0))
            if confidence < 0.25 and mode == "openai_ml":
                raise RuntimeError(f"OpenAI confidence too low: {confidence:.2f}")

            gains_l = _limit_step_change(openai["gains_l_db"], gains_fallback_l, max_delta_db=1.8)
            gains_r = _limit_step_change(openai["gains_r_db"], gains_fallback_r, max_delta_db=1.8)
            gains_l, gains_r = cross_channel_balance(gains_l, gains_r)

            model_info = {
                "type": "openai-remote",
                "model": openai.get("model", "unknown"),
                "confidence": confidence,
                "explanation": openai.get("explanation", ""),
                "fallback": fallback_info,
            }

            history.append(
                {
                    "ts": time.time(),
                    "style": style,
                    "confidence": confidence,
                    "mode": mode,
                }
            )
            os.makedirs("data", exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history[-30:], f, indent=2)
        except Exception as exc:
            if mode == "openai_ml":
                gains_l = gains_fallback_l
                gains_r = gains_fallback_r
                model_info = {
                    "type": "openai-fallback-local-ml",
                    "error": str(exc),
                    "fallback": fallback_info,
                }
            else:
                gains_l = gains_fallback_l
                gains_r = gains_fallback_r
                model_info = {
                    "type": "hybrid-openai-local-ml",
                    "remote_error": str(exc),
                    "fallback": fallback_info,
                }
    elif mode == "vis":
        gains_l = [0.0] * len(bands_hz)
        gains_r = [0.0] * len(bands_hz)
        target_l = sum(levels_l) / len(levels_l)
        target_r = sum(levels_r) / len(levels_r)
    elif mode == "ai":
        gains_l, target_l = suggest_ai(levels_l, bands_hz, style, max_boost, max_cut)
        gains_r, target_r = suggest_ai(levels_r, bands_hz, style, max_boost, max_cut)
        gains_l, gains_r = cross_channel_balance(gains_l, gains_r)
    else:
        gains_l, target_l = suggest_auto(levels_l, max_boost, max_cut)
        gains_r, target_r = suggest_auto(levels_r, max_boost, max_cut)

    return AnalysisResult(
        source=source,
        timestamp=time.time(),
        bands_hz=bands_hz,
        levels_l_db=levels_l,
        levels_r_db=levels_r,
        gains_l_db=gains_l,
        gains_r_db=gains_r,
        target_l_db=target_l,
        target_r_db=target_r,
        mode=mode,
        style=style,
        model_info=model_info,
        rate=rate,
        high_res=high_res,
    )


def analyze_stream_parametric(
    seconds: int = 8,
    rate: int = 48000,
    source: str = "",
    style: str = "balanced",
    max_boost: float = 6.0,
    max_cut: float = 6.0,
) -> "AnalysisResultParametric":
    from .parametric import ParamResult, suggest_parametric_openai

    @dataclass
    class AnalysisResultParametric:
        source: str
        timestamp: float
        bands_hz: List[float]
        levels_l_db: List[float]
        levels_r_db: List[float]
        param: "ParamResult"
        style: str
        rate: int

    source = source or get_default_monitor_source()
    raw_path = record_raw_float32(source, seconds, rate, channels=2)
    try:
        left, right = read_stereo_samples(raw_path, 2)
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass

    bands_hz = select_bands_20(rate=rate, high_res=False)
    levels_l, levels_r = analyze_levels(left, right, rate, bands_hz)
    param = suggest_parametric_openai(
        levels_l, levels_r, bands_hz,
        style=style, max_boost=max_boost, max_cut=max_cut,
    )
    return AnalysisResultParametric(
        source=source,
        timestamp=time.time(),
        bands_hz=bands_hz,
        levels_l_db=levels_l,
        levels_r_db=levels_r,
        param=param,
        style=style,
        rate=rate,
    )


def save_snapshot(path: str, result: AnalysisResult) -> None:
    def _json_safe(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        return value

    payload: Dict[str, object] = {
        "source": result.source,
        "timestamp": result.timestamp,
        "bands_hz": result.bands_hz,
        "levels_l_db": result.levels_l_db,
        "levels_r_db": result.levels_r_db,
        "gains_l_db": result.gains_l_db,
        "gains_r_db": result.gains_r_db,
        "target_l_db": result.target_l_db,
        "target_r_db": result.target_r_db,
        "mode": result.mode,
        "style": result.style,
        "model_info": result.model_info,
        "rate": result.rate,
        "high_res": result.high_res,
    }
    payload = _json_safe(payload)
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_path, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, allow_nan=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

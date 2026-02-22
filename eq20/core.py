#!/usr/bin/env python3
import json
import math
import os
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    with open(raw_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    if proc.returncode not in (0, 124):
        raise RuntimeError(f"Capture failed ({proc.returncode}): {proc.stderr.strip()}")
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

    levels_l, levels_r = analyze_levels(left, right, rate, BANDS_20_HZ)

    mode = mode.lower()
    if mode == "manual":
        if manual_l is None or manual_r is None:
            raise ValueError("manual mode requires manual_l and manual_r arrays")
        gains_l = manual_l[: len(BANDS_20_HZ)]
        gains_r = manual_r[: len(BANDS_20_HZ)]
        target_l = sum(levels_l) / len(levels_l)
        target_r = sum(levels_r) / len(levels_r)
    elif mode == "ai":
        gains_l, target_l = suggest_ai(levels_l, BANDS_20_HZ, style, max_boost, max_cut)
        gains_r, target_r = suggest_ai(levels_r, BANDS_20_HZ, style, max_boost, max_cut)
        gains_l, gains_r = cross_channel_balance(gains_l, gains_r)
    else:
        gains_l, target_l = suggest_auto(levels_l, max_boost, max_cut)
        gains_r, target_r = suggest_auto(levels_r, max_boost, max_cut)

    return AnalysisResult(
        source=source,
        timestamp=time.time(),
        bands_hz=BANDS_20_HZ,
        levels_l_db=levels_l,
        levels_r_db=levels_r,
        gains_l_db=gains_l,
        gains_r_db=gains_r,
        target_l_db=target_l,
        target_r_db=target_r,
        mode=mode,
        style=style,
    )


def save_snapshot(path: str, result: AnalysisResult) -> None:
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
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

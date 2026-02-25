#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


STYLES = ["balanced", "bass", "clarity", "vocal"]


@dataclass
class OnlineEqModel:
    # Per-band linear weights for left and right channel.
    # Shape: [20 bands][feature_count]
    weights_l: List[List[float]]
    weights_r: List[List[float]]
    step: int = 0


def _zeros(bands: int, feats: int) -> List[List[float]]:
    return [[0.0 for _ in range(feats)] for _ in range(bands)]


def _style_one_hot(style: str) -> List[float]:
    style = (style or "balanced").lower()
    return [1.0 if style == s else 0.0 for s in STYLES]


def _region_flags(hz: float) -> Tuple[float, float, float, float]:
    bass = 1.0 if hz <= 160 else 0.0
    low_mid = 1.0 if 160 < hz <= 1000 else 0.0
    presence = 1.0 if 1000 < hz <= 6000 else 0.0
    air = 1.0 if hz > 6000 else 0.0
    return bass, low_mid, presence, air


def make_features(level_db: float, hz: float, style: str) -> List[float]:
    # Small feature vector that works without third-party ML dependencies.
    # [bias, level, level^2, normalized band position, region flags, style one-hot]
    log_hz = math.log10(max(hz, 20.0))
    norm_hz = (log_hz - math.log10(20.0)) / (math.log10(20000.0) - math.log10(20.0))
    bass, low_mid, presence, air = _region_flags(hz)
    one_hot = _style_one_hot(style)
    return [
        1.0,
        level_db / 24.0,
        (level_db / 24.0) ** 2,
        norm_hz,
        bass,
        low_mid,
        presence,
        air,
        *one_hot,
    ]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def new_model(feature_count: int, bands: int = 20) -> OnlineEqModel:
    return OnlineEqModel(weights_l=_zeros(bands, feature_count), weights_r=_zeros(bands, feature_count), step=0)


def load_model(path: str, feature_count: int, bands: int = 20) -> OnlineEqModel:
    if not os.path.exists(path):
        return new_model(feature_count=feature_count, bands=bands)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    weights_l = payload.get("weights_l")
    weights_r = payload.get("weights_r")
    step = int(payload.get("step", 0))

    if not isinstance(weights_l, list) or not isinstance(weights_r, list):
        return new_model(feature_count=feature_count, bands=bands)

    model = new_model(feature_count=feature_count, bands=bands)
    for i in range(min(bands, len(weights_l))):
        row = weights_l[i]
        if isinstance(row, list):
            for j in range(min(feature_count, len(row))):
                model.weights_l[i][j] = float(row[j])
    for i in range(min(bands, len(weights_r))):
        row = weights_r[i]
        if isinstance(row, list):
            for j in range(min(feature_count, len(row))):
                model.weights_r[i][j] = float(row[j])
    model.step = step
    return model


def save_model(path: str, model: OnlineEqModel) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": model.step,
                "weights_l": model.weights_l,
                "weights_r": model.weights_r,
            },
            f,
            indent=2,
        )


def predict_and_learn(
    model: OnlineEqModel,
    bands_hz: List[float],
    levels_l_db: List[float],
    levels_r_db: List[float],
    style: str,
    target_l_db: List[float],
    target_r_db: List[float],
    max_boost: float,
    max_cut: float,
    learning_rate: float = 0.03,
    l2: float = 0.0008,
) -> Dict[str, object]:
    gains_l: List[float] = []
    gains_r: List[float] = []
    err_abs_sum = 0.0

    for i, hz in enumerate(bands_hz):
        x_l = make_features(levels_l_db[i], hz, style)
        x_r = make_features(levels_r_db[i], hz, style)

        pred_l = _dot(model.weights_l[i], x_l)
        pred_r = _dot(model.weights_r[i], x_r)

        tgt_l = target_l_db[i]
        tgt_r = target_r_db[i]

        err_l = pred_l - tgt_l
        err_r = pred_r - tgt_r
        err_abs_sum += abs(err_l) + abs(err_r)

        # Online SGD step with light L2 regularization.
        for j in range(len(model.weights_l[i])):
            model.weights_l[i][j] -= learning_rate * (err_l * x_l[j] + l2 * model.weights_l[i][j])
            model.weights_r[i][j] -= learning_rate * (err_r * x_r[j] + l2 * model.weights_r[i][j])

        gains_l.append(_clamp(pred_l, -max_cut, max_boost))
        gains_r.append(_clamp(pred_r, -max_cut, max_boost))

    model.step += 1
    mae = err_abs_sum / max(1, 2 * len(bands_hz))

    # Confidence grows with training progress and lower prediction error.
    progress = min(1.0, model.step / 600.0)
    error_score = max(0.0, 1.0 - min(1.0, mae / 6.0))
    confidence = 0.35 * progress + 0.65 * error_score

    return {
        "gains_l_db": gains_l,
        "gains_r_db": gains_r,
        "mae": mae,
        "confidence": confidence,
        "step": model.step,
    }

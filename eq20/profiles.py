#!/usr/bin/env python3
import json
from typing import List, Tuple


def _expand_20(values: List[float]) -> List[float]:
    if not values:
        raise ValueError("Empty gain list")
    if len(values) >= 20:
        return values[:20]
    # Linear repeat fill for short profiles.
    out: List[float] = []
    i = 0
    while len(out) < 20:
        out.append(values[i % len(values)])
        i += 1
    return out


def load_manual_profile(path: str) -> Tuple[List[float], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    left = cfg.get("left_gains_db")
    right = cfg.get("right_gains_db")
    both = cfg.get("gains_db")

    if both and (left is None and right is None):
        left = both
        right = both

    if left is None or right is None:
        raise ValueError("Profile must contain left_gains_db and right_gains_db or gains_db")

    return _expand_20(left), _expand_20(right)

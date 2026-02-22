#!/usr/bin/env python3
import subprocess
from typing import List


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip())


def apply_split_20band(freqs: List[float], gains_l: List[float], gains_r: List[float]) -> None:
    if not (len(freqs) >= 20 and len(gains_l) >= 20 and len(gains_r) >= 20):
        raise ValueError("Need 20 frequency and gain values per channel")

    _run(["gsettings", "set", "com.github.wwmm.easyeffects.streamoutputs", "plugins", "['equalizer']"])
    _run([
        "gsettings",
        "set",
        "com.github.wwmm.easyeffects.equalizer:/com/github/wwmm/easyeffects/streamoutputs/equalizer/",
        "bypass",
        "false",
    ])
    _run([
        "gsettings",
        "set",
        "com.github.wwmm.easyeffects.equalizer:/com/github/wwmm/easyeffects/streamoutputs/equalizer/",
        "mode",
        "IIR",
    ])
    _run([
        "gsettings",
        "set",
        "com.github.wwmm.easyeffects.equalizer:/com/github/wwmm/easyeffects/streamoutputs/equalizer/",
        "num-bands",
        "20",
    ])
    _run([
        "gsettings",
        "set",
        "com.github.wwmm.easyeffects.equalizer:/com/github/wwmm/easyeffects/streamoutputs/equalizer/",
        "split-channels",
        "true",
    ])

    for i in range(20):
        f = str(freqs[i])
        gl = str(gains_l[i])
        gr = str(gains_r[i])
        _run([
            "gsettings",
            "set",
            "com.github.wwmm.easyeffects.equalizer.channel:/com/github/wwmm/easyeffects/streamoutputs/equalizer/leftchannel/",
            f"band{i}-frequency",
            f,
        ])
        _run([
            "gsettings",
            "set",
            "com.github.wwmm.easyeffects.equalizer.channel:/com/github/wwmm/easyeffects/streamoutputs/equalizer/rightchannel/",
            f"band{i}-frequency",
            f,
        ])
        _run([
            "gsettings",
            "set",
            "com.github.wwmm.easyeffects.equalizer.channel:/com/github/wwmm/easyeffects/streamoutputs/equalizer/leftchannel/",
            f"band{i}-gain",
            gl,
        ])
        _run([
            "gsettings",
            "set",
            "com.github.wwmm.easyeffects.equalizer.channel:/com/github/wwmm/easyeffects/streamoutputs/equalizer/rightchannel/",
            f"band{i}-gain",
            gr,
        ])

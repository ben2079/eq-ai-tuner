#!/usr/bin/env python3
import subprocess
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .parametric import ParamBand

EQ_BASE = "/com/github/wwmm/easyeffects/streamoutputs/equalizer"


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip())


def _dconf_write(path: str, value: str) -> None:
    _run(["dconf", "write", path, value])


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


# EasyEffects type name → dconf string value
_TYPE_MAP = {
    "Bell": "'Bell'",
    "Low Shelf": "'Lo Shelf'",
    "High Shelf": "'Hi Shelf'",
    "Notch": "'Notch'",
    "High Pass": "'Hi Pass'",
    "Low Pass": "'Lo Pass'",
}


def apply_parametric_eq(bands_l: "List[ParamBand]", bands_r: "List[ParamBand]") -> None:
    """Write parametric EQ bands to EasyEffects via dconf.

    Uses at most 16 bands per channel. Unused bands are muted.
    """
    n = max(len(bands_l), len(bands_r), 1)
    n = min(n, 16)

    # Set global EQ config
    _dconf_write(f"{EQ_BASE}/bypass", "false")
    _dconf_write(f"{EQ_BASE}/mode", "'IIR'")
    _dconf_write(f"{EQ_BASE}/num-bands", str(n))
    _dconf_write(f"{EQ_BASE}/split-channels", "true")

    for ch, bands in (("leftchannel", bands_l), ("rightchannel", bands_r)):
        base = f"{EQ_BASE}/{ch}"
        for i in range(n):
            if i < len(bands):
                b = bands[i]
                dtype = _TYPE_MAP.get(b.type, "'Bell'")
                _dconf_write(f"{base}/band{i}-frequency", str(b.freq))
                _dconf_write(f"{base}/band{i}-gain", str(b.gain))
                _dconf_write(f"{base}/band{i}-q", str(b.q))
                _dconf_write(f"{base}/band{i}-type", dtype)
                _dconf_write(f"{base}/band{i}-mode", "'RLC (BT)'")
                _dconf_write(f"{base}/band{i}-mute", "false")
            else:
                # Mute unused bands
                _dconf_write(f"{base}/band{i}-mute", "true")
                _dconf_write(f"{base}/band{i}-gain", "0.0")



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

#!/usr/bin/env python3
import argparse
import json
import math
import os
import socket
import struct
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from eq20.core import analyze_stream, analyze_stream_parametric, get_default_monitor_source, save_snapshot
from eq20.easyeffects import apply_parametric_eq, apply_split_20band
from eq20.profiles import load_manual_profile


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


class FastSpectrumMonitor:
    """Continuous realtime spectrum via parec + numpy FFT, ~25 fps."""

    STEP_FRAMES = 1920   # 40 ms at 48 kHz
    FFT_SIZE = 4096      # ~85 ms window, ~11.7 Hz resolution
    OUT_BINS = 120       # log-spaced output bins
    MIN_HZ = 20.0
    MAX_HZ = 20000.0
    FLOOR_DB = -90.0

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc: Optional[object] = None
        self.rate = 48000
        self.source = ""

    def start(self, source: str = "", rate: int = 48000) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.rate = rate
        self.source = source
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc:
            try:
                proc.terminate()  # type: ignore[attr-defined]
            except OSError:
                pass

    def get_latest(self) -> Optional[dict]:
        with self.lock:
            return self.latest

    def _loop(self) -> None:
        try:
            import numpy as np
        except ImportError:
            return

        import subprocess as _sp

        rate = self.rate
        channels = 2
        fft_size = self.FFT_SIZE
        step = self.STEP_FRAMES
        bytes_per_frame = 4 * channels
        step_bytes = step * bytes_per_frame

        out_freqs: list = list(np.logspace(
            np.log10(self.MIN_HZ), np.log10(self.MAX_HZ), self.OUT_BINS
        ))
        fft_bin_hz = rate / fft_size
        out_bin_idx = [
            max(1, min(int(round(f / fft_bin_hz)), fft_size // 2 - 1))
            for f in out_freqs
        ]

        window = np.hanning(fft_size)
        window_sum_sq = float(np.sum(window ** 2))

        buf_l = np.zeros(fft_size, dtype=np.float32)
        buf_r = np.zeros(fft_size, dtype=np.float32)
        write_pos = 0

        source = self.source
        if not source:
            try:
                source = get_default_monitor_source()
            except Exception:
                return

        cmd = [
            "parec", "-d", source,
            "--format=float32le",
            f"--rate={rate}",
            f"--channels={channels}",
            "--raw",
        ]
        try:
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)
        except Exception:
            return
        self._proc = proc

        raw_buf = b""
        floor = self.FLOOR_DB
        try:
            while not self._stop.is_set():
                needed = step_bytes - len(raw_buf)
                if needed > 0:
                    chunk = proc.stdout.read(needed)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    raw_buf += chunk
                if len(raw_buf) < step_bytes:
                    continue

                frame_data = np.frombuffer(raw_buf[:step_bytes], dtype="<f4")
                raw_buf = raw_buf[step_bytes:]

                new_l = frame_data[0::2]
                new_r = frame_data[1::2]
                n = len(new_l)
                end = (write_pos + n) % fft_size
                if write_pos + n <= fft_size:
                    buf_l[write_pos:write_pos + n] = new_l
                    buf_r[write_pos:write_pos + n] = new_r
                else:
                    first = fft_size - write_pos
                    buf_l[write_pos:] = new_l[:first]
                    buf_l[:n - first] = new_l[first:]
                    buf_r[write_pos:] = new_r[:first]
                    buf_r[:n - first] = new_r[first:]
                write_pos = end

                ordered_l = np.roll(buf_l, -write_pos)
                ordered_r = np.roll(buf_r, -write_pos)

                Fl = np.fft.rfft(ordered_l * window)
                Fr = np.fft.rfft(ordered_r * window)
                power_l = np.abs(Fl) ** 2 / window_sum_sq
                power_r = np.abs(Fr) ** 2 / window_sum_sq

                def _db_bins(power: "np.ndarray") -> list:
                    out = []
                    for idx in out_bin_idx:
                        p = float(power[idx])
                        db = (10.0 * math.log10(p) if p > 1e-30 else floor)
                        out.append(max(floor, db))
                    return out

                payload = {
                    "freqs": out_freqs,
                    "bins_l": _db_bins(power_l),
                    "bins_r": _db_bins(power_r),
                    "ts": time.time(),
                }
                with self.lock:
                    self.latest = payload
        finally:
            try:
                proc.terminate()
            except OSError:
                pass
            self._proc = None


_spectrum_monitor = FastSpectrumMonitor()


def _probe_openai(timeout: float = 4.0) -> dict:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_EQ_MODEL", "gpt-4.1-mini").strip()
    info = {
        "configured": bool(key),
        "model": model,
        "ok": False,
        "error": "",
    }
    if not key:
        info["error"] = "OPENAI_API_KEY missing"
        return info

    payload = {
        "model": model,
        "max_tokens": 16,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only valid JSON object."},
            {"role": "user", "content": '{"ok":true}'},
        ],
    }
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        choices = data.get("choices", [])
        info["ok"] = isinstance(choices, list) and len(choices) > 0
        if not info["ok"]:
            info["error"] = "No choices in response"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _probe_claude(timeout: float = 4.0) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("CLAUDE_EQ_MODEL", "claude-3-5-sonnet-latest").strip()
    info = {
        "configured": bool(key),
        "model": model,
        "ok": False,
        "error": "",
    }
    if not key:
        info["error"] = "ANTHROPIC_API_KEY missing"
        return info

    payload = {
        "model": model,
        "max_tokens": 8,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "reply ok"}]}],
    }
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        content = data.get("content", [])
        info["ok"] = isinstance(content, list)
        if not info["ok"]:
            info["error"] = "No content in response"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _blend(prev_vals, new_vals, alpha):
    if not prev_vals:
        return list(new_vals)
    out = []
    for p, n in zip(prev_vals, new_vals):
        out.append((1.0 - alpha) * p + alpha * n)
    return out


class EqAgentController:
    def __init__(self, snapshot_path: str):
        self.snapshot_path = snapshot_path
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.last_error = ""
        self.last_run = 0.0
        self.iteration = 0
        self.prev_l = None
        self.prev_r = None
        self.cfg = {
            "mode": "hybrid_openai",
            "style": "balanced",
            "apply": True,
            "seconds": 2,
            "rate": 48000,
            "high_res": False,
            "interval": 1.0,
            "max_boost": 6.0,
            "max_cut": 6.0,
            "manual_profile": "configs/manual_profile.example.json",
            "model_path": "data/ml_eq_model.json",
            "smooth_alpha": 0.40,
            "source": "",
        }

    def _run_once_locked(self):
        manual_l = manual_r = None
        mode = self.cfg["mode"]
        if mode == "manual":
            manual_l, manual_r = load_manual_profile(self.cfg["manual_profile"])

        result = analyze_stream(
            seconds=int(self.cfg["seconds"]),
            rate=int(self.cfg.get("rate", 48000)),
            source=self.cfg.get("source", ""),
            mode=mode,
            style=self.cfg["style"],
            max_boost=float(self.cfg["max_boost"]),
            max_cut=float(self.cfg["max_cut"]),
            manual_l=manual_l,
            manual_r=manual_r,
            model_path=self.cfg.get("model_path", "data/ml_eq_model.json"),
            high_res=bool(self.cfg.get("high_res", False)),
        )

        alpha = float(self.cfg.get("smooth_alpha", 0.40))
        if mode in ("ai", "ml", "auto", "claude_ml", "hybrid", "openai_ml", "hybrid_openai") and 0.0 < alpha <= 1.0:
            result.gains_l_db = _blend(self.prev_l, result.gains_l_db, alpha)
            result.gains_r_db = _blend(self.prev_r, result.gains_r_db, alpha)

        self.prev_l = list(result.gains_l_db)
        self.prev_r = list(result.gains_r_db)
        save_snapshot(self.snapshot_path, result)

        if bool(self.cfg["apply"]) and self.cfg.get("mode", "") != "vis":
            apply_split_20band(result.bands_hz, result.gains_l_db, result.gains_r_db)

        self.last_run = time.time()
        self.iteration += 1

    def _loop(self):
        while not self.stop_event.is_set():
            start = time.time()
            try:
                with self.lock:
                    self._run_once_locked()
                    self.last_error = ""
                    interval = float(self.cfg["interval"])
            except Exception as exc:
                self.last_error = str(exc)
                interval = 1.0

            elapsed = time.time() - start
            sleep_for = max(0.0, interval - elapsed)
            self.stop_event.wait(sleep_for)

    def start(self, updates: dict):
        with self.lock:
            self.cfg.update(updates or {})
            if self.running:
                return
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.running = True
            self.thread.start()

    def stop(self):
        thread = None
        with self.lock:
            if not self.running:
                return
            self.stop_event.set()
            thread = self.thread
            self.running = False
            self.thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def update(self, updates: dict):
        with self.lock:
            self.cfg.update(updates or {})

    def run_once(self, updates: dict = None):
        with self.lock:
            if updates:
                self.cfg.update(updates)
            self._run_once_locked()

    def status(self):
        with self.lock:
            return {
                "running": self.running,
                "config": dict(self.cfg),
                "last_error": self.last_error,
                "last_run": self.last_run,
                "iteration": self.iteration,
            }


class Handler(SimpleHTTPRequestHandler):
    # Shared cache to survive partial snapshot writes across request handlers.
    snapshot_cache = json.dumps({"status": "waiting", "message": "No snapshot yet"})

    def __init__(
        self,
        *args,
        web_root: str = "web",
        snapshot_path: str = "data/latest_snapshot.json",
        eq_agent: EqAgentController = None,
        **kwargs,
    ):
        self.web_root = web_root
        self.snapshot_path = snapshot_path
        self.eq_agent = eq_agent
        super().__init__(*args, directory=self.web_root, **kwargs)

    def end_headers(self):
        # Avoid stale UI/assets when switching service versions.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/snapshot":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self._safe_snapshot_payload().encode("utf-8"))
            return
        if parsed.path == "/api/eq_agent":
            if self.eq_agent is None:
                self._json_response(503, {"ok": False, "error": "eq_agent not configured"})
            else:
                self._json_response(200, {"ok": True, "agent": self.eq_agent.status()})
            return
        if parsed.path == "/api/provider_health":
            qs = parse_qs(parsed.query)
            probe = qs.get("probe", ["0"])[0] == "1"
            payload = {
                "ok": True,
                "probe": probe,
                "openai": {
                    "configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                    "model": os.environ.get("OPENAI_EQ_MODEL", "gpt-4.1-mini").strip(),
                },
                "claude": {
                    "configured": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
                    "model": os.environ.get("CLAUDE_EQ_MODEL", "claude-3-5-sonnet-latest").strip(),
                },
            }
            if probe:
                payload["openai"] = _probe_openai()
                payload["claude"] = _probe_claude()
            self._json_response(200, payload)
            return
        if parsed.path == "/api/stream":
            self._stream_snapshot_sse()
            return
        if parsed.path == "/api/spectrum_stream":
            self._stream_spectrum_sse()
            return
        return super().do_GET()

    def _stream_snapshot_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_payload = ""
        try:
            while True:
                payload = self._safe_snapshot_payload()

                if payload != last_payload:
                    self.wfile.write(self._format_sse_event("snapshot", payload).encode("utf-8"))
                    self.wfile.flush()
                    last_payload = payload
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream_spectrum_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_ts = 0.0
        try:
            while True:
                data = _spectrum_monitor.get_latest()
                if data and data["ts"] != last_ts:
                    payload = json.dumps(data)
                    self.wfile.write(
                        self._format_sse_event("spectrum", payload).encode("utf-8")
                    )
                    self.wfile.flush()
                    last_ts = data["ts"]
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _safe_snapshot_payload(self) -> str:
        waiting = json.dumps({"status": "waiting", "message": "No snapshot yet"})
        if not os.path.exists(self.snapshot_path):
            return Handler.snapshot_cache or waiting

        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                payload = f.read()
            # Validate and normalize JSON so browsers never receive NaN/Infinity.
            parsed = json.loads(payload)
            parsed = self._json_safe(parsed)
            payload = json.dumps(parsed, ensure_ascii=False)
            Handler.snapshot_cache = payload
            return payload
        except Exception:
            return Handler.snapshot_cache or waiting

    def _json_safe(self, value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        return value

    @staticmethod
    def _format_sse_event(event_name: str, payload: str) -> str:
        # SSE requires each data line to start with "data: ".
        lines = payload.splitlines() or [""]
        out = [f"event: {event_name}"]
        out.extend(f"data: {line}" for line in lines)
        out.append("")
        out.append("")
        return "\n".join(out)

    def _json_response(self, code: int, payload: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/run", "/api/eq_agent", "/api/run_parametric"):
            self._json_response(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(body.decode("utf-8"))

            if parsed.path == "/api/run_parametric":
                self._handle_run_parametric(req)
                return

            mode = req.get("mode", "auto")
            style = req.get("style", "balanced")
            apply = bool(req.get("apply", False))
            seconds = int(req.get("seconds", 4))
            rate = int(req.get("rate", 48000))
            high_res = bool(req.get("high_res", False))
            interval = float(req.get("interval", 1.0))
            max_boost = float(req.get("max_boost", 6.0))
            max_cut = float(req.get("max_cut", 6.0))
            manual_profile = req.get("manual_profile", "configs/manual_profile.example.json")
            model_path = req.get("model_path", "data/ml_eq_model.json")
            smooth_alpha = float(req.get("smooth_alpha", 0.40))
            source = req.get("source", "")

            if parsed.path == "/api/eq_agent":
                if self.eq_agent is None:
                    self._json_response(503, {"ok": False, "error": "eq_agent not configured"})
                    return

                action = req.get("action", "status")
                updates = {
                    "mode": mode,
                    "style": style,
                    "apply": apply,
                    "seconds": seconds,
                    "rate": rate,
                    "high_res": high_res,
                    "interval": interval,
                    "max_boost": max_boost,
                    "max_cut": max_cut,
                    "manual_profile": manual_profile,
                    "model_path": model_path,
                    "smooth_alpha": smooth_alpha,
                    "source": source,
                }

                if action == "start":
                    self.eq_agent.start(updates)
                elif action == "stop":
                    self.eq_agent.stop()
                elif action == "update":
                    self.eq_agent.update(updates)
                elif action == "run_once":
                    self.eq_agent.run_once(updates)
                elif action == "status":
                    pass
                else:
                    self._json_response(400, {"ok": False, "error": f"Unknown action: {action}"})
                    return

                self._json_response(200, {"ok": True, "agent": self.eq_agent.status()})
                return

            manual_l = manual_r = None
            if mode == "manual":
                manual_l, manual_r = load_manual_profile(manual_profile)

            result = analyze_stream(
                seconds=seconds,
                rate=rate,
                mode=mode,
                style=style,
                max_boost=max_boost,
                max_cut=max_cut,
                manual_l=manual_l,
                manual_r=manual_r,
                model_path=model_path,
                source=source,
                high_res=high_res,
            )
            save_snapshot(self.snapshot_path, result)

            if apply:
                apply_split_20band(result.bands_hz, result.gains_l_db, result.gains_r_db)

            self._json_response(
                200,
                {
                    "ok": True,
                    "mode": result.mode,
                    "style": result.style,
                    "applied": apply,
                    "snapshot": self.snapshot_path,
                },
            )
        except Exception as exc:
            self._json_response(500, {"ok": False, "error": str(exc), "trace": traceback.format_exc()})

    def _handle_run_parametric(self, req: dict) -> None:
        style = req.get("style", "balanced")
        apply = bool(req.get("apply", True))
        seconds = int(req.get("seconds", 6))
        rate = int(req.get("rate", 48000))
        max_boost = float(req.get("max_boost", 6.0))
        max_cut = float(req.get("max_cut", 6.0))
        source = req.get("source", "")

        result = analyze_stream_parametric(
            seconds=seconds,
            rate=rate,
            source=source,
            style=style,
            max_boost=max_boost,
            max_cut=max_cut,
        )

        if result.param.error:
            self._json_response(500, {"ok": False, "error": result.param.error})
            return

        bands_l_json = [
            {"freq": b.freq, "gain": b.gain, "q": b.q, "type": b.type}
            for b in result.param.bands_l
        ]
        bands_r_json = [
            {"freq": b.freq, "gain": b.gain, "q": b.q, "type": b.type}
            for b in result.param.bands_r
        ]

        if apply:
            apply_parametric_eq(result.param.bands_l, result.param.bands_r)

        # Write a snapshot so the UI sees the new state
        from eq20.core import AnalysisResult
        import time as _time
        n = len(result.bands_hz)
        gains_l = [0.0] * n
        gains_r = [0.0] * n
        snapshot = AnalysisResult(
            source=result.source,
            timestamp=_time.time(),
            bands_hz=result.bands_hz,
            levels_l_db=result.levels_l_db,
            levels_r_db=result.levels_r_db,
            gains_l_db=gains_l,
            gains_r_db=gains_r,
            target_l_db=0.0,
            target_r_db=0.0,
            mode="parametric_openai",
            style=style,
            model_info={"bands_l": bands_l_json, "bands_r": bands_r_json, "model": result.param.model},
        )
        save_snapshot(self.snapshot_path, snapshot)

        self._json_response(200, {
            "ok": True,
            "applied": apply,
            "model": result.param.model,
            "bands_l": bands_l_json,
            "bands_r": bands_r_json,
        })


def main():
    parser = argparse.ArgumentParser(description="Live EQ dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--snapshot", default="data/latest_snapshot.json")
    args = parser.parse_args()

    eq_agent = EqAgentController(snapshot_path=args.snapshot)
    _spectrum_monitor.start(rate=48000)

    def factory(*factory_args, **factory_kwargs):
        return Handler(
            *factory_args,
            web_root=args.web_root,
            snapshot_path=args.snapshot,
            eq_agent=eq_agent,
            **factory_kwargs,
        )

    server_cls = ThreadingHTTPServerV6 if ":" in args.host else ThreadingHTTPServer
    httpd = server_cls((args.host, args.port), factory)
    host_for_url = args.host if ":" not in args.host else f"[{args.host}]"
    print(f"Dashboard: http://{host_for_url}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from eq20.core import analyze_stream, save_snapshot
from eq20.easyeffects import apply_split_20band
from eq20.profiles import load_manual_profile


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, web_root: str = "web", snapshot_path: str = "data/latest_snapshot.json", **kwargs):
        self.web_root = web_root
        self.snapshot_path = snapshot_path
        super().__init__(*args, directory=self.web_root, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/snapshot":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            if os.path.exists(self.snapshot_path):
                with open(self.snapshot_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({"status": "waiting", "message": "No snapshot yet"}).encode("utf-8"))
            return
        return super().do_GET()

    def _json_response(self, code: int, payload: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._json_response(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(body.decode("utf-8"))

            mode = req.get("mode", "auto")
            style = req.get("style", "balanced")
            apply = bool(req.get("apply", False))
            seconds = int(req.get("seconds", 4))
            max_boost = float(req.get("max_boost", 6.0))
            max_cut = float(req.get("max_cut", 6.0))
            manual_profile = req.get("manual_profile", "configs/manual_profile.example.json")

            manual_l = manual_r = None
            if mode == "manual":
                manual_l, manual_r = load_manual_profile(manual_profile)

            result = analyze_stream(
                seconds=seconds,
                mode=mode,
                style=style,
                max_boost=max_boost,
                max_cut=max_cut,
                manual_l=manual_l,
                manual_r=manual_r,
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


def main():
    parser = argparse.ArgumentParser(description="Live EQ dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--snapshot", default="data/latest_snapshot.json")
    args = parser.parse_args()

    def factory(*factory_args, **factory_kwargs):
        return Handler(*factory_args, web_root=args.web_root, snapshot_path=args.snapshot, **factory_kwargs)

    httpd = ThreadingHTTPServer((args.host, args.port), factory)
    print(f"Dashboard: http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

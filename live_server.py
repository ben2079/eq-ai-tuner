#!/usr/bin/env python3
import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


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

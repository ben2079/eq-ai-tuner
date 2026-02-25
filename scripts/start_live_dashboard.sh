#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 live_server.py --host 127.0.0.1 --port 8765 --web-root web --snapshot data/latest_snapshot.json

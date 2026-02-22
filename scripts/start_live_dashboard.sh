#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Start continuous analyzer in background
python3 eq_tool.py --mode ai --style balanced --loop --interval 3 --seconds 4 --max-boost 6 --max-cut 6 --snapshot data/latest_snapshot.json &
ANALYZER_PID=$!

cleanup() {
  kill "$ANALYZER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 live_server.py --host 127.0.0.1 --port 8765 --web-root web --snapshot data/latest_snapshot.json

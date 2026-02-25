#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PORT="8766"
HOST="0.0.0.0"
PID_FILE="data/live_server_${PORT}.pid"
LOG_FILE="data/live_server_${PORT}.log"

mkdir -p data

# Stop previous instance from pid file if it is still running.
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    kill "$old_pid" || true
    sleep 1
  fi
fi

# Kill any stale process still bound to this port.
if command -v fuser >/dev/null 2>&1; then
  fuser -k -n tcp "$PORT" >/dev/null 2>&1 || true
fi

# Export variables from .env for provider support.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

nohup python3 live_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --web-root web \
  --snapshot data/latest_snapshot.json \
  >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") url=http://$(hostname -I | awk '{print $1}'):${PORT}"

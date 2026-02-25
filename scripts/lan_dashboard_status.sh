#!/usr/bin/env bash
set -euo pipefail

PORT="8766"
PID_FILE="$(cd "$(dirname "$0")/.." && pwd)/data/live_server_${PORT}.pid"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "running pid=$pid"
  else
    echo "stopped (stale pid file)"
  fi
else
  echo "stopped"
fi

ss -ltnp | grep ":${PORT}" || true
curl -sS -m 3 -I "http://127.0.0.1:${PORT}" | head -n 1 || true

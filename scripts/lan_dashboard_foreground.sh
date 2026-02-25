#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec python3 live_server.py \
  --host 0.0.0.0 \
  --port 8766 \
  --web-root web \
  --snapshot data/latest_snapshot.json

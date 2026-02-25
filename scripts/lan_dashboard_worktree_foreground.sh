#!/usr/bin/env bash
set -euo pipefail

WORKTREE_DIR="/home/benjamin/eq-ai-tuner.worktrees/copilot-worktree-2026-02-25T18-56-01"
MAIN_DIR="/home/benjamin/eq-ai-tuner"

cd "$WORKTREE_DIR"

# Load provider keys from worktree .env if present, otherwise from main repo .env.
if [[ -f "$WORKTREE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$WORKTREE_DIR/.env"
  set +a
elif [[ -f "$MAIN_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$MAIN_DIR/.env"
  set +a
fi

exec python3 live_server.py \
  --host 0.0.0.0 \
  --port 8766 \
  --web-root web \
  --snapshot data/latest_snapshot.json

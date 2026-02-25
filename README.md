# EQ AI Tuner (2CH / 20-Band)

[![CI](https://github.com/ben2079/eq-ai-tuner/actions/workflows/ci.yml/badge.svg)](https://github.com/ben2079/eq-ai-tuner/actions/workflows/ci.yml)

Real-time stereo EQ tuning pipeline for Linux web audio (for example TIDAL in browser) with:

- `Manual` settings (per-channel, per-band)
- `Auto` settings (content-driven leveling)
- `AI-driven` settings (heuristic style engine + channel balancing)
- `ML` settings (local online model trained continuously on-device)
- `Claude ML` settings (remote inference with local safety gates)
- `OpenAI ML` settings (remote inference with local safety gates)
- Animated live dashboard (2CH/20-band)
- Optional direct apply to EasyEffects (split channels)

## What This Project Delivers

- 20-band analysis for left and right channel
- Separate gain recommendation per channel and band
- Continuous analysis loop writing live snapshots
- Browser-based animated EQ visualization
- EasyEffects integration for practical playback tuning

## Project Structure

- `eq_tool.py`
  - Main CLI for one-shot and loop analysis
  - Modes: `manual`, `auto`, `ai`, `ml`, `claude_ml`, `hybrid`, `openai_ml`, `hybrid_openai`
  - Optional apply to EasyEffects

- `live_server.py`
  - Lightweight HTTP server
  - Serves dashboard and `/api/snapshot`
  - `eq_agent` controller API for continuous background analysis
  - SSE endpoint `/api/stream` for low-latency live updates

- `eq20/core.py`
  - Audio capture + frequency analysis + gain logic

- `eq20/profiles.py`
  - Manual profile loader (`JSON`)

- `eq20/easyeffects.py`
  - Split-channel 20-band write to EasyEffects via `gsettings`

- `web/index.html`
  - Animated live 2CH/20-band graph

- `configs/manual_profile.example.json`
  - Example manual profile template

## Requirements

```bash
sudo apt update
sudo apt install -y pulseaudio-utils easyeffects
```

Python: `3.9+` recommended.

## Quick Start

### 1) One-shot Auto Analysis

```bash
cd eq-ai-tuner
python3 eq_tool.py --mode auto --seconds 6 --snapshot data/latest_snapshot.json
```

### 2) One-shot AI-driven Analysis + Apply to EasyEffects

```bash
python3 eq_tool.py --mode ai --style balanced --seconds 6 --max-boost 6 --max-cut 6 --apply
```

### 2b) One-shot ML Analysis + local model update

```bash
python3 eq_tool.py --mode ml --style balanced --seconds 2 --max-boost 6 --max-cut 6 --model-path data/ml_eq_model.json --apply
```

High-res band layout (up to 28 kHz, requires 96 kHz+ stream rate):

```bash
python3 eq_tool.py --mode hybrid_openai --rate 96000 --high-res --seconds 2 --max-boost 6 --max-cut 6 --apply
```

### 2c) One-shot Claude ML (with local fallback safety)

```bash
export ANTHROPIC_API_KEY="<your_key>"
export CLAUDE_EQ_MODEL="claude-3-5-sonnet-latest"
python3 eq_tool.py --mode claude_ml --style balanced --seconds 2 --max-boost 6 --max-cut 6 --model-path data/ml_eq_model.json --apply
```

### 2d) One-shot Hybrid (prefer Claude, fallback to local ML)

```bash
export ANTHROPIC_API_KEY="<your_key>"
python3 eq_tool.py --mode hybrid --style balanced --seconds 2 --max-boost 6 --max-cut 6 --model-path data/ml_eq_model.json --apply
```

### 2e) One-shot OpenAI ML (with local fallback safety)

```bash
export OPENAI_API_KEY="<your_key>"
export OPENAI_EQ_MODEL="gpt-4.1-mini"
python3 eq_tool.py --mode openai_ml --style balanced --seconds 2 --max-boost 6 --max-cut 6 --model-path data/ml_eq_model.json --apply
```

### 2f) One-shot Hybrid OpenAI (prefer OpenAI, fallback local ML)

```bash
export OPENAI_API_KEY="<your_key>"
python3 eq_tool.py --mode hybrid_openai --style balanced --seconds 2 --max-boost 6 --max-cut 6 --model-path data/ml_eq_model.json --apply
```

Styles:

- `balanced`
- `bass`
- `clarity`
- `vocal`

### 3) Manual Mode (per-channel)

Edit `configs/manual_profile.example.json`, then run:

```bash
python3 eq_tool.py --mode manual --manual-profile configs/manual_profile.example.json --apply
```

### 4) Animated Live Dashboard

```bash
chmod +x scripts/start_live_dashboard.sh
./scripts/start_live_dashboard.sh
```

Open:

- `http://127.0.0.1:8765`

This starts:

- live web server (`live_server.py`)
- optional background `eq_agent` loop (start in UI)

## Manual vs Auto vs AI-driven vs ML vs Claude ML vs OpenAI ML

- `manual`
  - uses your fixed left/right gain arrays
  - good for controlled personal profiles

- `auto`
  - computes channel-specific correction from current stream energy
  - straightforward content leveling

- `ai`
  - heuristic target shaping by style plus cross-channel balancing
  - smoother, listening-oriented correction behavior

- `ml`
  - local online model (`data/ml_eq_model.json`) learns continuously on-device
  - starts with heuristic teacher targets, then adapts over time
  - exposes training step, MAE and confidence in dashboard metadata

## High-Res Bands (26/28 kHz)

- At `48 kHz`, Nyquist is `24 kHz`, so 26/28 kHz analysis is not physically valid.
- Use `--rate 96000 --high-res` to enable the extended top-end band layout.
- UI now exposes `Sample Rate` and `High-Res Bands` directly.
- If `High-Res` is enabled below `96 kHz`, the analyzer automatically falls back to standard 20 kHz top band layout.

- `claude_ml`
  - calls Claude for 20-band stereo recommendations
  - applies strict local safety gates (clamp, deviation limit, channel balance)
  - if remote call fails or confidence is low, falls back to local ML and keeps running

- `hybrid`
  - tries Claude first, then falls back to local ML automatically
  - designed for stable continuous operation when network/API is unstable

- `openai_ml`
  - calls OpenAI for 20-band stereo recommendations
  - applies local safety gates (clamp, deviation limit, channel balance)
  - falls back to local ML on API/validation failures

- `hybrid_openai`
  - tries OpenAI first, then local ML fallback automatically
  - intended for stable background operation in `eq_agent`

> Note: "AI-driven" here is on-device heuristic intelligence, not cloud ML inference.

## Claude Integration

Environment variables:

- `ANTHROPIC_API_KEY` (required for `claude_ml`/`hybrid` remote inference)
- `CLAUDE_EQ_MODEL` (optional, default: `claude-3-5-sonnet-latest`)
- `OPENAI_API_KEY` (required for `openai_ml`/`hybrid_openai` remote inference)
- `OPENAI_EQ_MODEL` (optional, default: `gpt-4.1-mini`)

Runtime behavior:

- Local ML (`data/ml_eq_model.json`) is always available as safety fallback.
- Claude request history metadata is stored locally in `data/claude_history.json`.
- OpenAI request history metadata is stored locally in `data/openai_history.json`.

## EQ Agent API (continuous background)

`live_server.py` now includes an EQ agent service:

- `GET /api/eq_agent` -> status/config
- `POST /api/eq_agent` with `action=start|stop|update|run_once|status`

Minimal start example:

```bash
curl -sS -X POST http://127.0.0.1:8765/api/eq_agent \
  -H 'content-type: application/json' \
  -d '{"action":"start","mode":"ml","style":"balanced","seconds":2,"interval":1.0,"smooth_alpha":0.4,"apply":true}'
```

Live stream endpoint (used by UI):

- `GET /api/stream` (SSE, event name `snapshot`)

Provider health endpoint:

- `GET /api/provider_health` (checks env/config presence)
- `GET /api/provider_health?probe=1` (performs live low-cost API probe for OpenAI and Claude)

## Create GitHub Repo (after local init)

If GitHub CLI is available:

```bash
gh repo create eq-ai-tuner --public --source . --remote origin --push
```

Or manually:

```bash
git remote add origin git@github.com:<your-user>/eq-ai-tuner.git
git push -u origin main
```

## Safety Notes

- Start with low volume after applying new profiles.
- Prefer mild limits first (`+/-6 dB`).
- Re-run analysis on different content before locking in permanent settings.

## License

MIT. See `LICENSE`.

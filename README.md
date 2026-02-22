# EQ AI Tuner (2CH / 20-Band)

[![CI](https://github.com/ben2079/eq-ai-tuner/actions/workflows/ci.yml/badge.svg)](https://github.com/ben2079/eq-ai-tuner/actions/workflows/ci.yml)

Real-time stereo EQ tuning pipeline for Linux web audio (for example TIDAL in browser) with:

- `Manual` settings (per-channel, per-band)
- `Auto` settings (content-driven leveling)
- `AI-driven` settings (heuristic style engine + channel balancing)
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
  - Modes: `manual`, `auto`, `ai`
  - Optional apply to EasyEffects

- `live_server.py`
  - Lightweight HTTP server
  - Serves dashboard and `/api/snapshot`

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

- analyzer loop (`eq_tool.py --loop`)
- live web server (`live_server.py`)

## Manual vs Auto vs AI-driven

- `manual`
  - uses your fixed left/right gain arrays
  - good for controlled personal profiles

- `auto`
  - computes channel-specific correction from current stream energy
  - straightforward content leveling

- `ai`
  - heuristic target shaping by style plus cross-channel balancing
  - smoother, listening-oriented correction behavior

> Note: "AI-driven" here is on-device heuristic intelligence, not cloud ML inference.

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

## Contributing

See `CONTRIBUTING.md`.

## Community

- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security Policy: `SECURITY.md`
- Pull Request Template: `.github/pull_request_template.md`

## License

MIT. See `LICENSE`.

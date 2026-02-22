# Contributing

Thanks for your interest in contributing to EQ AI Tuner.

## Development Setup

1. Clone the repository.
2. Ensure Python 3.9+ is installed.
3. Install runtime dependencies on Linux:

```bash
sudo apt update
sudo apt install -y pulseaudio-utils easyeffects
```

## Coding Guidelines

- Keep code Python standard-library first when possible.
- Prefer small, focused changes and clear commit messages.
- Maintain 2-channel / 20-band behavior unless change is intentional.
- Keep UX responsive for live dashboard updates.

## Validation Before PR

Run these checks locally:

```bash
python3 -m py_compile eq_tool.py live_server.py eq20/core.py eq20/easyeffects.py eq20/profiles.py
python3 eq_tool.py --help
python3 live_server.py --help
```

## Pull Requests

- Describe what changed and why.
- Include screenshots for dashboard/UI updates.
- Mention test commands you ran.
- Keep PR scope focused.

## Issues

Use the issue templates for bug reports and feature requests.

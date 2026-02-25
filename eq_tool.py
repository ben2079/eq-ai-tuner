#!/usr/bin/env python3
import argparse
import os
import time

from eq20.core import analyze_stream, save_snapshot
from eq20.easyeffects import apply_split_20band
from eq20.profiles import load_manual_profile


def print_summary(result):
    print("Source:", result.source)
    print("Mode:", result.mode, "Style:", result.style)
    print("L gains:", ", ".join(f"{x:+.1f}" for x in result.gains_l_db))
    print("R gains:", ", ".join(f"{x:+.1f}" for x in result.gains_r_db))


def run_once(args):
    manual_l = manual_r = None
    if args.mode == "manual":
        manual_l, manual_r = load_manual_profile(args.manual_profile)

    res = analyze_stream(
        seconds=args.seconds,
        rate=args.rate,
        channels=args.channels,
        source=args.source,
        mode=args.mode,
        style=args.style,
        max_boost=args.max_boost,
        max_cut=args.max_cut,
        manual_l=manual_l,
        manual_r=manual_r,
        model_path=args.model_path,
        high_res=args.high_res,
    )

    os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
    save_snapshot(args.snapshot, res)
    print_summary(res)

    if args.apply:
        apply_split_20band(res.bands_hz, res.gains_l_db, res.gains_r_db)
        print("Applied to EasyEffects (split channels, 20-band).")


def run_loop(args):
    print("Starting continuous loop. Press Ctrl+C to stop.")
    while True:
        run_once(args)
        time.sleep(max(1, args.interval))


def build_parser():
    p = argparse.ArgumentParser(description="2CH/20-band EQ analyzer with manual, auto, AI, ML, Claude and OpenAI hybrid modes")
    p.add_argument(
        "--mode",
        choices=["manual", "auto", "ai", "ml", "claude_ml", "hybrid", "openai_ml", "hybrid_openai"],
        default="auto",
    )
    p.add_argument("--style", choices=["balanced", "bass", "clarity", "vocal"], default="balanced")
    p.add_argument("--manual-profile", default="configs/manual_profile.example.json")
    p.add_argument("--seconds", type=int, default=6)
    p.add_argument("--interval", type=int, default=3)
    p.add_argument("--rate", type=int, default=48000)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--source", default="")
    p.add_argument("--max-boost", type=float, default=12.0)
    p.add_argument("--max-cut", type=float, default=12.0)
    p.add_argument("--snapshot", default="data/latest_snapshot.json")
    p.add_argument("--model-path", default="data/ml_eq_model.json")
    p.add_argument("--high-res", action="store_true", help="Enable 20-band high-res frequency layout (requires >=96kHz)")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--loop", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.loop:
        run_loop(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()

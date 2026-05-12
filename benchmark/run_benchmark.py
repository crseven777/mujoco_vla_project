"""Run the first benchmark scaffold over recorded episodes.

This entry point intentionally evaluates the current recorded demonstrations
first. Real ACT/DP3/GR00T runners can plug into the same dataset and evaluator
interfaces once enough teleoperation data has been collected.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.datasets.dataset_adapter import discover_episodes, split_episodes
from benchmark.evaluators.evaluator import evaluate_recorded_episodes, placeholder_row


def _format_float(value: float) -> str:
    try:
        if value != value:
            return "N/A"
    except TypeError:
        return str(value)
    return f"{value:.6f}"


def _write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_markdown(rows) -> None:
    print("| Method | Success Rate | Avg Tracking Error | Avg Episode Length | Avg Inference Time | Smoothness | Notes |")
    print("|--------|--------------|--------------------|--------------------|--------------------|------------|-------|")
    for row in rows:
        print(
            "| "
            f"{row.method} | "
            f"{_format_float(row.success_rate)} | "
            f"{_format_float(row.avg_tracking_error)} | "
            f"{_format_float(row.avg_episode_length)} | "
            f"{_format_float(row.avg_inference_time_ms)} | "
            f"{_format_float(row.smoothness)} | "
            f"{row.notes} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark scaffold on recorded episodes")
    parser.add_argument("--data-root", type=str, default="data/samples")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="benchmark/results/benchmark_summary.csv")
    args = parser.parse_args()

    episodes = discover_episodes(args.data_root)
    if not episodes:
        raise SystemExit(f"No compatible episodes found under {args.data_root}")

    split = split_episodes(episodes)
    eval_set = split.test or split.val or split.train or episodes

    rows = [
        evaluate_recorded_episodes(eval_set, method="RecordedDemo", threshold=args.threshold),
        placeholder_row("ACT"),
        placeholder_row("DP3"),
        placeholder_row("GR00T"),
    ]

    _print_markdown(rows)
    _write_csv(rows, Path(args.output))
    print(f"\nSaved benchmark summary to: {args.output}")
    print(f"episodes: total={len(episodes)} train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

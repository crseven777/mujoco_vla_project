"""Benchmark evaluator for recorded upper-body tracking episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.datasets.dataset_adapter import load_episode
from benchmark.metrics.metrics import (
    average_episode_length,
    average_tracking_error,
    final_tracking_success,
    success_rate,
    trajectory_smoothness,
)


@dataclass
class BenchmarkRow:
    method: str
    success_rate: float
    avg_tracking_error: float
    avg_episode_length: float
    avg_inference_time_ms: float
    smoothness: float
    notes: str


def evaluate_recorded_episodes(
    episode_paths: list[Path],
    method: str = "RecordedDemo",
    threshold: float = 0.05,
) -> BenchmarkRow:
    successes: list[bool] = []
    errors: list[float] = []
    lengths: list[int] = []
    smoothness_values: list[float] = []

    for path in episode_paths:
        episode = load_episode(path)
        successes.append(
            final_tracking_success(
                episode.tracking_error_left,
                episode.tracking_error_right,
                threshold=threshold,
            )
        )
        errors.append(average_tracking_error(episode.tracking_error_left, episode.tracking_error_right))
        lengths.append(episode.length)
        smoothness_values.append(trajectory_smoothness(episode.action))

    return BenchmarkRow(
        method=method,
        success_rate=success_rate(successes),
        avg_tracking_error=float(np.nanmean(errors)) if errors else float("nan"),
        avg_episode_length=average_episode_length(lengths),
        avg_inference_time_ms=0.0,
        smoothness=float(np.nanmean(smoothness_values)) if smoothness_values else float("nan"),
        notes="computed from recorded tracking logs; no learned policy connected",
    )


def placeholder_row(method: str) -> BenchmarkRow:
    return BenchmarkRow(
        method=method,
        success_rate=0.0,
        avg_tracking_error=float("nan"),
        avg_episode_length=0.0,
        avg_inference_time_ms=float("nan"),
        smoothness=float("nan"),
        notes="runner interface exists; training/inference not connected yet",
    )

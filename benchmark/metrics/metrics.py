"""Common benchmark metrics for upper-body tracking tasks."""

from __future__ import annotations

import numpy as np


def average_tracking_error(error_left: np.ndarray, error_right: np.ndarray) -> float:
    values = np.concatenate([np.ravel(error_left), np.ravel(error_right)])
    valid = values[np.isfinite(values)]
    return float(np.mean(valid)) if valid.size else float("nan")


def final_tracking_success(error_left: np.ndarray, error_right: np.ndarray, threshold: float) -> bool:
    left = error_left[np.isfinite(error_left)]
    right = error_right[np.isfinite(error_right)]
    checks = []
    if left.size:
        checks.append(float(left[-1]) < threshold)
    if right.size:
        checks.append(float(right[-1]) < threshold)
    return bool(checks and all(checks))


def success_rate(success_flags: list[bool]) -> float:
    if not success_flags:
        return 0.0
    return float(np.mean(np.asarray(success_flags, dtype=float)))


def average_episode_length(lengths: list[int]) -> float:
    return float(np.mean(lengths)) if lengths else 0.0


def trajectory_smoothness(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=float)
    if actions.shape[0] < 2:
        return 0.0
    delta = np.diff(actions, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=-1)))

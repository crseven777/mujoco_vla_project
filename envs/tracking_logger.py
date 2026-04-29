"""Utilities for Stage-1 tracking logging and summary."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackingLogger:
    """Collect per-frame tracking signals and compute summary stats."""

    threshold_m: float = 0.05
    timestamps: list[float] = field(default_factory=list)
    target_eef_right: list[np.ndarray] = field(default_factory=list)
    actual_eef_right: list[np.ndarray] = field(default_factory=list)
    tracking_error_right: list[float] = field(default_factory=list)
    joint_state: list[np.ndarray] = field(default_factory=list)
    action_or_ctrl: list[np.ndarray] = field(default_factory=list)

    def record(
        self,
        timestamp: float,
        target_eef_right: np.ndarray,
        actual_eef_right: np.ndarray,
        joint_state: np.ndarray,
        action_or_ctrl: np.ndarray,
    ) -> None:
        err = float(np.linalg.norm(target_eef_right - actual_eef_right))
        self.timestamps.append(float(timestamp))
        self.target_eef_right.append(np.asarray(target_eef_right, dtype=float).copy())
        self.actual_eef_right.append(np.asarray(actual_eef_right, dtype=float).copy())
        self.tracking_error_right.append(err)
        self.joint_state.append(np.asarray(joint_state, dtype=float).copy())
        self.action_or_ctrl.append(np.asarray(action_or_ctrl, dtype=float).copy())

    def _arr(self, xs: list[Any], dtype=float) -> np.ndarray:
        if not xs:
            return np.array([], dtype=dtype)
        return np.asarray(xs, dtype=dtype)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": self._arr(self.timestamps, dtype=float),
            "target_eef_right": self._arr(self.target_eef_right, dtype=float),
            "actual_eef_right": self._arr(self.actual_eef_right, dtype=float),
            "tracking_error_right": self._arr(self.tracking_error_right, dtype=float),
            "joint_state": self._arr(self.joint_state, dtype=float),
            "action": self._arr(self.action_or_ctrl, dtype=float),
        }

    def summary(self) -> dict[str, Any]:
        errs = self._arr(self.tracking_error_right, dtype=float)
        if errs.size == 0:
            return {
                "num_frames": 0,
                "mean_tracking_error_m": math.nan,
                "max_tracking_error_m": math.nan,
                "final_tracking_error_m": math.nan,
                "within_threshold": False,
                "threshold_m": float(self.threshold_m),
                "has_nan": True,
                "has_divergence": True,
            }

        has_nan = bool(np.isnan(errs).any())
        mean_err = float(np.mean(errs))
        max_err = float(np.max(errs))
        final_err = float(errs[-1])
        within = bool(final_err <= self.threshold_m)
        has_div = bool(np.any(errs > 1.0) or np.any(np.diff(errs) > 0.5))

        return {
            "num_frames": int(errs.size),
            "mean_tracking_error_m": mean_err,
            "max_tracking_error_m": max_err,
            "final_tracking_error_m": final_err,
            "within_threshold": within,
            "threshold_m": float(self.threshold_m),
            "has_nan": has_nan,
            "has_divergence": has_div,
        }

    def save_np(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        arrays = self.as_arrays()
        np.save(os.path.join(output_dir, "joint_state.npy"), arrays["joint_state"])
        np.save(os.path.join(output_dir, "action.npy"), arrays["action"])
        np.save(os.path.join(output_dir, "target_eef.npy"), arrays["target_eef_right"])
        np.save(os.path.join(output_dir, "actual_eef.npy"), arrays["actual_eef_right"])
        np.save(os.path.join(output_dir, "tracking_error.npy"), arrays["tracking_error_right"])
        np.save(os.path.join(output_dir, "timestamps.npy"), arrays["timestamps"])

    def save_summary(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

"""Unified Stage-1 logger for bimanual tracking and RGBD capture."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import imageio.v3 as iio
import numpy as np


@dataclass
class Stage1Logger:
    output_dir: str
    save_rgbd: bool = True
    save_every_n_frames: int = 1

    timestamps: list[float] = field(default_factory=list)
    joint_state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    target_eef_left: list[np.ndarray] = field(default_factory=list)
    target_eef_right: list[np.ndarray] = field(default_factory=list)
    actual_eef_left: list[np.ndarray] = field(default_factory=list)
    actual_eef_right: list[np.ndarray] = field(default_factory=list)
    tracking_error_left: list[float] = field(default_factory=list)
    tracking_error_right: list[float] = field(default_factory=list)
    mode: list[str] = field(default_factory=list)
    camera_timestamps: list[float] = field(default_factory=list)
    camera_frame_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.depth_dir = os.path.join(self.output_dir, "depth")
        if self.save_rgbd:
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)

    def _as3(self, v):
        if v is None:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        return np.asarray(v, dtype=float).copy()

    def record(self, frame_idx: int, timestamp: float, mode: str, joint_state: np.ndarray, action: np.ndarray,
               target_left, target_right, actual_left, actual_right, err_left: float, err_right: float,
               camera_frame: dict | None = None):
        self.timestamps.append(float(timestamp))
        self.mode.append(mode)
        self.joint_state.append(np.asarray(joint_state, dtype=float).copy())
        self.action.append(np.asarray(action, dtype=float).copy())
        self.target_eef_left.append(self._as3(target_left))
        self.target_eef_right.append(self._as3(target_right))
        self.actual_eef_left.append(self._as3(actual_left))
        self.actual_eef_right.append(self._as3(actual_right))
        self.tracking_error_left.append(float(err_left) if err_left is not None else np.nan)
        self.tracking_error_right.append(float(err_right) if err_right is not None else np.nan)

        if self.save_rgbd and camera_frame is not None and frame_idx % max(1, self.save_every_n_frames) == 0:
            iio.imwrite(os.path.join(self.rgb_dir, f"{frame_idx:06d}.png"), camera_frame["rgb"])
            np.save(os.path.join(self.depth_dir, f"{frame_idx:06d}.npy"), camera_frame["depth"])
            self.camera_timestamps.append(float(camera_frame.get("timestamp", timestamp)))
            self.camera_frame_indices.append(frame_idx)

    def save(self, meta: dict, camera_meta: dict | None = None):
        np.save(os.path.join(self.output_dir, "joint_state.npy"), np.asarray(self.joint_state, dtype=float))
        np.save(os.path.join(self.output_dir, "action.npy"), np.asarray(self.action, dtype=float))
        np.save(os.path.join(self.output_dir, "target_eef_left.npy"), np.asarray(self.target_eef_left, dtype=float))
        np.save(os.path.join(self.output_dir, "target_eef_right.npy"), np.asarray(self.target_eef_right, dtype=float))
        np.save(os.path.join(self.output_dir, "actual_eef_left.npy"), np.asarray(self.actual_eef_left, dtype=float))
        np.save(os.path.join(self.output_dir, "actual_eef_right.npy"), np.asarray(self.actual_eef_right, dtype=float))
        np.save(os.path.join(self.output_dir, "tracking_error_left.npy"), np.asarray(self.tracking_error_left, dtype=float))
        np.save(os.path.join(self.output_dir, "tracking_error_right.npy"), np.asarray(self.tracking_error_right, dtype=float))
        np.save(os.path.join(self.output_dir, "timestamp.npy"), np.asarray(self.timestamps, dtype=float))

        if self.save_rgbd:
            np.save(os.path.join(self.output_dir, "camera_timestamps.npy"), np.asarray(self.camera_timestamps, dtype=float))
            np.save(os.path.join(self.output_dir, "camera_frame_indices.npy"), np.asarray(self.camera_frame_indices, dtype=int))
            if camera_meta is not None:
                with open(os.path.join(self.output_dir, "camera_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(camera_meta, f, indent=2)

        meta = dict(meta)
        meta["date"] = datetime.now().isoformat()
        with open(os.path.join(self.output_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def summarize_errors(err_left: np.ndarray, err_right: np.ndarray, threshold: float) -> dict:
        def _metrics(x):
            valid = x[np.isfinite(x)]
            if valid.size == 0:
                return {"mean": np.nan, "max": np.nan, "final": np.nan, "pass": False}
            return {
                "mean": float(np.mean(valid)),
                "max": float(np.max(valid)),
                "final": float(valid[-1]),
                "pass": bool(valid[-1] < threshold),
            }

        l = _metrics(np.asarray(err_left, dtype=float))
        r = _metrics(np.asarray(err_right, dtype=float))
        has_nan = bool(np.isnan(err_left).any() or np.isnan(err_right).any())
        has_divergence = bool(np.nanmax(err_left) > 1.0 or np.nanmax(err_right) > 1.0)
        return {
            "left": l,
            "right": r,
            "has_nan": has_nan,
            "has_divergence": has_divergence,
            "threshold": threshold,
        }

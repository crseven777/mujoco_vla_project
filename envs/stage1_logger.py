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
    left_wrist_pose: list[np.ndarray] = field(default_factory=list)
    right_wrist_pose: list[np.ndarray] = field(default_factory=list)
    raw_target_left: list[np.ndarray] = field(default_factory=list)
    raw_target_right: list[np.ndarray] = field(default_factory=list)
    transformed_target_left: list[np.ndarray] = field(default_factory=list)
    transformed_target_right: list[np.ndarray] = field(default_factory=list)
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

    def _as7(self, v):
        if v is None:
            return np.array([np.nan] * 7, dtype=float)
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size != 7:
            out = np.array([np.nan] * 7, dtype=float)
            out[: min(7, arr.size)] = arr[: min(7, arr.size)]
            return out
        return arr.copy()

    def record(self, frame_idx: int, timestamp: float, mode: str, joint_state: np.ndarray, action: np.ndarray,
               target_left, target_right, actual_left, actual_right, err_left: float, err_right: float,
               camera_frame: dict | None = None, left_wrist_pose=None, right_wrist_pose=None,
               raw_target_left=None, raw_target_right=None, transformed_target_left=None, transformed_target_right=None):
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
        self.left_wrist_pose.append(self._as7(left_wrist_pose))
        self.right_wrist_pose.append(self._as7(right_wrist_pose))
        self.raw_target_left.append(self._as3(raw_target_left))
        self.raw_target_right.append(self._as3(raw_target_right))
        self.transformed_target_left.append(self._as3(transformed_target_left))
        self.transformed_target_right.append(self._as3(transformed_target_right))

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
        np.save(os.path.join(self.output_dir, "left_wrist_pose.npy"), np.asarray(self.left_wrist_pose, dtype=float))
        np.save(os.path.join(self.output_dir, "right_wrist_pose.npy"), np.asarray(self.right_wrist_pose, dtype=float))
        np.save(os.path.join(self.output_dir, "raw_target_left.npy"), np.asarray(self.raw_target_left, dtype=float))
        np.save(os.path.join(self.output_dir, "raw_target_right.npy"), np.asarray(self.raw_target_right, dtype=float))
        np.save(os.path.join(self.output_dir, "transformed_target_left.npy"), np.asarray(self.transformed_target_left, dtype=float))
        np.save(os.path.join(self.output_dir, "transformed_target_right.npy"), np.asarray(self.transformed_target_right, dtype=float))
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
                return {"mean": np.nan, "max": np.nan, "final": np.nan, "pass": None, "active": False}
            return {
                "mean": float(np.mean(valid)),
                "max": float(np.max(valid)),
                "final": float(valid[-1]),
                "pass": bool(valid[-1] < threshold),
                "active": True,
            }

        left = np.asarray(err_left, dtype=float)
        right = np.asarray(err_right, dtype=float)
        l = _metrics(left)
        r = _metrics(right)

        def _has_nan_on_active_side(x):
            return bool(np.isfinite(x).any() and np.isnan(x).any())

        def _has_divergence(x):
            valid = x[np.isfinite(x)]
            return bool(valid.size > 0 and np.max(valid) > 1.0)

        has_nan = _has_nan_on_active_side(left) or _has_nan_on_active_side(right)
        has_divergence = _has_divergence(left) or _has_divergence(right)
        return {
            "left": l,
            "right": r,
            "has_nan": has_nan,
            "has_divergence": has_divergence,
            "threshold": threshold,
        }

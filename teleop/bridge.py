"""XR to MuJoCo bridge for end-effector target positions (position-only MVP)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Config for XR->MuJoCo position bridge."""

    # Linear mapping from XR position to MuJoCo world frame.
    xr_to_mj_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    xr_to_mj_trans: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Human-to-robot scale factor.
    arm_scale: float = 1.0

    # Body-center based offset alignment in MuJoCo frame.
    body_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # Workspace bounds in MuJoCo frame.
    left_workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.00, 0.70]))
    left_workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.55, 1.45]))
    right_workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.10, -0.55, 0.70]))
    right_workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.00, 1.45]))

    # Low-pass filtering: target = alpha * new + (1 - alpha) * prev
    lowpass_alpha: float = 0.25

    # Debug logging
    debug: bool = False
    debug_every_n: int = 20


class XRTeleopBridge:
    """Bridge module converting XR wrist poses to MuJoCo eef target positions."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.cfg = config if config is not None else BridgeConfig()
        self._validate_config()

        self._frame_idx = 0
        self._initialized = False

        self._ref_head_pos = None
        self._ref_left_pos = None
        self._ref_right_pos = None

        self._prev_left_target = None
        self._prev_right_target = None

        self._last_debug = {}

    def _validate_config(self) -> None:
        self.cfg.xr_to_mj_rot = np.asarray(self.cfg.xr_to_mj_rot, dtype=float).reshape(3, 3)
        self.cfg.xr_to_mj_trans = np.asarray(self.cfg.xr_to_mj_trans, dtype=float).reshape(3,)
        self.cfg.body_offset = np.asarray(self.cfg.body_offset, dtype=float).reshape(3,)

        self.cfg.left_workspace_min = np.asarray(self.cfg.left_workspace_min, dtype=float).reshape(3,)
        self.cfg.left_workspace_max = np.asarray(self.cfg.left_workspace_max, dtype=float).reshape(3,)
        self.cfg.right_workspace_min = np.asarray(self.cfg.right_workspace_min, dtype=float).reshape(3,)
        self.cfg.right_workspace_max = np.asarray(self.cfg.right_workspace_max, dtype=float).reshape(3,)

        self.cfg.lowpass_alpha = float(np.clip(self.cfg.lowpass_alpha, 0.0, 1.0))
        self.cfg.arm_scale = float(self.cfg.arm_scale)

    @staticmethod
    def _pose_to_pos(pose_4x4: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_4x4, dtype=float).reshape(4, 4)
        return pose[:3, 3].copy()

    def _map_xr_to_mj_pos(self, xr_pos: np.ndarray) -> np.ndarray:
        return self.cfg.xr_to_mj_rot @ xr_pos + self.cfg.xr_to_mj_trans

    def _apply_scale_about_ref(self, pos: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return ref + self.cfg.arm_scale * (pos - ref)

    @staticmethod
    def _clip_workspace(pos: np.ndarray, pmin: np.ndarray, pmax: np.ndarray) -> np.ndarray:
        return np.clip(pos, pmin, pmax)

    def _lowpass(self, new_pos: np.ndarray, prev_pos: Optional[np.ndarray]) -> np.ndarray:
        if prev_pos is None:
            return new_pos
        a = self.cfg.lowpass_alpha
        return a * new_pos + (1.0 - a) * prev_pos

    @staticmethod
    def _safe_finite(new_value: np.ndarray, fallback: Optional[np.ndarray]) -> np.ndarray:
        if np.all(np.isfinite(new_value)):
            return new_value
        if fallback is None:
            return np.zeros(3, dtype=float)
        return fallback

    def _init_reference(self, head_mj: np.ndarray, left_mj: np.ndarray, right_mj: np.ndarray) -> None:
        self._ref_head_pos = head_mj.copy()
        self._ref_left_pos = left_mj.copy()
        self._ref_right_pos = right_mj.copy()
        self._initialized = True

    def update(self, xr_state: Dict) -> Dict:
        """Convert xr_state to bridge_output.

        Expected xr_state fields:
          - timestamp
          - head_pose (4x4)
          - left_wrist_pose (4x4)
          - right_wrist_pose (4x4)
        """
        self._frame_idx += 1

        timestamp = float(xr_state.get("timestamp", 0.0))

        head_xr = self._pose_to_pos(xr_state["head_pose"])
        left_xr = self._pose_to_pos(xr_state["left_wrist_pose"])
        right_xr = self._pose_to_pos(xr_state["right_wrist_pose"])

        head_mj = self._map_xr_to_mj_pos(head_xr)
        left_mj = self._map_xr_to_mj_pos(left_xr)
        right_mj = self._map_xr_to_mj_pos(right_xr)

        if not self._initialized:
            self._init_reference(head_mj, left_mj, right_mj)

        # Body-center alignment: keep relative motion around initial head reference.
        head_delta = head_mj - self._ref_head_pos
        align_offset = self.cfg.body_offset + head_delta

        left_scaled = self._apply_scale_about_ref(left_mj, self._ref_left_pos)
        right_scaled = self._apply_scale_about_ref(right_mj, self._ref_right_pos)

        left_aligned = left_scaled + align_offset
        right_aligned = right_scaled + align_offset

        left_clipped = self._clip_workspace(left_aligned, self.cfg.left_workspace_min, self.cfg.left_workspace_max)
        right_clipped = self._clip_workspace(right_aligned, self.cfg.right_workspace_min, self.cfg.right_workspace_max)

        left_target = self._lowpass(left_clipped, self._prev_left_target)
        right_target = self._lowpass(right_clipped, self._prev_right_target)

        left_target = self._safe_finite(left_target, self._prev_left_target)
        right_target = self._safe_finite(right_target, self._prev_right_target)

        self._prev_left_target = left_target.copy()
        self._prev_right_target = right_target.copy()

        self._last_debug = {
            "frame": self._frame_idx,
            "timestamp": timestamp,
            "head_xr": head_xr,
            "left_xr": left_xr,
            "right_xr": right_xr,
            "left_mapped": left_mj,
            "right_mapped": right_mj,
            "left_aligned": left_aligned,
            "right_aligned": right_aligned,
            "left_target": left_target,
            "right_target": right_target,
        }

        if self.cfg.debug and (self._frame_idx % max(1, self.cfg.debug_every_n) == 0):
            LOGGER.info(
                "[bridge] t=%.3f | XR_R=%s | target_R=%s | XR_L=%s | target_L=%s",
                timestamp,
                np.array2string(right_xr, precision=3),
                np.array2string(right_target, precision=3),
                np.array2string(left_xr, precision=3),
                np.array2string(left_target, precision=3),
            )

        return {
            "timestamp": timestamp,
            "head_pose": np.asarray(xr_state["head_pose"], dtype=float).reshape(4, 4),
            "left_wrist_pose": np.asarray(xr_state["left_wrist_pose"], dtype=float).reshape(4, 4),
            "right_wrist_pose": np.asarray(xr_state["right_wrist_pose"], dtype=float).reshape(4, 4),
            "left_target_pos": left_target,
            "right_target_pos": right_target,
        }

    def get_debug_snapshot(self) -> Dict:
        return dict(self._last_debug)

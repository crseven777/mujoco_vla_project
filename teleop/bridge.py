"""Bridge layer: XR wrist poses -> MuJoCo dual-hand target positions.

Coordinate definition:
- XR frame: incoming wearable/controller tracking frame.
- Robot base frame: MuJoCo world/body frame used by controller targets.

Pose format:
- wrist pose is (7,) = [x, y, z, qw, qx, qy, qz]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4,)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = q / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def pose7_to_mat4(pose7: np.ndarray) -> np.ndarray:
    p = np.asarray(pose7, dtype=float).reshape(7,)
    t = p[:3]
    q = p[3:]
    m = np.eye(4, dtype=float)
    m[:3, :3] = _quat_wxyz_to_rot(q)
    m[:3, 3] = t
    return m


def mat4_to_pose7(mat4: np.ndarray) -> np.ndarray:
    m = np.asarray(mat4, dtype=float).reshape(4, 4)
    t = m[:3, 3]
    q = _rot_to_quat_wxyz(m[:3, :3])
    return np.concatenate([t, q], axis=0)


@dataclass
class BridgeConfig:
    # 4x4 rigid transform: XR frame -> robot base frame.
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    # independent xyz scale
    scale_xyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    # origin alignment
    xr_origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    robot_origin: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.0, 1.05]))
    # workspace clamp
    left_min_bound: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.02, 0.70]))
    left_max_bound: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.55, 1.45]))
    right_min_bound: np.ndarray = field(default_factory=lambda: np.array([0.10, -0.55, 0.70]))
    right_max_bound: np.ndarray = field(default_factory=lambda: np.array([0.75, -0.02, 1.45]))
    # smoothing alpha
    smoothing_alpha: float = 0.2
    debug: bool = False
    debug_every_n: int = 20


class XRTeleopBridge:
    """XR pose to bimanual MuJoCo target converter."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.cfg = config if config is not None else BridgeConfig()
        self._validate()
        self._frame = 0
        self._left_prev = None
        self._right_prev = None
        self._last_debug = {}

    def _validate(self) -> None:
        self.cfg.transform_matrix = np.asarray(self.cfg.transform_matrix, dtype=float).reshape(4, 4)
        self.cfg.scale_xyz = np.asarray(self.cfg.scale_xyz, dtype=float).reshape(3,)
        self.cfg.xr_origin = np.asarray(self.cfg.xr_origin, dtype=float).reshape(3,)
        self.cfg.robot_origin = np.asarray(self.cfg.robot_origin, dtype=float).reshape(3,)
        self.cfg.left_min_bound = np.asarray(self.cfg.left_min_bound, dtype=float).reshape(3,)
        self.cfg.left_max_bound = np.asarray(self.cfg.left_max_bound, dtype=float).reshape(3,)
        self.cfg.right_min_bound = np.asarray(self.cfg.right_min_bound, dtype=float).reshape(3,)
        self.cfg.right_max_bound = np.asarray(self.cfg.right_max_bound, dtype=float).reshape(3,)
        self.cfg.smoothing_alpha = float(np.clip(self.cfg.smoothing_alpha, 0.0, 1.0))

    def _transform_wrist_pose(self, wrist_pose7: np.ndarray) -> np.ndarray:
        wrist_T_xr = pose7_to_mat4(wrist_pose7)
        wrist_T_robot = self.cfg.transform_matrix @ wrist_T_xr
        return mat4_to_pose7(wrist_T_robot)

    def _scale_and_origin(self, pos_robot: np.ndarray) -> np.ndarray:
        centered = pos_robot - self.cfg.xr_origin
        scaled = self.cfg.scale_xyz * centered
        return scaled + self.cfg.robot_origin

    @staticmethod
    def _lowpass(new: np.ndarray, prev: Optional[np.ndarray], alpha: float) -> np.ndarray:
        if prev is None:
            return new
        return alpha * new + (1.0 - alpha) * prev

    @staticmethod
    def _finite_or_prev(new: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        if np.all(np.isfinite(new)):
            return new
        return np.zeros(3, dtype=float) if prev is None else prev

    def update(self, xr_state: Dict) -> Dict:
        self._frame += 1
        ts = float(xr_state.get("timestamp", 0.0))

        left_wrist_xr = np.asarray(xr_state["left_wrist_pose"], dtype=float).reshape(7,)
        right_wrist_xr = np.asarray(xr_state["right_wrist_pose"], dtype=float).reshape(7,)

        # 1) Coordinate transform (xr frame -> robot base frame)
        left_wrist_robot = self._transform_wrist_pose(left_wrist_xr)
        right_wrist_robot = self._transform_wrist_pose(right_wrist_xr)

        # 2) Scaling with origin alignment
        left_raw_target = self._scale_and_origin(left_wrist_robot[:3])
        right_raw_target = self._scale_and_origin(right_wrist_robot[:3])

        # 3) Workspace clamp
        left_clamped = np.clip(left_raw_target, self.cfg.left_min_bound, self.cfg.left_max_bound)
        right_clamped = np.clip(right_raw_target, self.cfg.right_min_bound, self.cfg.right_max_bound)

        # 4) Exponential smoothing
        left_target = self._lowpass(left_clamped, self._left_prev, self.cfg.smoothing_alpha)
        right_target = self._lowpass(right_clamped, self._right_prev, self.cfg.smoothing_alpha)
        left_target = self._finite_or_prev(left_target, self._left_prev)
        right_target = self._finite_or_prev(right_target, self._right_prev)
        self._left_prev = left_target.copy()
        self._right_prev = right_target.copy()

        self._last_debug = {
            "timestamp": ts,
            "left_wrist_robot_pose": left_wrist_robot.copy(),
            "right_wrist_robot_pose": right_wrist_robot.copy(),
            "left_raw_target": left_raw_target.copy(),
            "right_raw_target": right_raw_target.copy(),
            "left_target": left_target.copy(),
            "right_target": right_target.copy(),
        }

        if self.cfg.debug and self._frame % max(1, self.cfg.debug_every_n) == 0:
            LOGGER.info(
                "[bridge] t=%.3f | L_raw=%s -> L_tgt=%s | R_raw=%s -> R_tgt=%s",
                ts,
                np.array2string(left_raw_target, precision=3),
                np.array2string(left_target, precision=3),
                np.array2string(right_raw_target, precision=3),
                np.array2string(right_target, precision=3),
            )

        return {
            "timestamp": ts,
            "left_wrist_pose": left_wrist_robot,   # shape (7,)
            "right_wrist_pose": right_wrist_robot, # shape (7,)
            "left_target_pos": left_target,        # shape (3,)
            "right_target_pos": right_target,      # shape (3,)
            "raw_left_target_pos": left_raw_target,
            "raw_right_target_pos": right_raw_target,
        }

    def get_debug_snapshot(self) -> Dict:
        return dict(self._last_debug)

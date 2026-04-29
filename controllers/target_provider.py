"""Unified target provider for static and trajectory Stage-1 modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.trajectory_generator import TrajectoryGenerator


@dataclass
class TargetProviderConfig:
    mode: str = "trajectory_bimanual"
    trajectory_type: str = "circle"
    right_target_center: tuple[float, float, float] = (0.35, -0.22, 1.08)
    left_target_center: tuple[float, float, float] = (0.35, 0.22, 1.08)
    right_amplitude: tuple[float, float, float] = (0.06, 0.06, 0.04)
    left_amplitude: tuple[float, float, float] = (0.06, 0.06, 0.04)
    right_radius: float = 0.08
    left_radius: float = 0.08
    frequency: float = 0.12
    right_axis: str = "xy"
    left_axis: str = "xy"
    right_phase: float = 0.0
    left_phase: float = np.pi


class TargetProvider:
    """Provide left/right end-effector targets for all stage-1 modes."""

    def __init__(self, cfg: TargetProviderConfig):
        self.cfg = cfg

    def _traj(self, hand: str, t: float) -> np.ndarray:
        if hand == "right":
            center = np.asarray(self.cfg.right_target_center, dtype=float)
            amp = np.asarray(self.cfg.right_amplitude, dtype=float)
            radius = float(self.cfg.right_radius)
            axis = self.cfg.right_axis
            phase = float(self.cfg.right_phase)
        else:
            center = np.asarray(self.cfg.left_target_center, dtype=float)
            amp = np.asarray(self.cfg.left_amplitude, dtype=float)
            radius = float(self.cfg.left_radius)
            axis = self.cfg.left_axis
            phase = float(self.cfg.left_phase)

        tr = self.cfg.trajectory_type
        if tr == "line":
            return TrajectoryGenerator.line(t, center=center, amplitude=amp, frequency=self.cfg.frequency, axis=axis, phase=phase)
        if tr == "circle":
            return TrajectoryGenerator.circle(t, center=center, radius=radius, frequency=self.cfg.frequency, axis=axis, phase=phase)
        if tr in ("sin", "sinusoidal", "figure8", "figure-eight"):
            return TrajectoryGenerator.sinusoidal(t, center=center, amplitude=amp, frequency=self.cfg.frequency, axis=axis, phase=phase)
        return TrajectoryGenerator.circle(t, center=center, radius=radius, frequency=self.cfg.frequency, axis=axis, phase=phase)

    def get_target(self, timestamp: float) -> dict:
        m = self.cfg.mode
        left_target = None
        right_target = None

        if m == "static_right":
            right_target = np.asarray(self.cfg.right_target_center, dtype=float)
        elif m == "static_left":
            left_target = np.asarray(self.cfg.left_target_center, dtype=float)
        elif m == "static_bimanual":
            right_target = np.asarray(self.cfg.right_target_center, dtype=float)
            left_target = np.asarray(self.cfg.left_target_center, dtype=float)
        elif m == "trajectory_right":
            right_target = self._traj("right", timestamp)
        elif m == "trajectory_left":
            left_target = self._traj("left", timestamp)
        elif m == "trajectory_bimanual":
            right_target = self._traj("right", timestamp)
            left_target = self._traj("left", timestamp)

        return {
            "timestamp": float(timestamp),
            "left_target_pos": left_target,
            "right_target_pos": right_target,
            "mode": m,
        }

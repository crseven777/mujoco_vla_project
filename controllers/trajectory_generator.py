"""Trajectory generators for Stage-1 end-effector targets."""

from __future__ import annotations

import numpy as np


class TrajectoryGenerator:
    """Generate smooth 3D trajectories with safe defaults for upper-body reach."""

    @staticmethod
    def line(
        t: float,
        center: np.ndarray,
        amplitude: np.ndarray,
        frequency: float = 0.15,
        axis: str = "x",
        phase: float = 0.0,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=float)
        amp = np.asarray(amplitude, dtype=float)
        s = np.sin(2.0 * np.pi * frequency * t + phase)
        pos = center.copy()

        if axis == "x":
            pos[0] = center[0] + amp[0] * s
        elif axis == "y":
            pos[1] = center[1] + amp[1] * s
        elif axis == "z":
            pos[2] = center[2] + amp[2] * s
        elif axis == "xy":
            pos[0] = center[0] + amp[0] * s
            pos[1] = center[1] + amp[1] * s
        else:
            pos[0] = center[0] + amp[0] * s
        return pos

    @staticmethod
    def circle(
        t: float,
        center: np.ndarray,
        radius: float = 0.08,
        frequency: float = 0.12,
        axis: str = "xy",
        phase: float = 0.0,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=float)
        w = 2.0 * np.pi * frequency * t + phase
        c = np.cos(w)
        s = np.sin(w)
        pos = center.copy()

        if axis == "xy":
            pos[0] = center[0] + radius * c
            pos[1] = center[1] + radius * s
        elif axis == "xz":
            pos[0] = center[0] + radius * c
            pos[2] = center[2] + radius * s
        elif axis == "yz":
            pos[1] = center[1] + radius * c
            pos[2] = center[2] + radius * s
        else:
            pos[0] = center[0] + radius * c
            pos[1] = center[1] + radius * s
        return pos

    @staticmethod
    def sinusoidal(
        t: float,
        center: np.ndarray,
        amplitude: np.ndarray,
        frequency: float = 0.15,
        axis: str = "xy",
        phase: float = 0.0,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=float)
        amp = np.asarray(amplitude, dtype=float)
        w = 2.0 * np.pi * frequency * t + phase
        pos = center.copy()

        if axis == "xy":
            pos[0] = center[0] + amp[0] * np.sin(w)
            pos[1] = center[1] + amp[1] * np.sin(2.0 * w)
        elif axis == "xz":
            pos[0] = center[0] + amp[0] * np.sin(w)
            pos[2] = center[2] + amp[2] * np.sin(2.0 * w)
        elif axis == "yz":
            pos[1] = center[1] + amp[1] * np.sin(w)
            pos[2] = center[2] + amp[2] * np.sin(2.0 * w)
        else:
            pos[0] = center[0] + amp[0] * np.sin(w)
            pos[1] = center[1] + amp[1] * np.sin(2.0 * w)
        return pos

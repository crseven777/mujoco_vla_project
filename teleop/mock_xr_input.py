"""Mock XR pose generator for bridge validation (no XR hardware)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class MockXRConfig:
    base_head: tuple[float, float, float] = (0.0, 0.0, 1.55)
    base_left: tuple[float, float, float] = (-0.22, 0.18, 0.95)
    base_right: tuple[float, float, float] = (0.22, -0.18, 0.95)
    right_radius: float = 0.12
    left_amp: tuple[float, float, float] = (0.04, 0.03, 0.03)
    freq_hz: float = 0.15


class MockXRInput:
    """Generate wrist poses as pose7 [x, y, z, qw, qx, qy, qz]."""

    def __init__(self, cfg: MockXRConfig | None = None):
        self.cfg = cfg or MockXRConfig()
        self._t0 = time.time()

    @staticmethod
    def _pose7(x: float, y: float, z: float, qw: float = 1.0, qx: float = 0.0, qy: float = 0.0, qz: float = 0.0):
        return np.array([x, y, z, qw, qx, qy, qz], dtype=float)

    def read_state(self) -> dict:
        t = time.time() - self._t0
        w = 2.0 * np.pi * self.cfg.freq_hz

        hx, hy, hz = self.cfg.base_head
        lx, ly, lz = self.cfg.base_left
        rx, ry, rz = self.cfg.base_right

        # Right wrist: circular motion in XY + small Z oscillation.
        right_pos = np.array(
            [
                rx + self.cfg.right_radius * np.cos(w * t),
                ry + self.cfg.right_radius * np.sin(w * t),
                rz + 0.04 * np.sin(0.7 * w * t),
            ],
            dtype=float,
        )

        # Left wrist: sinusoidal line-like motion.
        ax, ay, az = self.cfg.left_amp
        left_pos = np.array(
            [
                lx + ax * np.sin(w * t),
                ly + ay * np.sin(0.6 * w * t + np.pi / 2.0),
                lz + az * np.cos(0.8 * w * t),
            ],
            dtype=float,
        )

        head_pos = np.array([hx + 0.01 * np.sin(0.3 * w * t), hy, hz], dtype=float)

        return {
            "timestamp": time.time(),
            "head_pose": self._pose7(head_pos[0], head_pos[1], head_pos[2]),
            "left_wrist_pose": self._pose7(left_pos[0], left_pos[1], left_pos[2]),
            "right_wrist_pose": self._pose7(right_pos[0], right_pos[1], right_pos[2]),
        }

    def close(self) -> None:
        return None

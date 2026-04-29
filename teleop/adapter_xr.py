"""XR adapter to read head/wrist poses from xr_teleoperate without modifying its source."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class XRAdapterConfig:
    # Explicit path to xr_teleoperate repo root (contains teleop/).
    xr_repo_root: str = "/home/wll/xr_teleoperate"

    # TeleVuerWrapper args.
    use_hand_tracking: bool = True
    binocular: bool = True
    img_shape: tuple = (480, 1280)
    display_fps: float = 30.0
    display_mode: str = "pass-through"
    zmq: bool = False
    webrtc: bool = False
    webrtc_url: Optional[str] = None


class XRAdapter:
    """Pull-based adapter returning normalized xr_state dict."""

    def __init__(self, config: Optional[XRAdapterConfig] = None):
        self.cfg = config if config is not None else XRAdapterConfig()
        self._wrapper = None

    def start(self) -> None:
        # Lazy import to avoid hard dependency during mock-only tests.
        import sys

        teleop_dir = os.path.join(self.cfg.xr_repo_root, "teleop")
        if teleop_dir not in sys.path:
            sys.path.insert(0, teleop_dir)

        from televuer import TeleVuerWrapper  # noqa: WPS433

        self._wrapper = TeleVuerWrapper(
            use_hand_tracking=self.cfg.use_hand_tracking,
            binocular=self.cfg.binocular,
            img_shape=self.cfg.img_shape,
            display_fps=self.cfg.display_fps,
            display_mode=self.cfg.display_mode,
            zmq=self.cfg.zmq,
            webrtc=self.cfg.webrtc,
            webrtc_url=self.cfg.webrtc_url,
        )

    def read_state(self) -> Dict:
        if self._wrapper is None:
            raise RuntimeError("XRAdapter is not started. Call start() first.")

        tele_data = self._wrapper.get_tele_data()

        xr_state = {
            "timestamp": time.time(),
            "head_pose": np.asarray(tele_data.head_pose, dtype=float).reshape(4, 4),
            "left_wrist_pose": np.asarray(tele_data.left_wrist_pose, dtype=float).reshape(4, 4),
            "right_wrist_pose": np.asarray(tele_data.right_wrist_pose, dtype=float).reshape(4, 4),
        }
        return xr_state

    def close(self) -> None:
        if self._wrapper is not None:
            self._wrapper.close()
            self._wrapper = None

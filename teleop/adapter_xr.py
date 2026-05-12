"""XR adapter to read head/wrist poses from xr_teleoperate without modifying its source."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from teleop.bridge import mat4_to_pose7


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
    extra_python_paths: tuple[str, ...] = (
        "/home/wll/miniconda3/envs/tv/lib/python3.10/site-packages",
    )


class XRAdapter:
    """Pull-based adapter returning normalized xr_state dict."""

    def __init__(self, config: Optional[XRAdapterConfig] = None):
        self.cfg = config if config is not None else XRAdapterConfig()
        self._wrapper = None

    def start(self) -> None:
        # Lazy import to avoid hard dependency during mock-only tests.
        import sys

        teleop_dir = os.path.join(self.cfg.xr_repo_root, "teleop")
        televuer_src_dir = os.path.join(teleop_dir, "televuer", "src")
        paths = (teleop_dir, televuer_src_dir, *self.cfg.extra_python_paths)
        for path in paths:
            if not os.path.exists(path):
                continue
            if path not in sys.path:
                sys.path.insert(0, path)

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
        head_pose = np.asarray(tele_data.head_pose, dtype=float).reshape(4, 4)
        left_wrist_pose = np.asarray(tele_data.left_wrist_pose, dtype=float).reshape(4, 4)
        right_wrist_pose = np.asarray(tele_data.right_wrist_pose, dtype=float).reshape(4, 4)

        xr_state = {
            "timestamp": time.time(),
            "head_pose": mat4_to_pose7(head_pose),
            "left_wrist_pose": mat4_to_pose7(left_wrist_pose),
            "right_wrist_pose": mat4_to_pose7(right_wrist_pose),
            "right_ctrl_trigger": bool(getattr(tele_data, "right_ctrl_trigger", False)),
            "right_ctrl_trigger_value": float(getattr(tele_data, "right_ctrl_triggerValue", 10.0)),
            "right_ctrl_squeeze": bool(getattr(tele_data, "right_ctrl_squeeze", False)),
            "right_ctrl_squeeze_value": float(getattr(tele_data, "right_ctrl_squeezeValue", 0.0)),
            "right_ctrl_thumbstick_value": np.asarray(
                getattr(tele_data, "right_ctrl_thumbstickValue", np.zeros(2)),
                dtype=float,
            ).copy(),
            "left_ctrl_trigger": bool(getattr(tele_data, "left_ctrl_trigger", False)),
            "left_ctrl_trigger_value": float(getattr(tele_data, "left_ctrl_triggerValue", 10.0)),
            "left_ctrl_squeeze": bool(getattr(tele_data, "left_ctrl_squeeze", False)),
            "left_ctrl_squeeze_value": float(getattr(tele_data, "left_ctrl_squeezeValue", 0.0)),
            "left_ctrl_thumbstick_value": np.asarray(
                getattr(tele_data, "left_ctrl_thumbstickValue", np.zeros(2)),
                dtype=float,
            ).copy(),
        }
        return xr_state

    def close(self) -> None:
        if self._wrapper is not None:
            if hasattr(self._wrapper, "close"):
                self._wrapper.close()
            self._wrapper = None

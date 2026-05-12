"""Dataset adapter for recorded MuJoCo upper-body episodes.

This module defines the stable benchmark data contract before any policy
implementation is connected. It reads the stage-1/stage-2 episode format and
exposes unified observations/actions for ACT, DP3, and GR00T runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class EpisodeData:
    path: Path
    joint_state: np.ndarray
    action: np.ndarray
    target_eef_left: np.ndarray
    target_eef_right: np.ndarray
    actual_eef_left: np.ndarray
    actual_eef_right: np.ndarray
    tracking_error_left: np.ndarray
    tracking_error_right: np.ndarray
    timestamp: np.ndarray

    @property
    def length(self) -> int:
        return int(self.timestamp.shape[0])


@dataclass(frozen=True)
class DatasetSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


REQUIRED_FILES = (
    "joint_state.npy",
    "action.npy",
    "target_eef_left.npy",
    "target_eef_right.npy",
    "actual_eef_left.npy",
    "actual_eef_right.npy",
    "tracking_error_left.npy",
    "tracking_error_right.npy",
    "timestamp.npy",
)


def is_episode_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_FILES)


def discover_episodes(root: str | Path) -> list[Path]:
    root = Path(root)
    if is_episode_dir(root):
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_episode_dir(path))


def split_episodes(episodes: list[Path], train: int = 35, val: int = 5, test: int = 10) -> DatasetSplit:
    episodes = sorted(episodes)
    return DatasetSplit(
        train=episodes[:train],
        val=episodes[train : train + val],
        test=episodes[train + val : train + val + test],
    )


def load_episode(path: str | Path) -> EpisodeData:
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Episode {path} is missing files: {missing}")

    return EpisodeData(
        path=path,
        joint_state=np.load(path / "joint_state.npy"),
        action=np.load(path / "action.npy"),
        target_eef_left=np.load(path / "target_eef_left.npy"),
        target_eef_right=np.load(path / "target_eef_right.npy"),
        actual_eef_left=np.load(path / "actual_eef_left.npy"),
        actual_eef_right=np.load(path / "actual_eef_right.npy"),
        tracking_error_left=np.load(path / "tracking_error_left.npy"),
        tracking_error_right=np.load(path / "tracking_error_right.npy"),
        timestamp=np.load(path / "timestamp.npy"),
    )


def iter_observation_action(episode: EpisodeData) -> Iterator[tuple[dict, np.ndarray]]:
    rgb_dir = episode.path / "rgb"
    depth_dir = episode.path / "depth"
    for idx in range(episode.length):
        obs = {
            "timestamp": float(episode.timestamp[idx]),
            "joint_state": episode.joint_state[idx],
            "target_eef_left": episode.target_eef_left[idx],
            "target_eef_right": episode.target_eef_right[idx],
            "actual_eef_left": episode.actual_eef_left[idx],
            "actual_eef_right": episode.actual_eef_right[idx],
            "rgb_path": rgb_dir / f"{idx:06d}.png",
            "depth_path": depth_dir / f"{idx:06d}.npy",
            "instruction": "follow left and right end-effector targets",
        }
        yield obs, episode.action[idx]

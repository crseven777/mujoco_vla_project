"""Validate one recorded episode against the stage-3 data contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "meta.json",
    "state.npy",
    "action.npy",
    "target_eef.npy",
    "actual_eef.npy",
    "timestamps.npy",
    "success.txt",
)

REQUIRED_META_KEYS = (
    "task_name",
    "instruction",
    "robot_type",
    "sampling_rate_hz",
    "episode_length",
    "operator_id",
    "date",
)


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def validate_episode(path: Path, require_rgbd: bool = False) -> dict:
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    with (path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    missing_meta = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing_meta:
        raise ValueError(f"meta.json missing keys: {missing_meta}")

    state = _load_array(path / "state.npy")
    action = _load_array(path / "action.npy")
    target_eef = _load_array(path / "target_eef.npy")
    actual_eef = _load_array(path / "actual_eef.npy")
    timestamps = _load_array(path / "timestamps.npy")

    n = int(timestamps.shape[0])
    arrays = {
        "state.npy": state,
        "action.npy": action,
        "target_eef.npy": target_eef,
        "actual_eef.npy": actual_eef,
    }
    bad_lengths = {name: arr.shape[0] for name, arr in arrays.items() if arr.shape[0] != n}
    if bad_lengths:
        raise ValueError(f"Array lengths do not match timestamps length {n}: {bad_lengths}")

    rgb_files = sorted((path / "rgb").glob("*.png")) if (path / "rgb").exists() else []
    depth_files = sorted((path / "depth").glob("*.npy")) if (path / "depth").exists() else []
    if require_rgbd and (not rgb_files or not depth_files):
        raise ValueError("RGBD is required but rgb/ or depth/ frames are missing")
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"RGB/depth frame count mismatch: rgb={len(rgb_files)} depth={len(depth_files)}")

    with (path / "success.txt").open("r", encoding="utf-8") as f:
        success_value = f.read().strip()
    if success_value not in {"0", "1"}:
        raise ValueError("success.txt must contain 0 or 1")

    return {
        "path": str(path),
        "steps": n,
        "state_shape": tuple(state.shape),
        "action_shape": tuple(action.shape),
        "target_eef_shape": tuple(target_eef.shape),
        "actual_eef_shape": tuple(actual_eef.shape),
        "rgb_frames": len(rgb_files),
        "depth_frames": len(depth_files),
        "success": success_value == "1",
        "instruction": meta["instruction"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a recorded episode format")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--require-rgbd", action="store_true")
    args = parser.parse_args()

    summary = validate_episode(args.episode_dir, require_rgbd=args.require_rgbd)
    print("Episode format OK")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

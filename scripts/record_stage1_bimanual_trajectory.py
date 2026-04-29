from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stage1_pipeline import load_simple_yaml, run_stage1


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Stage1 bimanual trajectory sample")
    parser.add_argument("--config", type=str, default="configs/stage1_bimanual_trajectory.yaml")
    parser.add_argument("--output-dir", type=str, default="data/samples/stage1_bimanual_trajectory")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    cfg["target_mode"] = "trajectory_bimanual"
    cfg["output_dir"] = args.output_dir
    cfg["save_rgbd"] = True

    result = run_stage1(cfg, output_dir=args.output_dir, record_rgbd=True)
    s = result.summary

    print("=== Stage1 Bimanual Trajectory Record ===")
    print(f"output_dir: {result.output_dir}")
    print(f"episode length: {result.episode_length}")
    print(f"saved frame count: {result.saved_frame_count}")
    print(f"left mean/max/final (m): {s['left']['mean']:.6f} / {s['left']['max']:.6f} / {s['left']['final']:.6f}")
    print(f"right mean/max/final (m): {s['right']['mean']:.6f} / {s['right']['max']:.6f} / {s['right']['final']:.6f}")
    print(f"threshold pass left/right: {s['left']['pass']} / {s['right']['pass']}")
    print(f"has NaN: {s['has_nan']} | has divergence: {s['has_divergence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Visualize recorded episode data.
"""
import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio


def load_episode(episode_dir):
    """Load episode data from directory."""
    # Load metadata
    with open(os.path.join(episode_dir, "meta.json"), 'r') as f:
        metadata = json.load(f)
    
    # Load state data
    state = np.load(os.path.join(episode_dir, "state.npy"))
    target_eef = np.load(os.path.join(episode_dir, "target_eef.npy"))
    actual_eef = np.load(os.path.join(episode_dir, "actual_eef.npy"))
    timestamps = np.load(os.path.join(episode_dir, "timestamps.npy"))
    
    # Load camera params if available
    camera_intrinsics = None
    camera_extrinsics = None
    intrinsics_path = os.path.join(episode_dir, "camera_intrinsics.npy")
    extrinsics_path = os.path.join(episode_dir, "camera_extrinsics.npy")
    if os.path.exists(intrinsics_path):
        camera_intrinsics = np.load(intrinsics_path)
        camera_extrinsics = np.load(extrinsics_path)
    
    # Find RGB and depth images
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")
    
    rgb_images = []
    depth_images = []
    
    if os.path.exists(rgb_dir):
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        rgb_images = [os.path.join(rgb_dir, f) for f in rgb_files]
    
    if os.path.exists(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        depth_images = [os.path.join(depth_dir, f) for f in depth_files]
    
    return {
        'metadata': metadata,
        'state': state,
        'target_eef': target_eef,
        'actual_eef': actual_eef,
        'timestamps': timestamps,
        'camera_intrinsics': camera_intrinsics,
        'camera_extrinsics': camera_extrinsics,
        'rgb_images': rgb_images,
        'depth_images': depth_images
    }


def visualize_episode(episode_dir):
    """Visualize episode data."""
    print(f"Loading episode from: {episode_dir}")
    data = load_episode(episode_dir)

    # Basic alignment checks
    num_steps = len(data['timestamps'])
    if data['state'].shape[0] != num_steps:
        raise ValueError(f"state length {data['state'].shape[0]} != timestamps length {num_steps}")
    if data['target_eef'].shape[0] != num_steps:
        raise ValueError(f"target_eef length {data['target_eef'].shape[0]} != timestamps length {num_steps}")
    if data['actual_eef'].shape[0] != num_steps:
        raise ValueError(f"actual_eef length {data['actual_eef'].shape[0]} != timestamps length {num_steps}")
    
    # Print metadata
    print("\n" + "=" * 60)
    print("Episode Metadata")
    print("=" * 60)
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Joint positions over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(data['timestamps'], data['state'][:, :7])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Joint Position (rad)')
    ax1.set_title('Right Arm Joint Positions')
    ax1.legend([f'J{i+1}' for i in range(7)], loc='upper right', fontsize=8)
    ax1.grid(True)
    
    # Plot 2: End-effector position (3D trajectory)
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    ax2.plot(data['actual_eef'][:, 0], data['actual_eef'][:, 1], 
             data['actual_eef'][:, 2], 'b-', label='Actual')
    ax2.plot(data['target_eef'][:, 0], data['target_eef'][:, 1], 
             data['target_eef'][:, 2], 'r--', label='Target')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('End-Effector Trajectory')
    ax2.legend()
    
    # Plot 3: Tracking error over time
    ax3 = plt.subplot(2, 3, 3)
    error = np.linalg.norm(data['actual_eef'] - data['target_eef'], axis=1)
    ax3.plot(data['timestamps'], error)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Position Tracking Error')
    ax3.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4-6: Sample RGB and depth images (if available)
    if data['rgb_images']:
        # Load middle frame
        mid_idx = len(data['rgb_images']) // 2
        rgb_img = iio.imread(data['rgb_images'][mid_idx])
        
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(rgb_img)
        ax4.set_title(f'Sample RGB Image (Frame {mid_idx})')
        ax4.axis('off')
    
    if data['depth_images']:
        mid_idx = len(data['depth_images']) // 2
        depth_img = np.load(data['depth_images'][mid_idx])
        
        ax5 = plt.subplot(2, 3, 5)
        # Handle invalid depth values
        depth_img = np.where(np.isfinite(depth_img), depth_img, 0)
        im = ax5.imshow(depth_img, cmap='viridis')
        ax5.set_title(f'Sample Depth Image (Frame {mid_idx})')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    
    # Plot 6: Camera intrinsics
    ax6 = plt.subplot(2, 3, 6)
    if data['camera_intrinsics'] is not None:
        if data['camera_intrinsics'].ndim == 3:
            intr = data['camera_intrinsics'][len(data['camera_intrinsics']) // 2]
        else:
            intr = data['camera_intrinsics']
        intrinsics_text = f"Camera Intrinsics:\n"
        intrinsics_text += f"fx: {intr[0, 0]:.2f}\n"
        intrinsics_text += f"fy: {intr[1, 1]:.2f}\n"
        intrinsics_text += f"cx: {intr[0, 2]:.2f}\n"
        intrinsics_text += f"cy: {intr[1, 2]:.2f}"
        ax6.text(0.1, 0.5, intrinsics_text, fontsize=12, family='monospace')
    else:
        ax6.text(0.1, 0.5, "No camera data available", fontsize=12)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Camera Parameters')
    
    plt.tight_layout()
    plt.savefig(os.path.join(episode_dir, "visualization.png"), dpi=150)
    print(f"\nVisualization saved to: {os.path.join(episode_dir, 'visualization.png')}")
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Episode Statistics")
    print("=" * 60)
    print(f"  Duration: {data['timestamps'][-1]:.2f}s")
    print(f"  Mean tracking error: {np.mean(error):.4f}m")
    print(f"  Max tracking error: {np.max(error):.4f}m")
    print(f"  Final tracking error: {error[-1]:.4f}m")
    print(f"  Converged (< 1cm): {np.any(error < 0.01)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded episode data")
    parser.add_argument("episode_dir", type=str, nargs="?", 
                       help="Path to episode directory")
    parser.add_argument("--latest", action="store_true",
                       help="Use latest episode in data/raw")
    
    args = parser.parse_args()
    
    if args.latest:
        # Find latest episode
        data_dir = "data/raw"
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist")
            return
        
        episodes = [d for d in os.listdir(data_dir) if d.startswith("episode_")]
        if not episodes:
            print(f"Error: No episodes found in {data_dir}")
            return
        
        episodes.sort()
        episode_dir = os.path.join(data_dir, episodes[-1])
    elif args.episode_dir:
        episode_dir = args.episode_dir
    else:
        print("Error: Please specify --episode_dir or --latest")
        return
    
    visualize_episode(episode_dir)


if __name__ == "__main__":
    main()

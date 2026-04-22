"""
Data recorder for saving simulation data.
"""
import os
import json
import numpy as np
from datetime import datetime


class DataRecorder:
    """Recorder for saving episode data."""
    
    def __init__(self, output_dir="data/raw"):
        """
        Initialize data recorder.
        
        Args:
            output_dir: Base directory for saving data
        """
        self.output_dir = output_dir
        self.episode_dir = None
        self.episode_count = 0
        self.data = {
            'frame_indices': [],
            'timestamps': [],
            'joint_positions': [],
            'joint_velocities': [],
            'target_eef_positions': [],
            'actual_eef_positions': [],
            'camera_timestamps': [],
            'camera_intrinsics': [],
            'camera_extrinsics': []
        }
        self.metadata = {}
    
    def start_episode(self, task_name="eef_tracking", instruction="Track target position"):
        """Start a new episode."""
        self.episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = os.path.join(
            self.output_dir, 
            f"episode_{self.episode_count:03d}_{timestamp}"
        )
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(os.path.join(self.episode_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.episode_dir, "depth"), exist_ok=True)
        
        # Reset data
        self.data = {
            'frame_indices': [],
            'timestamps': [],
            'joint_positions': [],
            'joint_velocities': [],
            'target_eef_positions': [],
            'actual_eef_positions': [],
            'camera_timestamps': [],
            'camera_intrinsics': [],
            'camera_extrinsics': []
        }
        
        # Metadata
        self.metadata = {
            'task_name': task_name,
            'instruction': instruction,
            'robot_type': 'g1_29dof',
            'timestamp': timestamp,
            'start_time': datetime.now().isoformat(),
            'episode_id': self.episode_count
        }
        
        print(f"Started episode {self.episode_count}, saving to {self.episode_dir}")
    
    def record(self, timestamp, joint_pos, joint_vel, target_eef, actual_eef,
               rgb_image=None, depth_image=None, camera_params=None):
        """Record one timestep of data."""
        frame_idx = len(self.data['timestamps'])
        self.data['frame_indices'].append(frame_idx)
        self.data['timestamps'].append(timestamp)
        self.data['joint_positions'].append(joint_pos.copy())
        self.data['joint_velocities'].append(joint_vel.copy())
        self.data['target_eef_positions'].append(target_eef.copy())
        self.data['actual_eef_positions'].append(actual_eef.copy())

        # Save images with frame index aligned to state/target/eef rows.
        if rgb_image is not None:
            rgb_path = os.path.join(self.episode_dir, "rgb", f"{frame_idx:06d}.png")
            import imageio.v3 as iio
            iio.imwrite(rgb_path, rgb_image)

        if depth_image is not None:
            depth_path = os.path.join(self.episode_dir, "depth", f"{frame_idx:06d}.npy")
            np.save(depth_path, depth_image)

        if camera_params is not None:
            self.data['camera_intrinsics'].append(camera_params['intrinsics'].copy())
            self.data['camera_extrinsics'].append(camera_params['extrinsics'].copy())
            self.data['camera_timestamps'].append(camera_params.get('timestamp', timestamp))
    
    def end_episode(self, success=True):
        """End episode and save all data."""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['success'] = success
        self.metadata['num_steps'] = len(self.data['timestamps'])
        
        # Save numpy data
        np.save(os.path.join(self.episode_dir, "state.npy"), 
                np.array(self.data['joint_positions']))
        np.save(os.path.join(self.episode_dir, "action.npy"),
                np.array(self.data['joint_velocities']))
        np.save(os.path.join(self.episode_dir, "target_eef.npy"),
                np.array(self.data['target_eef_positions']))
        np.save(os.path.join(self.episode_dir, "actual_eef.npy"),
                np.array(self.data['actual_eef_positions']))
        
        # Save timestamps
        np.save(os.path.join(self.episode_dir, "timestamps.npy"),
                np.array(self.data['timestamps']))
        np.save(os.path.join(self.episode_dir, "frame_indices.npy"),
                np.array(self.data['frame_indices']))
        
        # Save camera params
        if self.data['camera_intrinsics']:
            np.save(os.path.join(self.episode_dir, "camera_intrinsics.npy"),
                    np.array(self.data['camera_intrinsics']))
            np.save(os.path.join(self.episode_dir, "camera_extrinsics.npy"),
                    np.array(self.data['camera_extrinsics']))
            np.save(os.path.join(self.episode_dir, "camera_timestamps.npy"),
                    np.array(self.data['camera_timestamps']))
        
        # Save metadata
        with open(os.path.join(self.episode_dir, "meta.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save success flag
        with open(os.path.join(self.episode_dir, "success.txt"), 'w') as f:
            f.write("1" if success else "0")
        
        print(f"Episode {self.episode_count} saved. Steps: {self.metadata['num_steps']}")

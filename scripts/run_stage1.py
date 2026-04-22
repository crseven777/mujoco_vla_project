"""
Stage 1: G1 Upper Body End-Effector Position Tracking

This script demonstrates end-effector position following in MuJoCo.
- Loads G1 robot model
- Implements PD controller for right hand tracking
- Captures RGBD images
- Records all data
"""
import os
import sys
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.pd_controller import PDController
from envs.rgbd_camera import RGBDCamera
from envs.data_recorder import DataRecorder


def _get_joint_qpos_qvel_ids(model, joint_names):
    """Get qpos/qvel indices for a list of 1-DoF joints."""
    qpos_ids = []
    qvel_ids = []
    for joint_name in joint_names:
        joint_id = model.joint(joint_name).id
        qpos_ids.append(model.jnt_qposadr[joint_id])
        qvel_ids.append(model.jnt_dofadr[joint_id])
    return qpos_ids, qvel_ids


def _build_non_arm_holding_set(model, arm_actuator_names):
    """Build actuator/joint index sets for holding all non-arm joints fixed."""
    arm_set = set(arm_actuator_names)
    non_arm_actuator_ids = []
    non_arm_joint_names = []

    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if actuator_name not in arm_set:
            non_arm_actuator_ids.append(actuator_id)
            # In this model each motor actuator has the same name as its controlled joint.
            non_arm_joint_names.append(actuator_name)

    non_arm_qpos_ids, non_arm_qvel_ids = _get_joint_qpos_qvel_ids(model, non_arm_joint_names)
    return non_arm_actuator_ids, non_arm_qpos_ids, non_arm_qvel_ids


def run_stage1(output_dir="data/raw", duration=10.0, save_video=False):
    """
    Run Stage 1 demonstration.
    
    Args:
        output_dir: Directory to save recorded data
        duration: Simulation duration in seconds
        save_video: Whether to save video frames
    """
    print("=" * 60)
    print("Stage 1: G1 Upper Body End-Effector Tracking")
    print("=" * 60)
    
    # Load model
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "assets", "g1_upper_body_scene.xml")
    print(f"Loading model from: {xml_path}")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Simulation parameters
    timestep = model.opt.timestep
    control_freq = 100  # Hz
    control_timestep = 1.0 / control_freq
    steps_per_control = int(control_timestep / timestep)
    
    print(f"Simulation timestep: {timestep:.4f}s")
    print(f"Control frequency: {control_freq} Hz")
    print(f"Steps per control: {steps_per_control}")
    
    # Right arm joints and actuators (from G1 model)
    right_arm_joints = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint", 
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ]
    
    right_arm_actuators = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", 
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ]

    # For this task we only drive right arm; all other joints (including left arm) are fixed.
    active_arm_actuators = right_arm_actuators
    
    
    # Initialize controller
    print("Initializing PD controller...")
    controller = PDController(
        model, data,
        arm_joints=right_arm_joints,
        arm_actuators=right_arm_actuators,
        eef_site="right_eef_site",
        kp=45.0,
        kd=6.0,
        max_joint_speed=0.6,
        posture_gain=0.15,
        task_gain=2.0,
        control_mode="kinematic"
    )
    
    # Initialize camera
    print("Initializing RGBD camera...")
    camera = RGBDCamera(
        model, data,
        camera_name="front_camera",
        width=640,
        height=480
    )
    
    # Initialize data recorder
    print("Initializing data recorder...")
    recorder = DataRecorder(output_dir=output_dir)
    recorder.start_episode(
        task_name="eef_position_tracking",
        instruction="Track static target position with right hand"
    )
    
    # Simulation
    max_steps = int(duration / timestep)
    print(f"Running simulation for {duration}s ({max_steps} steps)")
    
    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Hold all non-active joints fixed, so only right arm is allowed to move.
    non_arm_actuator_ids, non_arm_qpos_ids, non_arm_qvel_ids = _build_non_arm_holding_set(
        model, active_arm_actuators
    )
    non_arm_qpos_ref = data.qpos[non_arm_qpos_ids].copy()

    # Use scene red target marker as tracking goal.
    mujoco.mj_forward(model, data)
    initial_eef_pos, _ = controller.get_eef_jacobian()
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_right")
    if target_site_id >= 0:
        target_pos = data.site_xpos[target_site_id].copy()
    else:
        target_pos = np.array([0.3, -0.3, 1.2])
    print(f"Initial right EEF: {initial_eef_pos}")
    print(f"Target right EEF:  {target_pos}")

    print("Floating base is welded in scene XML")
    print(f"Holding {len(non_arm_actuator_ids)} non-arm actuators fixed")
    
    step_count = 0
    control_count = 0
    converged = False
    start_time = time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Update target visualization
            viewer.user_scn.ngeom = 0
            target_geom = viewer.user_scn.geoms[0]
            target_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            target_geom.size[:] = [0.05, 0, 0]
            target_geom.rgba[:] = [1, 0, 0, 0.5]
            target_geom.pos[:] = target_pos
            viewer.user_scn.ngeom = 1

            # Control loop
            while viewer.is_running() and step_count < max_steps:
                # Kinematic lock for all non-active joints (legs, waist, left arm).
                data.qpos[non_arm_qpos_ids] = non_arm_qpos_ref
                data.qvel[non_arm_qvel_ids] = 0.0
                data.ctrl[non_arm_actuator_ids] = 0.0
                mujoco.mj_forward(model, data)

                # Step simulation
                mujoco.mj_step(model, data)
                step_count += 1

                # Control at lower frequency
                if step_count % steps_per_control == 0:
                    control_count += 1

                    # Compute control
                    converged, error = controller.step(target_pos, dt=control_timestep)

                    # Get current state
                    current_joint_pos = data.qpos[7:model.nq].copy()
                    current_joint_vel = data.qvel[6:model.nv].copy()

                    # Get actual end-effector position
                    actual_eef_pos, _ = controller.get_eef_jacobian()

                    # Capture synchronized RGBD frame and camera parameters.
                    camera_frame = camera.capture(timestamp=data.time)

                    # Record one aligned frame each control tick.
                    recorder.record(
                        timestamp=data.time,
                        joint_pos=current_joint_pos,
                        joint_vel=current_joint_vel,
                        target_eef=target_pos,
                        actual_eef=actual_eef_pos,
                        rgb_image=camera_frame["rgb"],
                        depth_image=camera_frame["depth"],
                        camera_params=camera_frame
                    )

                    # Update viewer info
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

                    # Update target visualization
                    target_geom.pos[:] = target_pos

                    # Print status every 50 control steps
                    if control_count % 50 == 0:
                        filtered_error = getattr(controller, "_ema_error", error)
                        print(f"Time: {data.time:.2f}s | "
                              f"Error: {error:.4f}m | "
                              f"Filtered: {filtered_error:.4f}m | "
                              f"Converged: {converged} | "
                              f"EEF: {actual_eef_pos}")

                # Sync viewer
                viewer.sync()
    finally:
        camera.close()

    # End episode
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f}s")
    print(f"Total steps: {step_count}")
    print(f"Control steps: {control_count}")

    recorder.end_episode(success=converged)
    print(f"Data saved to: {recorder.episode_dir}")
    
    return recorder.episode_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: End-effector tracking")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for recorded data")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Simulation duration in seconds")
    parser.add_argument("--save-video", action="store_true",
                       help="Save video frames")
    
    args = parser.parse_args()
    
    run_stage1(
        output_dir=args.output_dir,
        duration=args.duration,
        save_video=args.save_video
    )

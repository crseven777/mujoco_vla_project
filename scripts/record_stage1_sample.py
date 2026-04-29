"""Record one minimal Stage-1 episode sample."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.pd_controller import PDController
from envs.rgbd_camera import RGBDCamera
from envs.tracking_logger import TrackingLogger


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _get_joint_qpos_qvel_ids(model, joint_names):
    qpos_ids, qvel_ids = [], []
    for joint_name in joint_names:
        jid = model.joint(joint_name).id
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])
    return qpos_ids, qvel_ids


def _build_non_arm_holding_set(model, arm_actuator_names):
    arm_set = set(arm_actuator_names)
    non_arm_actuator_ids = []
    non_arm_joint_names = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name not in arm_set:
            non_arm_actuator_ids.append(actuator_id)
            non_arm_joint_names.append(name)
    non_arm_qpos_ids, non_arm_qvel_ids = _get_joint_qpos_qvel_ids(model, non_arm_joint_names)
    return non_arm_actuator_ids, non_arm_qpos_ids, non_arm_qvel_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-freq", type=float, default=100.0)
    parser.add_argument("--output-dir", type=str, default="data/samples/stage1_static_right_hand")
    parser.add_argument("--camera-name", type=str, default="front_camera")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}. Use --overwrite.")
        shutil.rmtree(args.output_dir)

    rgb_dir = os.path.join(args.output_dir, "rgb")
    depth_dir = os.path.join(args.output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "g1_upper_body_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    controller = PDController(
        model,
        data,
        arm_joints=RIGHT_ARM_JOINTS,
        arm_actuators=RIGHT_ARM_JOINTS,
        eef_site="right_eef_site",
        kp=45.0,
        kd=6.0,
        max_joint_speed=0.6,
        posture_gain=0.15,
        task_gain=2.0,
        control_mode="kinematic",
    )
    camera = RGBDCamera(model, data, camera_name=args.camera_name, width=640, height=480)

    target_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_right")
    target = data.site_xpos[target_sid].copy() if target_sid >= 0 else np.array([0.3, -0.3, 1.2])

    non_arm_actuator_ids, non_arm_qpos_ids, non_arm_qvel_ids = _build_non_arm_holding_set(model, RIGHT_ARM_JOINTS)
    non_arm_qpos_ref = data.qpos[non_arm_qpos_ids].copy()
    qpos_ids, _ = _get_joint_qpos_qvel_ids(model, RIGHT_ARM_JOINTS)

    timestep = model.opt.timestep
    steps_per_control = max(1, int(round((1.0 / args.control_freq) / timestep)))
    max_steps = int(args.duration / timestep)

    logger = TrackingLogger(threshold_m=0.05)

    try:
        frame_idx = 0
        for step in range(max_steps):
            data.qpos[non_arm_qpos_ids] = non_arm_qpos_ref
            data.qvel[non_arm_qvel_ids] = 0.0
            data.ctrl[non_arm_actuator_ids] = 0.0
            mujoco.mj_forward(model, data)

            mujoco.mj_step(model, data)
            if step % steps_per_control != 0:
                continue

            controller.step(target, dt=steps_per_control * timestep)
            actual, _ = controller.get_eef_jacobian()
            joint_state = data.qpos[qpos_ids].copy()
            action = controller.last_action.copy()
            logger.record(
                timestamp=data.time,
                target_eef_right=target,
                actual_eef_right=actual,
                joint_state=joint_state,
                action_or_ctrl=action,
            )

            frame = camera.capture(timestamp=float(data.time))
            iio.imwrite(os.path.join(rgb_dir, f"{frame_idx:06d}.png"), frame["rgb"])
            np.save(os.path.join(depth_dir, f"{frame_idx:06d}.npy"), frame["depth"])
            frame_idx += 1
    finally:
        camera.close()

    logger.save_np(args.output_dir)

    s = logger.summary()
    meta = {
        "task_name": "stage1_static_right_hand",
        "robot_name": "g1_29dof",
        "mujoco_model_path": xml_path,
        "camera_name": args.camera_name,
        "sampling_rate": args.control_freq,
        "episode_length": int(s["num_frames"]),
        "target_type": "static",
        "target_position": target.tolist(),
        "date": datetime.now().isoformat(),
        "camera_intrinsics": frame["intrinsics"].tolist(),
        "camera_extrinsics": frame["extrinsics"].tolist(),
        "summary": s,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Recorded sample at: {args.output_dir}")
    print(f"mean error: {s['mean_tracking_error_m']:.6f} m")
    print(f"max error: {s['max_tracking_error_m']:.6f} m")
    print(f"final error: {s['final_tracking_error_m']:.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

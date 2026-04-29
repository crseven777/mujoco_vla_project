"""Stage 2 MVP: XR -> bridge -> MuJoCo right-hand target tracking.

This demo keeps the original Stage-1 right-arm controller and only replaces
its target source with XR+bridge output. First version focuses on right hand.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.pd_controller import PDController
from teleop.bridge import BridgeConfig, XRTeleopBridge


def _get_joint_qpos_qvel_ids(model, joint_names):
    qpos_ids = []
    qvel_ids = []
    for joint_name in joint_names:
        joint_id = model.joint(joint_name).id
        qpos_ids.append(model.jnt_qposadr[joint_id])
        qvel_ids.append(model.jnt_dofadr[joint_id])
    return qpos_ids, qvel_ids


def _build_non_arm_holding_set(model, arm_actuator_names):
    arm_set = set(arm_actuator_names)
    non_arm_actuator_ids = []
    non_arm_joint_names = []

    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if actuator_name not in arm_set:
            non_arm_actuator_ids.append(actuator_id)
            non_arm_joint_names.append(actuator_name)

    non_arm_qpos_ids, non_arm_qvel_ids = _get_joint_qpos_qvel_ids(model, non_arm_joint_names)
    return non_arm_actuator_ids, non_arm_qpos_ids, non_arm_qvel_ids


def _make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = np.array([x, y, z], dtype=float)
    return pose


class MockXRSource:
    """No-device XR source for bridge and controller testing."""

    def __init__(self):
        self._t0 = time.time()

    def read_state(self):
        t = time.time() - self._t0
        return {
            "timestamp": time.time(),
            "head_pose": _make_pose(0.02 * np.sin(0.4 * t), 0.0, 1.5),
            "left_wrist_pose": _make_pose(0.3, 0.18, 1.05),
            "right_wrist_pose": _make_pose(
                0.40 + 0.10 * np.cos(2.0 * t),
                -0.20 + 0.08 * np.sin(2.0 * t),
                1.10 + 0.05 * np.sin(1.5 * t),
            ),
        }

    def close(self):
        return None


def run_teleop_demo(duration: float = 30.0, control_freq: float = 100.0, mock_xr: bool = True):
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "g1_upper_body_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    timestep = model.opt.timestep
    control_timestep = 1.0 / control_freq
    steps_per_control = max(1, int(control_timestep / timestep))

    right_arm_joints = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    right_arm_actuators = list(right_arm_joints)

    controller = PDController(
        model,
        data,
        arm_joints=right_arm_joints,
        arm_actuators=right_arm_actuators,
        eef_site="right_eef_site",
        kp=45.0,
        kd=6.0,
        max_joint_speed=0.6,
        posture_gain=0.15,
        task_gain=2.0,
        control_mode="kinematic",
    )

    # Bridge config tuned for current scene bounds (same defaults as bridge.py)
    bridge_cfg = BridgeConfig(
        xr_to_mj_rot=np.eye(3),
        xr_to_mj_trans=np.array([0.20, 0.0, 0.80]),
        arm_scale=1.10,
        body_offset=np.array([0.0, 0.0, 0.0]),
        lowpass_alpha=0.20,
        debug=True,
        debug_every_n=20,
    )
    bridge = XRTeleopBridge(bridge_cfg)

    if mock_xr:
        xr_source = MockXRSource()
    else:
        from teleop.adapter_xr import XRAdapter

        xr_source = XRAdapter()
        xr_source.start()

    max_steps = int(duration / timestep)
    mujoco.mj_resetData(model, data)

    non_arm_actuator_ids, non_arm_qpos_ids, non_arm_qvel_ids = _build_non_arm_holding_set(
        model, right_arm_actuators
    )
    non_arm_qpos_ref = data.qpos[non_arm_qpos_ids].copy()

    control_count = 0
    step_count = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.user_scn.ngeom = 1
            target_geom = viewer.user_scn.geoms[0]
            target_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            target_geom.size[:] = [0.05, 0, 0]
            target_geom.rgba[:] = [1, 0, 0, 0.5]

            while viewer.is_running() and step_count < max_steps:
                data.qpos[non_arm_qpos_ids] = non_arm_qpos_ref
                data.qvel[non_arm_qvel_ids] = 0.0
                data.ctrl[non_arm_actuator_ids] = 0.0
                mujoco.mj_forward(model, data)

                mujoco.mj_step(model, data)
                step_count += 1

                if step_count % steps_per_control == 0:
                    control_count += 1

                    xr_state = xr_source.read_state()
                    bridge_out = bridge.update(xr_state)
                    target_pos = bridge_out["right_target_pos"]

                    _, err = controller.step(target_pos, dt=control_timestep)
                    eef_pos, _ = controller.get_eef_jacobian()

                    target_geom.pos[:] = target_pos

                    if control_count % 20 == 0:
                        print(
                            f"t={data.time:.2f}s | XR_R={xr_state['right_wrist_pose'][:3,3]} "
                            f"| target_R={target_pos} | eef_R={eef_pos} | err={err:.4f}"
                        )

                viewer.sync()
    finally:
        xr_source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage2 MVP: XR bridge to MuJoCo right hand")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--control-freq", type=float, default=100.0)
    parser.add_argument(
        "--mock-xr",
        action="store_true",
        help="Use internal mock XR signal (no headset required)",
    )
    args = parser.parse_args()

    run_teleop_demo(duration=args.duration, control_freq=args.control_freq, mock_xr=args.mock_xr)

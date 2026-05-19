"""Stage2 Task2 demo: mock XR -> bridge -> bimanual MuJoCo targets.

Modes:
- trajectory: use existing stage1 target provider
- teleop: use mock XR + bridge
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.bimanual_controller import BimanualController, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from controllers.target_provider import TargetProvider, TargetProviderConfig
from envs.rgbd_camera import RGBDCamera
from envs.stage1_logger import Stage1Logger
from teleop.adapter_xr import XRAdapter, XRAdapterConfig
from teleop.bridge import BridgeConfig, XRTeleopBridge
from teleop.mock_xr_input import MockXRInput


def _parse_scalar(raw: str):
    s = raw.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if any(c in s for c in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        pass
    if s.startswith("[") and s.endswith("]"):
        return ast.literal_eval(s)
    return s.strip("\"'")


def load_simple_yaml(path: str) -> dict:
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            cfg[k.strip()] = _parse_scalar(v)
    return cfg


def _get_joint_qpos_ids(model, joint_names):
    return [model.jnt_qposadr[model.joint(name).id] for name in joint_names]


def _build_non_arm_holding_set(model, arm_actuator_names):
    arm_set = set(arm_actuator_names)
    non_arm_actuator_ids = []
    non_arm_joint_names = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name not in arm_set:
            non_arm_actuator_ids.append(actuator_id)
            non_arm_joint_names.append(name)
    qpos_ids = [model.jnt_qposadr[model.joint(n).id] for n in non_arm_joint_names]
    qvel_ids = [model.jnt_dofadr[model.joint(n).id] for n in non_arm_joint_names]
    return non_arm_actuator_ids, qpos_ids, qvel_ids


def _draw_sphere(user_scn, idx: int, pos: np.ndarray, size: float, rgba: tuple[float, float, float, float]):
    g = user_scn.geoms[idx]
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = [size, 0.0, 0.0]
    g.rgba[:] = np.asarray(rgba, dtype=float)
    g.pos[:] = np.asarray(pos, dtype=float)


def run_demo(
    duration: float,
    warmup_duration: float,
    control_dt: float,
    mode: str,
    cfg_stage1: str,
    cfg_bridge: str,
    output_dir: str,
    max_joint_speed: float | None = None,
    task_gain: float | None = None,
    hand_mode: str = "bimanual",
    xr_source_kind: str = "mock",
    xr_repo_root: str = "/home/wll/xr_teleoperate",
    use_hand_tracking: bool = True,
    display_mode: str = "pass-through",
    webrtc_url: str | None = None,
    show_markers: bool = True,
    record_rgbd: bool = False,
    save_every_n_frames: int = 1,
    camera_name: str | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    task_name: str = "stage2_teleop_tracking",
    instruction: str = "follow the XR controller with the robot upper body end effectors",
    operator_id: str = "unknown",
    success: bool = True,
):
    stage1_cfg = load_simple_yaml(cfg_stage1)
    bridge_cfg_dict = load_simple_yaml(cfg_bridge)

    xml_path = str(stage1_cfg.get("mujoco_model_path", "assets/g1_upper_body_scene.xml"))
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    timestep = model.opt.timestep
    steps_per_control = max(1, int(round(control_dt / timestep)))
    max_steps = int((warmup_duration + duration) / timestep)

    cfg_max_joint_speed = float(stage1_cfg.get("max_joint_speed", 0.6))
    cfg_task_gain = float(stage1_cfg.get("task_gain", 2.0))
    if mode == "teleop":
        # Teleop-following priority: default higher bandwidth on controller side.
        cfg_max_joint_speed = max(cfg_max_joint_speed, 2.2)
        cfg_task_gain = max(cfg_task_gain, 5.0)
    controller = BimanualController(
        model=model,
        data=data,
        control_mode=str(stage1_cfg.get("control_mode", "kinematic")),
        right_eef_site=str(stage1_cfg.get("right_eef_site_name", "right_eef_site")),
        left_eef_site=str(stage1_cfg.get("left_eef_site_name", "left_eef_site")),
        kp=float(stage1_cfg.get("kp", 45.0)),
        kd=float(stage1_cfg.get("kd", 6.0)),
        max_joint_speed=float(max_joint_speed if max_joint_speed is not None else cfg_max_joint_speed),
        posture_gain=float(stage1_cfg.get("posture_gain", 0.15)),
        task_gain=float(task_gain if task_gain is not None else cfg_task_gain),
    )

    target_provider = TargetProvider(
        TargetProviderConfig(
            mode=str(stage1_cfg.get("target_mode", "trajectory_bimanual")),
            trajectory_type=str(stage1_cfg.get("trajectory_type", "circle")),
            right_target_center=tuple(stage1_cfg.get("right_target_center", [0.35, -0.22, 1.08])),
            left_target_center=tuple(stage1_cfg.get("left_target_center", [0.35, 0.22, 1.08])),
            right_amplitude=tuple(stage1_cfg.get("right_trajectory_amplitude", [0.06, 0.05, 0.03])),
            left_amplitude=tuple(stage1_cfg.get("left_trajectory_amplitude", [0.06, 0.05, 0.03])),
            right_radius=float(stage1_cfg.get("right_trajectory_radius", 0.08)),
            left_radius=float(stage1_cfg.get("left_trajectory_radius", 0.08)),
            frequency=float(stage1_cfg.get("frequency", 0.12)),
            right_axis=str(stage1_cfg.get("right_axis", "xy")),
            left_axis=str(stage1_cfg.get("left_axis", "xy")),
        )
    )

    bridge = XRTeleopBridge(
        BridgeConfig(
            transform_matrix=np.asarray(bridge_cfg_dict.get("transform_matrix", np.eye(4)), dtype=float),
            scale_xyz=np.asarray(bridge_cfg_dict.get("scale_factor", [1.0, 1.0, 1.0]), dtype=float),
            xr_origin=np.asarray(bridge_cfg_dict.get("xr_origin", [0.0, 0.0, 0.0]), dtype=float),
            left_robot_origin=np.asarray(bridge_cfg_dict.get("left_robot_origin", [0.35, 0.22, 1.05]), dtype=float),
            right_robot_origin=np.asarray(bridge_cfg_dict.get("right_robot_origin", [0.35, -0.22, 1.05]), dtype=float),
            left_min_bound=np.asarray(bridge_cfg_dict.get("left_min_bound", [0.10, 0.02, 0.70]), dtype=float),
            left_max_bound=np.asarray(bridge_cfg_dict.get("left_max_bound", [0.75, 0.55, 1.45]), dtype=float),
            right_min_bound=np.asarray(bridge_cfg_dict.get("right_min_bound", [0.10, -0.55, 0.70]), dtype=float),
            right_max_bound=np.asarray(bridge_cfg_dict.get("right_max_bound", [0.75, -0.02, 1.45]), dtype=float),
            smoothing_alpha=float(bridge_cfg_dict.get("smoothing_alpha", 0.2)),
            max_target_speed=float(bridge_cfg_dict.get("max_target_speed", 0.0)),
            debug=True,
            debug_every_n=20,
        )
    )
    if mode == "teleop" and xr_source_kind == "real":
        xr_source = XRAdapter(
            XRAdapterConfig(
                xr_repo_root=xr_repo_root,
                use_hand_tracking=use_hand_tracking,
                display_mode=display_mode,
                webrtc_url=webrtc_url,
            )
        )
        xr_source.start()
        print(f"XR source: real xr_teleoperate ({xr_repo_root})")
    else:
        xr_source = MockXRInput()
        print("XR source: mock")

    camera = None
    camera_meta = None
    if record_rgbd:
        camera = RGBDCamera(
            model=model,
            data=data,
            camera_name=camera_name or str(stage1_cfg.get("camera_name", "front_camera")),
            width=int(camera_width or stage1_cfg.get("camera_width", 640)),
            height=int(camera_height or stage1_cfg.get("camera_height", 480)),
        )
        camera_meta = camera.get_camera_params()

    logger = Stage1Logger(
        output_dir=output_dir,
        save_rgbd=record_rgbd,
        save_every_n_frames=save_every_n_frames,
    )

    non_arm_act_ids, non_arm_qpos_ids, non_arm_qvel_ids = _build_non_arm_holding_set(model, controller.all_active_actuator_names)
    non_arm_qpos_ref = data.qpos[non_arm_qpos_ids].copy()
    left_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(stage1_cfg.get("left_marker_name", "target_left")))
    right_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(stage1_cfg.get("right_marker_name", "target_right")))
    if not show_markers:
        for marker_id in (left_marker_id, right_marker_id):
            if marker_id >= 0:
                model.site_rgba[marker_id, 3] = 0.0

    left_qpos_ids = _get_joint_qpos_ids(model, LEFT_ARM_JOINTS)
    right_qpos_ids = _get_joint_qpos_ids(model, RIGHT_ARM_JOINTS)

    frame_idx = 0
    ctrl_count = 0
    record_count = 0
    recording_started = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 6 spheres: xr_left, xr_right, target_left, target_right, actual_left, actual_right
        viewer.user_scn.ngeom = 6 if show_markers else 0

        while viewer.is_running() and frame_idx < max_steps:
            data.qpos[non_arm_qpos_ids] = non_arm_qpos_ref
            data.qvel[non_arm_qvel_ids] = 0.0
            data.ctrl[non_arm_act_ids] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            if frame_idx % steps_per_control == 0:
                ctrl_count += 1

                xr_state = xr_source.read_state()
                bridge_out = bridge.update(xr_state)

                if mode == "teleop":
                    left_target = bridge_out["left_target_pos"]
                    right_target = bridge_out["right_target_pos"]
                    if hand_mode == "right_only":
                        left_target = None
                    elif hand_mode == "left_only":
                        right_target = None
                    mode_name = "teleop"
                else:
                    tgt = target_provider.get_target(data.time)
                    left_target = tgt["left_target_pos"]
                    right_target = tgt["right_target_pos"]
                    if hand_mode == "right_only":
                        left_target = None
                    elif hand_mode == "left_only":
                        right_target = None
                    mode_name = "trajectory"

                out = controller.step(left_target, right_target, dt=control_dt)
                left_actual = out["left_actual"]
                right_actual = out["right_actual"]

                # Keep scene's original target sites synchronized for intuitive visualization.
                if left_marker_id >= 0 and left_target is not None:
                    model.site_pos[left_marker_id] = np.asarray(left_target, dtype=float)
                if right_marker_id >= 0 and right_target is not None:
                    model.site_pos[right_marker_id] = np.asarray(right_target, dtype=float)
                mujoco.mj_forward(model, data)

                if show_markers:
                    # Visualize XR raw wrist positions (green shades)
                    xr_left = np.asarray(xr_state["left_wrist_pose"][:3], dtype=float)
                    xr_right = np.asarray(xr_state["right_wrist_pose"][:3], dtype=float)
                    _draw_sphere(viewer.user_scn, 0, xr_left, 0.03, (0.0, 1.0, 0.0, 0.8))
                    _draw_sphere(viewer.user_scn, 1, xr_right, 0.03, (0.2, 0.8, 0.2, 0.8))

                    # Visualize bridge / controller targets (red/blue)
                    left_tgt_vis = np.asarray(left_target, dtype=float) if left_target is not None else left_actual
                    right_tgt_vis = np.asarray(right_target, dtype=float) if right_target is not None else right_actual
                    _draw_sphere(viewer.user_scn, 2, left_tgt_vis, 0.035, (0.1, 0.3, 1.0, 0.7))
                    _draw_sphere(viewer.user_scn, 3, right_tgt_vis, 0.035, (1.0, 0.1, 0.1, 0.7))

                    # Visualize actual eef positions (yellow/cyan)
                    _draw_sphere(viewer.user_scn, 4, left_actual, 0.02, (0.0, 1.0, 1.0, 0.9))
                    _draw_sphere(viewer.user_scn, 5, right_actual, 0.02, (1.0, 1.0, 0.0, 0.9))

                is_recording = data.time >= warmup_duration
                if is_recording and not recording_started:
                    recording_started = True
                    print(f"Recording started at sim t={data.time:.2f}s")

                if is_recording:
                    camera_frame = camera.capture(timestamp=float(data.time - warmup_duration)) if camera is not None else None
                    joint_state = np.concatenate([data.qpos[left_qpos_ids].copy(), data.qpos[right_qpos_ids].copy()])
                    logger.record(
                        frame_idx=record_count,
                        timestamp=float(data.time - warmup_duration),
                        mode=f"{mode_name}:{hand_mode}",
                        joint_state=joint_state,
                        action=out["action"],
                        target_left=left_target,
                        target_right=right_target,
                        actual_left=left_actual,
                        actual_right=right_actual,
                        err_left=out["left_error"],
                        err_right=out["right_error"],
                        camera_frame=camera_frame,
                        left_wrist_pose=bridge_out["left_wrist_pose"],
                        right_wrist_pose=bridge_out["right_wrist_pose"],
                        raw_target_left=bridge_out["raw_left_target_pos"],
                        raw_target_right=bridge_out["raw_right_target_pos"],
                        transformed_target_left=bridge_out["left_target_pos"],
                        transformed_target_right=bridge_out["right_target_pos"],
                    )
                    record_count += 1

                if ctrl_count % 20 == 0:
                    xr_right_pos = np.asarray(xr_state["right_wrist_pose"][:3], dtype=float)
                    right_trigger = xr_state.get("right_ctrl_trigger", False)
                    right_trigger_value = xr_state.get("right_ctrl_trigger_value", np.nan)
                    right_squeeze = xr_state.get("right_ctrl_squeeze", False)
                    right_squeeze_value = xr_state.get("right_ctrl_squeeze_value", np.nan)
                    print(
                        f"t={data.time:.2f}s | mode={mode_name} | "
                        f"L err={out['left_error']:.4f} | R err={out['right_error']:.4f} | "
                        f"xr R={xr_right_pos} | raw R={bridge_out['raw_right_target_pos']} -> tgt R={right_target} | "
                        f"R trig={right_trigger}/{right_trigger_value:.2f} sqz={right_squeeze}/{right_squeeze_value:.2f}"
                    )

            frame_idx += 1
            viewer.sync()

    err_l = np.asarray(logger.tracking_error_left, dtype=float)
    err_r = np.asarray(logger.tracking_error_right, dtype=float)
    summary = Stage1Logger.summarize_errors(err_l, err_r, threshold=0.05)
    logger.save(
        meta={
            "task_name": task_name,
            "instruction": instruction,
            "robot_type": "g1_29dof_upper_body",
            "operator_id": operator_id,
            "sampling_rate_hz": float(1.0 / control_dt),
            "warmup_duration": warmup_duration,
            "record_duration": duration,
            "mode": mode,
            "hand_mode": hand_mode,
            "xr_source": xr_source_kind,
            "xr_input_mode": "hand" if use_hand_tracking else "controller",
            "show_markers": show_markers,
            "success": success,
            "summary": summary,
            "bridge_config": bridge_cfg_dict,
            "stage1_config": cfg_stage1,
        },
        camera_meta=camera_meta,
    )
    if camera is not None:
        camera.close()
    xr_source.close()
    print(f"Saved logs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage2 Task2 demo: XR/bridge to MuJoCo target tracking")
    parser.add_argument("--mode", choices=["trajectory", "teleop"], default="teleop")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup-duration", type=float, default=0.0, help="Run this many sim seconds before recording")
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--stage1-config", type=str, default="configs/stage1_bimanual_trajectory.yaml")
    parser.add_argument("--bridge-config", type=str, default="configs/bridge.yaml")
    parser.add_argument("--output-dir", type=str, default="data/samples/stage2_bridge_demo")
    parser.add_argument("--max-joint-speed", type=float, default=None, help="Override arm joint speed limit (rad/s)")
    parser.add_argument("--task-gain", type=float, default=None, help="Override task-space gain")
    parser.add_argument("--hand-mode", choices=["bimanual", "right_only", "left_only"], default="bimanual")
    parser.add_argument("--xr-source", choices=["mock", "real"], default="mock")
    parser.add_argument("--xr-repo-root", type=str, default="/home/wll/xr_teleoperate")
    parser.add_argument("--xr-input-mode", choices=["hand", "controller"], default="hand")
    parser.add_argument("--display-mode", type=str, default="pass-through")
    parser.add_argument("--webrtc-url", type=str, default=None)
    parser.add_argument("--record-rgbd", action="store_true", help="Record RGB/depth frames and camera parameters")
    parser.add_argument("--save-every-n-frames", type=int, default=1, help="Save every N control frames when recording RGBD")
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--task-name", type=str, default="stage2_teleop_tracking")
    parser.add_argument(
        "--instruction",
        type=str,
        default="follow the XR controller with the robot upper body end effectors",
    )
    parser.add_argument("--operator-id", type=str, default="unknown")
    parser.add_argument("--success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--show-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide target/debug spheres in the MuJoCo viewer",
    )
    args = parser.parse_args()

    run_demo(
        duration=args.duration,
        warmup_duration=args.warmup_duration,
        control_dt=args.control_dt,
        mode=args.mode,
        cfg_stage1=args.stage1_config,
        cfg_bridge=args.bridge_config,
        output_dir=args.output_dir,
        max_joint_speed=args.max_joint_speed,
        task_gain=args.task_gain,
        hand_mode=args.hand_mode,
        xr_source_kind=args.xr_source,
        xr_repo_root=args.xr_repo_root,
        use_hand_tracking=(args.xr_input_mode == "hand"),
        display_mode=args.display_mode,
        webrtc_url=args.webrtc_url,
        show_markers=args.show_markers,
        record_rgbd=args.record_rgbd,
        save_every_n_frames=args.save_every_n_frames,
        camera_name=args.camera_name,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        task_name=args.task_name,
        instruction=args.instruction,
        operator_id=args.operator_id,
        success=args.success,
    )

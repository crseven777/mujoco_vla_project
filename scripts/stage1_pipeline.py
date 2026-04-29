"""Shared pipeline for Stage-1 static/trajectory right/left/bimanual runs and recording."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.bimanual_controller import BimanualController, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from controllers.target_provider import TargetProvider, TargetProviderConfig
from envs.rgbd_camera import RGBDCamera
from envs.stage1_logger import Stage1Logger


@dataclass
class RunResult:
    summary: dict
    episode_length: int
    saved_frame_count: int
    output_dir: str


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


def run_stage1(cfg: dict, output_dir: str | None = None, record_rgbd: bool = True) -> RunResult:
    xml_path = cfg["mujoco_model_path"]
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    control_mode = str(cfg.get("control_mode", "kinematic"))
    dt = float(cfg.get("control_dt", 0.01))
    timestep = model.opt.timestep
    steps_per_control = max(1, int(round(dt / timestep)))
    max_steps = int(cfg.get("max_steps", 1200))

    right_eef = str(cfg.get("right_eef_site_name", "right_eef_site"))
    left_eef = str(cfg.get("left_eef_site_name", "left_eef_site"))

    controller = BimanualController(
        model=model,
        data=data,
        control_mode=control_mode,
        right_eef_site=right_eef,
        left_eef_site=left_eef,
        kp=float(cfg.get("kp", 45.0)),
        kd=float(cfg.get("kd", 6.0)),
        max_joint_speed=float(cfg.get("max_joint_speed", 0.6)),
        posture_gain=float(cfg.get("posture_gain", 0.15)),
        task_gain=float(cfg.get("task_gain", 2.0)),
    )

    target_cfg = TargetProviderConfig(
        mode=str(cfg.get("target_mode", "trajectory_bimanual")),
        trajectory_type=str(cfg.get("trajectory_type", "circle")),
        right_target_center=tuple(cfg.get("right_target_center", [0.35, -0.22, 1.08])),
        left_target_center=tuple(cfg.get("left_target_center", [0.35, 0.22, 1.08])),
        right_amplitude=tuple(cfg.get("right_trajectory_amplitude", [0.06, 0.05, 0.03])),
        left_amplitude=tuple(cfg.get("left_trajectory_amplitude", [0.06, 0.05, 0.03])),
        right_radius=float(cfg.get("right_trajectory_radius", 0.08)),
        left_radius=float(cfg.get("left_trajectory_radius", 0.08)),
        frequency=float(cfg.get("frequency", 0.12)),
        right_axis=str(cfg.get("right_axis", "xy")),
        left_axis=str(cfg.get("left_axis", "xy")),
    )
    target_provider = TargetProvider(target_cfg)

    cam = None
    save_rgbd = bool(cfg.get("save_rgbd", record_rgbd and True))
    if save_rgbd:
        cam = RGBDCamera(
            model,
            data,
            camera_name=str(cfg.get("camera_name", "front_camera")),
            width=int(cfg.get("camera_width", 640)),
            height=int(cfg.get("camera_height", 480)),
        )

    out_dir = output_dir or str(cfg.get("output_dir", "data/samples/stage1_bimanual_trajectory"))
    logger = Stage1Logger(
        output_dir=out_dir,
        save_rgbd=save_rgbd,
        save_every_n_frames=int(cfg.get("save_every_n_frames", 1)),
    )

    non_arm_act_ids, non_arm_qpos_ids, non_arm_qvel_ids = _build_non_arm_holding_set(model, controller.all_active_actuator_names)
    non_arm_qpos_ref = data.qpos[non_arm_qpos_ids].copy()

    left_marker_name = str(cfg.get("left_marker_name", "target_left"))
    right_marker_name = str(cfg.get("right_marker_name", "target_right"))
    left_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_marker_name)
    right_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, right_marker_name)

    left_qpos_ids = _get_joint_qpos_ids(model, LEFT_ARM_JOINTS)
    right_qpos_ids = _get_joint_qpos_ids(model, RIGHT_ARM_JOINTS)

    frame_idx = 0
    last_cam = None
    try:
        for step in range(max_steps):
            data.qpos[non_arm_qpos_ids] = non_arm_qpos_ref
            data.qvel[non_arm_qvel_ids] = 0.0
            data.ctrl[non_arm_act_ids] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            if step % steps_per_control != 0:
                continue

            tgt = target_provider.get_target(data.time)

            if right_marker_id >= 0 and tgt["right_target_pos"] is not None:
                model.site_pos[right_marker_id] = np.asarray(tgt["right_target_pos"], dtype=float)
            if left_marker_id >= 0 and tgt["left_target_pos"] is not None:
                model.site_pos[left_marker_id] = np.asarray(tgt["left_target_pos"], dtype=float)
            mujoco.mj_forward(model, data)

            out = controller.step(tgt["left_target_pos"], tgt["right_target_pos"], dt=dt)

            camera_frame = None
            if cam is not None:
                camera_frame = cam.capture(timestamp=float(data.time))
                last_cam = camera_frame

            joint_state = np.concatenate([
                data.qpos[left_qpos_ids].copy(),
                data.qpos[right_qpos_ids].copy(),
            ])

            logger.record(
                frame_idx=frame_idx,
                timestamp=float(data.time),
                mode=tgt["mode"],
                joint_state=joint_state,
                action=out["action"],
                target_left=tgt["left_target_pos"],
                target_right=tgt["right_target_pos"],
                actual_left=out["left_actual"],
                actual_right=out["right_actual"],
                err_left=out["left_error"],
                err_right=out["right_error"],
                camera_frame=camera_frame,
            )
            frame_idx += 1
    finally:
        if cam is not None:
            cam.close()

    err_l = np.asarray(logger.tracking_error_left, dtype=float)
    err_r = np.asarray(logger.tracking_error_right, dtype=float)
    threshold = float(cfg.get("tracking_threshold", 0.05))
    summary = Stage1Logger.summarize_errors(err_l, err_r, threshold=threshold)

    meta = {
        "task_name": "stage1_bimanual_trajectory",
        "robot_name": "g1_29dof",
        "mujoco_model_path": xml_path,
        "right_eef_site_name": right_eef,
        "left_eef_site_name": left_eef,
        "camera_name": cfg.get("camera_name", "front_camera"),
        "sampling_rate": float(1.0 / dt),
        "episode_length": frame_idx,
        "control_mode": control_mode,
        "target_mode": cfg.get("target_mode", "trajectory_bimanual"),
        "trajectory_type": cfg.get("trajectory_type", "circle"),
        "target_config": {
            "right_target_center": cfg.get("right_target_center", [0.35, -0.22, 1.08]),
            "left_target_center": cfg.get("left_target_center", [0.35, 0.22, 1.08]),
            "right_trajectory_radius": cfg.get("right_trajectory_radius", 0.08),
            "left_trajectory_radius": cfg.get("left_trajectory_radius", 0.08),
            "frequency": cfg.get("frequency", 0.12),
        },
        "tracking_threshold": threshold,
        "summary": summary,
    }

    camera_meta = None
    if save_rgbd and last_cam is not None:
        camera_meta = {
            "camera_name": cfg.get("camera_name", "front_camera"),
            "width": int(cfg.get("camera_width", 640)),
            "height": int(cfg.get("camera_height", 480)),
            "intrinsics": last_cam["intrinsics"].tolist(),
            "extrinsics": last_cam["extrinsics"].tolist(),
            "save_every_n_frames": int(cfg.get("save_every_n_frames", 1)),
        }

    logger.save(meta=meta, camera_meta=camera_meta)

    return RunResult(
        summary=summary,
        episode_length=frame_idx,
        saved_frame_count=len(logger.camera_frame_indices),
        output_dir=out_dir,
    )

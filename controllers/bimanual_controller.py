"""Bimanual wrapper for left/right end-effector tracking."""

from __future__ import annotations

import numpy as np

from controllers.pd_controller import PDController


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


class BimanualController:
    def __init__(
        self,
        model,
        data,
        control_mode: str = "kinematic",
        right_eef_site: str = "right_eef_site",
        left_eef_site: str = "left_eef_site",
        kp: float = 45.0,
        kd: float = 6.0,
        max_joint_speed: float = 0.6,
        posture_gain: float = 0.15,
        task_gain: float = 2.0,
    ):
        self.model = model
        self.data = data

        self.right = PDController(
            model,
            data,
            arm_joints=RIGHT_ARM_JOINTS,
            arm_actuators=RIGHT_ARM_JOINTS,
            eef_site=right_eef_site,
            kp=kp,
            kd=kd,
            max_joint_speed=max_joint_speed,
            posture_gain=posture_gain,
            task_gain=task_gain,
            control_mode=control_mode,
        )
        self.left = PDController(
            model,
            data,
            arm_joints=LEFT_ARM_JOINTS,
            arm_actuators=LEFT_ARM_JOINTS,
            eef_site=left_eef_site,
            kp=kp,
            kd=kd,
            max_joint_speed=max_joint_speed,
            posture_gain=posture_gain,
            task_gain=task_gain,
            control_mode=control_mode,
        )

        self.right_site_id = self.model.site(right_eef_site).id
        self.left_site_id = self.model.site(left_eef_site).id

    @property
    def all_active_actuator_names(self) -> list[str]:
        return RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS

    def get_actual(self) -> tuple[np.ndarray, np.ndarray]:
        left_actual = self.data.site_xpos[self.left_site_id].copy()
        right_actual = self.data.site_xpos[self.right_site_id].copy()
        return left_actual, right_actual

    def step(self, left_target_pos, right_target_pos, dt: float) -> dict:
        if right_target_pos is not None:
            self.right.step(np.asarray(right_target_pos, dtype=float), dt=dt)
        if left_target_pos is not None:
            self.left.step(np.asarray(left_target_pos, dtype=float), dt=dt)

        left_actual, right_actual = self.get_actual()

        right_error = np.nan
        left_error = np.nan
        if right_target_pos is not None:
            right_error = float(np.linalg.norm(np.asarray(right_target_pos, dtype=float) - right_actual))
        if left_target_pos is not None:
            left_error = float(np.linalg.norm(np.asarray(left_target_pos, dtype=float) - left_actual))

        action = np.concatenate([self.left.last_action, self.right.last_action], axis=0)

        return {
            "left_actual": left_actual,
            "right_actual": right_actual,
            "left_error": left_error,
            "right_error": right_error,
            "action": action,
        }

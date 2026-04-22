"""
PD Controller for end-effector position tracking.
"""
import numpy as np
import mujoco


class PDController:
    """PD controller for end-effector position tracking using Jacobian."""
    
    def __init__(
        self,
        model,
        data,
        arm_joints,
        arm_actuators,
        eef_site,
        kp=100.0,
        kd=10.0,
        max_joint_speed=1.5,
        posture_gain=0.25,
        task_gain=1.5,
        control_mode="torque",
        convergence_threshold=0.04,
        convergence_hysteresis=0.005,
        convergence_required_steps=10,
        error_ema_alpha=0.2,
        latch_convergence=True
    ):
        """
        Initialize PD controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            arm_joints: List of joint names for the arm
            arm_actuators: List of actuator names for the arm
            eef_site: Name of end-effector site
            kp: Proportional gain
            kd: Derivative gain
        """
        self.model = model
        self.data = data
        self.arm_joints = arm_joints
        self.arm_actuators = arm_actuators
        self.eef_site = eef_site
        
        # Get joint and actuator IDs
        self.arm_joint_ids = [self.model.joint(joint_name).id for joint_name in self.arm_joints]
        self.arm_actuator_ids = [self.model.actuator(actuator_name).id for actuator_name in self.arm_actuators]
        
        # Get qpos and qvel addresses for arm joints
        self.arm_joint_qpos_ids = []
        self.arm_joint_qvel_ids = []
        for joint_name in self.arm_joints:
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            self.arm_joint_qpos_ids.append(qpos_addr)
            self.arm_joint_qvel_ids.append(qvel_addr)
        
        try:
            self.eef_site_id = self.model.site(self.eef_site).id
        except KeyError as exc:
            raise ValueError(
                f"End-effector site '{self.eef_site}' not found in model. "
                "Please check XML site definition."
            ) from exc
        
        # Control parameters
        self.kp = kp
        self.kd = kd
        self.max_joint_speed = max_joint_speed
        self.posture_gain = posture_gain
        self.task_gain = task_gain
        self.control_mode = control_mode
        self.convergence_threshold = convergence_threshold
        self.convergence_hysteresis = convergence_hysteresis
        self.convergence_required_steps = convergence_required_steps
        self.error_ema_alpha = error_ema_alpha
        self.latch_convergence = latch_convergence
        self.error_history = []
        self._converged = False
        self._below_threshold_count = 0
        self._ema_error = None

        # Joint limits for safety.
        self.arm_joint_lower = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_joint_upper = self.model.jnt_range[self.arm_joint_ids, 1].copy()
        self.arm_torque_lower = self.model.jnt_actfrcrange[self.arm_joint_ids, 0].copy()
        self.arm_torque_upper = self.model.jnt_actfrcrange[self.arm_joint_ids, 1].copy()

        # Prefer a natural arm posture around initial qpos.
        self.posture_ref = self.data.qpos[self.arm_joint_qpos_ids].copy()
    
    def get_eef_jacobian(self):
        """Compute end-effector position and Jacobian.""" 
        # Current end effector position
        site_pos = self.data.site_xpos[self.eef_site_id].copy()
        
        # Compute Jacobian
        jac_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, None, self.eef_site_id)
        
        # Select relevant columns for arm joints
        jac_eef = jac_pos[:, self.arm_joint_qvel_ids]
        
        return site_pos, jac_eef
    
    def pd_control(self, target_joint_pos):
        """Apply PD control to track target joint positions."""
        current_pos = self.data.qpos[self.arm_joint_qpos_ids].copy()
        current_vel = self.data.qvel[self.arm_joint_qvel_ids].copy()
        
        # Position and velocity errors
        pos_error = target_joint_pos - current_pos
        vel_error = -current_vel
        
        # Add gravity/coriolis compensation to make tracking physically consistent.
        bias_torque = self.data.qfrc_bias[self.arm_joint_qvel_ids].copy()
        torques = bias_torque + self.kp * pos_error + self.kd * vel_error
        torques = np.clip(torques, self.arm_torque_lower, self.arm_torque_upper)
        
        # Apply torques to arm joints
        self.data.ctrl[self.arm_actuator_ids] = torques
    
    def step(self, target_pos, dt=None):
        """Execute one control step towards target position."""
        # Get current eef position and Jacobian
        current_pos, jacobian = self.get_eef_jacobian()
        
        # Compute position error
        pos_error = target_pos - current_pos
        
        error_norm = np.linalg.norm(pos_error)
        if self._ema_error is None:
            self._ema_error = error_norm
        else:
            a = self.error_ema_alpha
            self._ema_error = a * error_norm + (1.0 - a) * self._ema_error

        # Convergence decision with hysteresis and dwell-time.
        enter_threshold = self.convergence_threshold
        exit_threshold = self.convergence_threshold + self.convergence_hysteresis
        if not self._converged:
            if self._ema_error < enter_threshold:
                self._below_threshold_count += 1
            else:
                self._below_threshold_count = 0
            if self._below_threshold_count >= self.convergence_required_steps:
                self._converged = True
        elif (not self.latch_convergence) and self._ema_error > exit_threshold:
            self._converged = False
            self._below_threshold_count = 0
        
        # Compute joint velocity using damped least squares + null-space posture term.
        damping = 0.1
        task_vel = pos_error * self.task_gain
        jj_t = jacobian @ jacobian.T
        inv = np.linalg.solve(jj_t + (damping ** 2) * np.eye(3), np.eye(3))
        j_pinv = jacobian.T @ inv

        desired_vel_task = j_pinv @ task_vel
        current_joint_pos = self.data.qpos[self.arm_joint_qpos_ids].copy()
        desired_vel_posture = self.posture_gain * (self.posture_ref - current_joint_pos)
        nullspace = np.eye(len(self.arm_joint_qvel_ids)) - (j_pinv @ jacobian)
        desired_vel = desired_vel_task + nullspace @ desired_vel_posture
        desired_vel = np.clip(desired_vel, -self.max_joint_speed, self.max_joint_speed)
        
        # Integrate to get joint position target
        if dt is None:
            dt = self.model.opt.timestep
        target_joint_pos = current_joint_pos + desired_vel * dt
        target_joint_pos = np.clip(target_joint_pos, self.arm_joint_lower, self.arm_joint_upper)
        
        if self.control_mode == "kinematic":
            # Directly update joint states in kinematic mode for smooth, stable tracking.
            joint_vel = (target_joint_pos - current_joint_pos) / max(dt, 1e-6)
            joint_vel = np.clip(joint_vel, -self.max_joint_speed, self.max_joint_speed)
            self.data.qpos[self.arm_joint_qpos_ids] = target_joint_pos
            self.data.qvel[self.arm_joint_qvel_ids] = joint_vel
        else:
            # Apply PD torque control.
            self.pd_control(target_joint_pos)
        
        return self._converged, error_norm

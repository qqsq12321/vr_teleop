"""Shared per-arm teleoperation controller.

This module owns the device-independent part of the arm/hand path:
wrist residual tracking, hand retargeting, IK, and MuJoCo joint mapping.
Simulation and real-robot entries should differ only in how they dispatch the
solved command.
"""

from __future__ import annotations

import numpy as np
import mujoco

from utils.core.body.ik import solve_body_pose_ik
from utils.core.body.wrist_tracker import WristTracker
from utils.core.hand.retarget import HandRetargeter
from utils.core.quaternion import matrix_to_quaternion


def slerp_step(current_quat: np.ndarray, target_quat: np.ndarray, gain: float) -> np.ndarray:
    """Move one fraction toward target_quat using SLERP."""
    q1 = np.asarray(current_quat, dtype=np.float64)
    q2 = np.asarray(target_quat, dtype=np.float64)
    if np.dot(q1, q2) < 0:
        q2 = -q2
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    angle = np.arccos(abs(dot))
    if angle < 1e-6:
        return q1
    step = min(angle * gain, angle)
    sin_a = np.sin(angle)
    w1 = np.sin(angle - step) / sin_a
    w2 = np.sin(step) / sin_a
    if dot < 0:
        w2 = -w2
    q = w1 * q1 + w2 * q2
    norm = np.linalg.norm(q)
    return q / norm if norm > 0.0 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


class TeleopArmController:
    """Per-arm controller shared by simulation and real robot runners."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik_data: mujoco.MjData,
        side: str,
        args,
        *,
        hand_config: str | None = None,
        hand_type: str = "",
        home_qpos: np.ndarray | None = None,
        robot_label: str = "robot",
        arm_joint_names: tuple[str, ...] | None = None,
        hand_joint_names: tuple[str, ...] | None = None,
        ee_body_name: str | None = None,
        base_body_name: str | None = "base_link",
        hand_qpos_mapping: tuple[int, ...] | list[int] | np.ndarray | None = None,
        position_deadband: float = 0.0,
        rotation_deadband_deg: float = 0.0,
        pos_gain: float | None = None,
        rot_gain: float | None = None,
    ):
        self.model = model
        self.data = data
        self.ik_data = ik_data
        self.side = side
        self.args = args
        self.robot_label = robot_label
        self.hand_type = hand_type
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain

        if ee_body_name is None:
            ee_body_name = f"{side}_j7"
        self.ee_body_name = ee_body_name
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        if self.ee_body_id == -1:
            raise ValueError(f"Body '{self.ee_body_name}' not found.")

        if arm_joint_names is None:
            raise ValueError("arm_joint_names must be provided.")
        self.arm_joint_ids = []
        self.arm_dof_indices = []
        for jname in arm_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(f"Joint '{jname}' not found.")
            self.arm_joint_ids.append(jid)
            self.arm_dof_indices.append(model.jnt_qposadr[jid])
        self.arm_dof_indices = np.array(self.arm_dof_indices, dtype=int)

        self.arm_home_qpos = (
            np.array(home_qpos, dtype=np.float64)
            if home_qpos is not None
            else np.zeros(len(self.arm_dof_indices), dtype=np.float64)
        )
        if self.arm_home_qpos.shape[0] != len(self.arm_dof_indices):
            raise ValueError(
                f"{self.robot_label} {side} home qpos size {self.arm_home_qpos.shape[0]} "
                f"!= arm DoF count {len(self.arm_dof_indices)}."
            )

        if hand_joint_names is None:
            raise ValueError("hand_joint_names must be provided.")
        self.hand_joint_ids = []
        self.hand_dof_indices = []
        self.hand_joint_names = []
        for jname in hand_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.hand_joint_ids.append(jid)
                self.hand_dof_indices.append(model.jnt_qposadr[jid])
                self.hand_joint_names.append(jname)
        self.hand_dof_indices = np.array(self.hand_dof_indices, dtype=int)

        self.arm_actuator_ids = [
            self._actuator_id_for_joint(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
            for jid in self.arm_joint_ids
        ]
        self.hand_actuator_ids = [self._actuator_id_for_joint(name) for name in self.hand_joint_names]

        self.hand_retargeter = HandRetargeter(hand_config, side) if hand_config else None
        self.hand_qpos_mapping = (
            np.array(hand_qpos_mapping, dtype=int) if hand_qpos_mapping is not None else None
        )
        self.latest_hand_qpos: np.ndarray | None = None

        initial_body_pos = data.xpos[self.ee_body_id].copy()
        initial_body_quat = matrix_to_quaternion(data.xmat[self.ee_body_id].reshape(3, 3).copy())
        # VR adapters already convert wrist translation into the project world frame
        # (X forward, Y left, Z up). Most existing scenes have their base frame
        # aligned with that world frame, so applying base_xmat is harmless/useful.
        # A spec may set base_body_name=None when the imported model is wrapped in
        # an alignment rotation: applying that rotation again would rotate only the
        # translation residual while the already-correct quaternion stays unchanged.
        base_xmat = None
        if base_body_name is not None:
            base_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
            )
            if base_body_id != -1:
                base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

        self.tracker = WristTracker(
            initial_body_pos,
            initial_body_quat,
            position_scale=args.position_scale,
            ema_alpha=args.ema_alpha,
            negate_rot_xy=False,
            base_xmat=base_xmat,
            position_deadband=position_deadband,
            rotation_deadband_deg=rotation_deadband_deg,
        )
        self.cmd_pos = initial_body_pos.copy()
        self.cmd_quat = np.array(initial_body_quat, dtype=np.float64)

    @property
    def target_position(self):
        return self.tracker.target_position

    @property
    def target_quaternion(self):
        return self.tracker.target_quaternion

    @property
    def latest_residual(self):
        return self.tracker.residual

    @property
    def latest_euler_residual(self):
        return self.tracker.euler_residual

    def sync_from_joint_positions(self, current_q: np.ndarray) -> None:
        """Sync MuJoCo arm state from real joints and reset the robot-side baseline."""
        current_q = np.asarray(current_q, dtype=np.float64)
        n = min(current_q.shape[0], len(self.arm_dof_indices))
        self.data.qpos[self.arm_dof_indices[:n]] = current_q[:n]
        mujoco.mj_forward(self.model, self.data)
        self.reset_baseline_from_current_pose()

    def reset_baseline_from_current_pose(self) -> None:
        self.tracker.reset_baseline(
            self.data.xpos[self.ee_body_id].copy(),
            matrix_to_quaternion(self.data.xmat[self.ee_body_id].reshape(3, 3).copy()),
        )
        self.cmd_pos = self.tracker.target_position.copy()
        self.cmd_quat = np.array(self.tracker.target_quaternion, dtype=np.float64)

    def update_from_pose(self, robot_position, robot_quaternion) -> None:
        self.tracker.update(robot_position, robot_quaternion)

    def update_hand_from_mediapipe(self, mediapipe_pts: np.ndarray | None) -> bool:
        if mediapipe_pts is None or self.hand_retargeter is None or not self.hand_retargeter.available:
            return False
        result = self.hand_retargeter.retarget_mediapipe(mediapipe_pts)
        if result is not None:
            self.latest_hand_qpos = result
            return True
        return False

    def update_hand_from_raw_landmarks(self, raw_landmarks: list[float] | tuple[float, ...] | None) -> bool:
        if raw_landmarks is None or self.hand_retargeter is None:
            return False
        result = self.hand_retargeter.retarget(raw_landmarks)
        if result is not None:
            self.latest_hand_qpos = result
            return True
        return False

    def mapped_hand_qpos(self) -> np.ndarray | None:
        if self.latest_hand_qpos is None:
            return None
        if self.hand_qpos_mapping is None:
            return self.latest_hand_qpos
        return self.latest_hand_qpos[self.hand_qpos_mapping]

    def step_ik(self) -> np.ndarray:
        return self.solve_ik(self.target_position, self.target_quaternion)

    def step_smoothed_ik(self, last_valid_packet_time: float, now: float, packet_timeout_s: float) -> np.ndarray | None:
        if not self.tracker.initialized or now - last_valid_packet_time > packet_timeout_s:
            return None
        if self.pos_gain is not None:
            self.cmd_pos += (self.tracker.target_position - self.cmd_pos) * self.pos_gain
        else:
            self.cmd_pos = self.tracker.target_position.copy()
        if self.rot_gain is not None:
            self.cmd_quat = slerp_step(self.cmd_quat, self.tracker.target_quaternion, self.rot_gain)
        else:
            self.cmd_quat = np.array(self.tracker.target_quaternion, dtype=np.float64)
        return self.solve_ik(self.cmd_pos, self.cmd_quat)

    def solve_ik(self, target_position: np.ndarray, target_quaternion: np.ndarray) -> np.ndarray:
        return solve_body_pose_ik(
            self.model,
            self.ik_data,
            self.ee_body_id,
            target_position,
            target_quaternion,
            self.data.qpos[: self.model.nq],
            rot_weight=self.args.rot_weight,
            damping=self.args.ik_damping,
            current_q_weight=self.args.ik_current_weight,
            dof_indices=self.arm_dof_indices,
            home_qpos=self.arm_home_qpos,
        )

    def write_solution_to_data(self, q_sol: np.ndarray, *, include_hand: bool = True) -> None:
        for joint_id in self.arm_joint_ids:
            qadr = self.model.jnt_qposadr[joint_id]
            if qadr < q_sol.shape[0]:
                self.data.qpos[qadr] = q_sol[qadr]

        if not include_hand:
            return
        mapped = self.mapped_hand_qpos()
        if mapped is not None and len(self.hand_dof_indices) == len(mapped):
            for qadr, value in zip(self.hand_dof_indices, mapped):
                self.data.qpos[qadr] = value

    def apply_qpos(self, q_sol: np.ndarray) -> None:
        self.write_solution_to_data(q_sol, include_hand=True)

    def apply_ctrl(self, q_sol: np.ndarray) -> None:
        for joint_id, actuator_id in zip(self.arm_joint_ids, self.arm_actuator_ids):
            qadr = self.model.jnt_qposadr[joint_id]
            if qadr < q_sol.shape[0] and not self._set_ctrl_target(actuator_id, q_sol[qadr]):
                self.data.qpos[qadr] = q_sol[qadr]

        mapped = self.mapped_hand_qpos()
        if mapped is not None and len(self.hand_dof_indices) == len(mapped):
            for qadr, actuator_id, value in zip(self.hand_dof_indices, self.hand_actuator_ids, mapped):
                if not self._set_ctrl_target(actuator_id, value):
                    self.data.qpos[qadr] = value

    def _actuator_id_for_joint(self, joint_name: str | None) -> int:
        if not joint_name:
            return -1
        candidates = (joint_name, joint_name.upper())
        for actuator_name in candidates:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if aid != -1:
                return aid
        return -1

    def _set_ctrl_target(self, actuator_id: int, value: float) -> bool:
        if actuator_id == -1:
            return False
        lo, hi = self.model.actuator_ctrlrange[actuator_id]
        self.data.ctrl[actuator_id] = float(np.clip(value, lo, hi))
        return True

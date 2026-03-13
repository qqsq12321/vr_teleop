from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import socket
import time
from pathlib import Path
from typing import Sequence, Optional

import mujoco
import numpy as np
from mujoco import viewer

from util.ik import solve_pose_ik
from util.quaternion import (
    matrix_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    transform_vr_to_robot_pose,
)
from util.udp_socket import (
    parse_right_wrist_pose,
    parse_right_landmarks,
    parse_left_wrist_pose,
    parse_left_landmarks,
    pinch_distance_from_landmarks,
    pinch_to_gripper,
)


def _default_scene_path() -> Path:
    # Use the existing aloha scene, which is usually at ../mujoco_menagerie/aloha/scene.xml relative to this file's parent dir
    return Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "aloha" / "scene.xml"


class ArmController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik_data: mujoco.MjData,
        side: str,  # "left" or "right"
        args: argparse.Namespace,
    ):
        self.model = model
        self.data = data
        self.ik_data = ik_data
        self.side = side
        self.args = args

        # Identify site
        site_name = f"{side}/gripper"
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            # Fallback for aloha scene might use prefix "aloha_scene/"? 
            # Or aloha scene includes aloha.xml so names should be preserved.
            pass
        if self.site_id == -1:
             raise ValueError(f"Site '{site_name}' not found.")

        # Identify gripper actuator
        gripper_actuator_name = f"{side}/gripper"
        self.gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name
        )
        
        # Identify arm joints (for IK)
        # We assume joints starting with "{side}/" are for this arm.
        # We exclude gripper joints from IK.
        self.arm_joint_ids = []
        self.arm_dof_indices = []
        
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not name:
                continue
            if name.startswith(f"{side}/"):
                # Exclude gripper fingers from IK
                if "finger" in name:
                    continue
                self.arm_joint_ids.append(jid)
                qadr = model.jnt_qposadr[jid]
                # Assuming 1-DoF joints
                self.arm_dof_indices.append(qadr)
        
        
        # Determine home qpos for this arm from default
        # Aloha Home pose approx (safe pose)
        left_qpos_default = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        right_qpos_default = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        
        # Create a home vector matching the arm_dof_indices size
        # We assume strict ordering of arm joints matches the default list above.
        # This is a bit fragile but matches _apply_initial_pose logic.
        self.arm_home_qpos = np.array(
            left_qpos_default if side == "left" else right_qpos_default, 
            dtype=np.float64
        )
        
        self.arm_dof_indices = np.array(self.arm_dof_indices, dtype=int)
        
        # Verify sizes
        if len(self.arm_home_qpos) != len(self.arm_dof_indices):
            print(f"Warning: Home qpos size {len(self.arm_home_qpos)} != DoF indices {len(self.arm_dof_indices)} for {side} arm. Regularization disabled.")
            self.arm_home_qpos = None

        # Initial state setup
        self.initial_site_pos = data.site_xpos[self.site_id].copy()
        self.initial_site_quat = matrix_to_quaternion(
            data.site_xmat[self.site_id].reshape(3, 3).copy()
        )

        base_body_name = f"{side}/base_link"
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.base_xmat = None
        if base_body_id != -1:
            self.base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

        # Teleop state
        self.initial_wrist_position = None
        self.initial_wrist_quaternion = None
        self.target_position = self.initial_site_pos.copy()
        self.target_quaternion = np.array(self.initial_site_quat, dtype=np.float64)
        
        self.latest_residual = None
        self.latest_euler_residual = None
        self.smoothed_residual = None
        self.latest_gripper_cmd = None

    def update(self, packet_msg: str):
        # Parse relevant data based on side
        if self.side == "right":
            wrist_pose = parse_right_wrist_pose(packet_msg)
            landmarks = parse_right_landmarks(packet_msg)
        else:
            wrist_pose = parse_left_wrist_pose(packet_msg)
            landmarks = parse_left_landmarks(packet_msg)

        # Gripper logic
        if landmarks is not None and self.gripper_actuator_id != -1:
            pinch_distance = pinch_distance_from_landmarks(landmarks)
            if pinch_distance is not None:
                self.latest_gripper_cmd = pinch_to_gripper(pinch_distance)

        # Pose logic
        if wrist_pose is not None:
            # Check if all wrist pose values are valid numbers before using them
            valid_pose = True
            for val in wrist_pose:
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    valid_pose = False
                    break
            
            if valid_pose:
                wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
                wrist_quaternion = (wrist_pose[3], wrist_pose[4], wrist_pose[5], wrist_pose[6])
                
                robot_position, robot_quaternion = transform_vr_to_robot_pose(
                    wrist_position, wrist_quaternion
                )

                if self.initial_wrist_position is None:
                    self.initial_wrist_position = robot_position
                    self.initial_wrist_quaternion = robot_quaternion
                else:
                    residual = np.array(
                        [
                            robot_position[0] - self.initial_wrist_position[0],
                            robot_position[1] - self.initial_wrist_position[1],
                            robot_position[2] - self.initial_wrist_position[2],
                        ],
                        dtype=np.float64,
                    )
                    if self.base_xmat is not None:
                        residual = self.base_xmat @ residual
                    
                    if self.smoothed_residual is None:
                        self.smoothed_residual = residual
                    else:
                        self.smoothed_residual = (
                            self.args.ema_alpha * residual
                            + (1.0 - self.args.ema_alpha) * self.smoothed_residual
                        )
                    
                    self.target_position = (
                        self.initial_site_pos + self.args.position_scale * self.smoothed_residual
                    )

                    relative_quaternion = quaternion_multiply(
                        robot_quaternion,
                        quaternion_inverse(self.initial_wrist_quaternion),
                    )
                    self.target_quaternion = np.array(
                        quaternion_multiply(relative_quaternion, self.initial_site_quat),
                        dtype=np.float64,
                    )
                    norm = np.linalg.norm(self.target_quaternion)
                    if norm > 0.0:
                        self.target_quaternion /= norm
                    
                    euler_residual = quaternion_to_euler_xyz(
                        relative_quaternion[0],
                        relative_quaternion[1],
                        relative_quaternion[2],
                        relative_quaternion[3],
                    )
                    self.latest_residual = self.smoothed_residual
                    self.latest_euler_residual = euler_residual

    def step_ik(self):
        # Solve IK specifically for this arm's DoFs
        q_sol = solve_pose_ik(
            self.model,
            self.ik_data,
            self.site_id,
            self.target_position,
            self.target_quaternion,
            self.data.qpos[: self.model.nq],
            rot_weight=self.args.rot_weight,
            damping=self.args.ik_damping,
            current_q_weight=self.args.ik_current_weight,
            dof_indices=self.arm_dof_indices,
            home_qpos=self.arm_home_qpos,
        )
        return q_sol

    def apply_control(self, q_sol: np.ndarray):
        if self.model.nu:
            # Apply joint controls
             for act_id in range(self.model.nu):
                joint_id = self.model.actuator_trnid[act_id, 0]
                if joint_id < 0:
                    continue
                if joint_id in self.arm_joint_ids:
                    qadr = self.model.jnt_qposadr[joint_id]
                    if qadr < q_sol.shape[0]:
                        self.data.ctrl[act_id] = q_sol[qadr]
            
            # Apply gripper control
             if self.latest_gripper_cmd is not None and self.gripper_actuator_id != -1:
                self.data.ctrl[self.gripper_actuator_id] = self.latest_gripper_cmd


def _apply_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Set initial pose for Aloha (both arms)
    # Aloha Home pose approx (safe pose)
    left_qpos = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    right_qpos = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]

    # Map to joints
    for i, name in enumerate(["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]):
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left/{name}")
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right/{name}")
        if lid != -1:
            data.qpos[model.jnt_qposadr[lid]] = left_qpos[i]
        if rid != -1:
            data.qpos[model.jnt_qposadr[rid]] = right_qpos[i]

    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bimanual Teleop for Aloha.")
    parser.add_argument(
        "--scene",
        default=str(_default_scene_path()),
        help="Path to a MuJoCo XML scene file.",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--position-scale",
        type=float,
        default=3.0,
        help="Scale for wrist position residuals.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help="EMA smoothing factor for wrist residuals (0-1).",
    )
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=1.0,
        help="Weight for orientation error in IK.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1e-3,
        help="Damping factor for IK solver.",
    )
    parser.add_argument(
        "--ik-current-weight",
        type=float,
        default=0.1,
        help="Weight for penalizing deviation from current pose in IK.",
    )
    args = parser.parse_args()

    xml_path = Path(args.scene).expanduser().resolve()
    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.004 # Increase timestep to 4ms to double simulation speed per step
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose(model, data)
    
    # Initialize controllers for both arms
    left_arm = ArmController(model, data, ik_data, "left", args)
    right_arm = ArmController(model, data, ik_data, "right", args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)

    last_log_time = time.time()

    sync_start_time = time.time()
    
    with viewer.launch_passive(model, data) as vis:
        while vis.is_running():
            # Process all packets in buffer to ensure smooth trajectory integration
            try:
                sock_data, _ = sock.recvfrom(4096)
                message = sock_data.decode("utf-8", errors="ignore")
                left_arm.update(message)
                right_arm.update(message)
            except BlockingIOError:
                pass

            q_left = left_arm.step_ik()
            q_right = right_arm.step_ik()
            
            left_arm.apply_control(q_left)
            right_arm.apply_control(q_right)

            # Step physics
            mujoco.mj_step(model, data)
            
            vis.sync()

            now = time.time()
            if now - last_log_time > 1.0:
                print(f"L resid: {left_arm.latest_residual} R resid: {right_arm.latest_residual}")
                last_log_time = now

            # sleep_time = model.opt.timestep
            sleep_time = 0.0001   
            if sleep_time > 0:
                time.sleep(sleep_time)



if __name__ == "__main__":
    main()

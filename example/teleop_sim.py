"""Teleoperate a simulated robot arm via VR hand tracking (MuJoCo).

Supports multiple robot configurations via --robot:
  piper          Piper single-arm with pinch gripper
  kinova_gripper Kinova Gen3 + Robotiq 2F-85 gripper
  kinova_wuji    Kinova Gen3 + Wuji dexterous hand (20 DOF)
  aloha          Aloha bimanual (dual 6-DOF arms + grippers)

Input sources (--input-source):
  quest3         Meta Quest 3 via UDP (default)
  avp            Apple Vision Pro via avp_stream / Tracking Streamer

Examples:
    python example/teleop_sim.py --robot piper --port 9000
    python example/teleop_sim.py --robot kinova_gripper
    python example/teleop_sim.py --robot kinova_gripper --input-source avp --avp-ip 192.168.1.100
    python example/teleop_sim.py --robot kinova_wuji --hand-config path/to/config.yaml
    python example/teleop_sim.py --robot aloha --position-scale 3.0
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"))

import argparse
import time
from pathlib import Path

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
    make_socket,
    recv_latest_packet,
    parse_left_landmarks,
    parse_left_wrist_pose,
    parse_right_landmarks,
    parse_right_wrist_pose,
    pinch_distance_from_landmarks,
    pinch_to_gripper,
)
from util.wrist_tracker import WristTracker
from util.hand_retarget import HandRetargeter

# ---------------------------------------------------------------------------
# Scene / config paths
# ---------------------------------------------------------------------------

_SCENE_DIR = Path(__file__).resolve().parent / "scene"
_ALOHA_SCENE = _SCENE_DIR / "aloha" / "scene.xml"


# ---------------------------------------------------------------------------
# Robot configurations
# ---------------------------------------------------------------------------

ROBOT_CONFIGS = {
    "piper": {
        "scene_xml": str(_SCENE_DIR / "scene_piper.xml"),
        "site_name": "piper_ee_site",
        "site_fallback": "ee_site",
        "base_body_name": "piper_base_link",
        "base_body_fallback": "base_link",
        "home_qpos": np.array([0.0, 0.9, -0.9, 0.0, 0.4, 0.0, 0.0], dtype=np.float64),
        "position_scale": 1.5,
        "hand_type": "gripper",
        "negate_rot_xy": False,
        "skip_tail_joints": 0,
        "gripper_actuator_name": "piper_gripper",
        "gripper_actuator_fallback": "gripper",
        "gripper_joint_names": ["piper_joint7", "piper_joint8", "joint7", "joint8"],
        "pinch_max_distance": 0.1,
        "gripper_max": 0.035,
        "bimanual": False,
    },
    "kinova_gripper": {
        "scene_xml": str(_SCENE_DIR / "scene_kinova_gen3.xml"),
        "site_name": "kinova_ee_site",
        "site_fallback": None,
        "base_body_name": "base_link",
        "base_body_fallback": None,
        "home_qpos": np.array(
            [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633],
            dtype=np.float64,
        ),
        "position_scale": 1.5,
        "hand_type": "gripper",
        "negate_rot_xy": True,
        "skip_tail_joints": 0,
        "gripper_actuator_name": "fingers_actuator",
        "gripper_actuator_fallback": None,
        "gripper_joint_names": [],
        "pinch_max_distance": 0.1,
        "gripper_max": 0.8,
        "bimanual": False,
    },
    "kinova_wuji": {
        "scene_xml": str(_SCENE_DIR / "scene_kinova_gen3_wuji.xml"),
        "site_name": "kinova_ee_site",
        "site_fallback": None,
        "base_body_name": "base_link",
        "base_body_fallback": None,
        "home_qpos": np.array(
            [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633],
            dtype=np.float64,
        ),
        "negate_rot_xy": True,
        "position_scale": 1.5,
        "hand_type": "wuji",
        "negate_rot_xy": True,
        "skip_tail_joints": 0,
        "num_arm_actuators": 7,
        "num_hand_actuators": 20,
        "bimanual": False,
    },
    "aloha": {
        "scene_xml": str(_ALOHA_SCENE),
        "position_scale": 3.0,
        "bimanual": True,
    },
}

# ---------------------------------------------------------------------------
# Gripper helpers
# ---------------------------------------------------------------------------


def _pinch_to_gripper(pinch_distance: float, pinch_max: float, gripper_max: float) -> float:
    """Map pinch distance to gripper control value (generic)."""
    scaled = pinch_distance / pinch_max
    clamped = min(1.0, max(0.0, scaled))
    return gripper_max * (1.0 - clamped)


# ---------------------------------------------------------------------------
# Aloha bimanual ArmController (kept from original teleop_bimanual.py)
# ---------------------------------------------------------------------------


class ArmController:
    """Per-arm controller for bimanual teleop (Aloha)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik_data: mujoco.MjData,
        side: str,
        args: argparse.Namespace,
    ):
        self.model = model
        self.data = data
        self.ik_data = ik_data
        self.side = side
        self.args = args

        site_name = f"{side}/gripper"
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"Site '{site_name}' not found.")

        gripper_actuator_name = f"{side}/gripper"
        self.gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name
        )

        self.arm_joint_ids = []
        self.arm_dof_indices = []
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not name:
                continue
            if name.startswith(f"{side}/") and "finger" not in name:
                self.arm_joint_ids.append(jid)
                self.arm_dof_indices.append(model.jnt_qposadr[jid])

        home_qpos_default = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        self.arm_home_qpos = np.array(home_qpos_default, dtype=np.float64)
        self.arm_dof_indices = np.array(self.arm_dof_indices, dtype=int)

        if len(self.arm_home_qpos) != len(self.arm_dof_indices):
            print(
                f"Warning: Home qpos size {len(self.arm_home_qpos)} != "
                f"DoF indices {len(self.arm_dof_indices)} for {side} arm. "
                "Regularization disabled."
            )
            self.arm_home_qpos = None

        self.initial_site_pos = data.site_xpos[self.site_id].copy()
        self.initial_site_quat = matrix_to_quaternion(
            data.site_xmat[self.site_id].reshape(3, 3).copy()
        )
        base_body_name = f"{side}/base_link"
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.base_xmat = None
        if base_body_id != -1:
            self.base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

        self.initial_wrist_position = None
        self.initial_wrist_quaternion = None
        self.target_position = self.initial_site_pos.copy()
        self.target_quaternion = np.array(self.initial_site_quat, dtype=np.float64)
        self.latest_residual = None
        self.latest_euler_residual = None
        self.smoothed_residual = None
        self.latest_gripper_cmd = None

    def update(self, packet_msg: str):
        if self.side == "right":
            wrist_pose = parse_right_wrist_pose(packet_msg)
            landmarks = parse_right_landmarks(packet_msg)
        else:
            wrist_pose = parse_left_wrist_pose(packet_msg)
            landmarks = parse_left_landmarks(packet_msg)

        if landmarks is not None and self.gripper_actuator_id != -1:
            pinch_distance = pinch_distance_from_landmarks(landmarks)
            if pinch_distance is not None:
                self.latest_gripper_cmd = pinch_to_gripper(pinch_distance)

        if wrist_pose is not None:
            valid_pose = all(
                isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)
                for v in wrist_pose
            )
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
                    self.latest_residual = self.smoothed_residual
                    self.latest_euler_residual = quaternion_to_euler_xyz(
                        relative_quaternion[0],
                        relative_quaternion[1],
                        relative_quaternion[2],
                        relative_quaternion[3],
                    )

    def step_ik(self):
        return solve_pose_ik(
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

    def apply_control(self, q_sol: np.ndarray):
        if self.model.nu:
            for act_id in range(self.model.nu):
                joint_id = self.model.actuator_trnid[act_id, 0]
                if joint_id < 0:
                    continue
                if joint_id in self.arm_joint_ids:
                    qadr = self.model.jnt_qposadr[joint_id]
                    if qadr < q_sol.shape[0]:
                        self.data.ctrl[act_id] = q_sol[qadr]
            if self.latest_gripper_cmd is not None and self.gripper_actuator_id != -1:
                self.data.ctrl[self.gripper_actuator_id] = self.latest_gripper_cmd


# ---------------------------------------------------------------------------
# Initial pose helpers
# ---------------------------------------------------------------------------


def _apply_initial_pose_single(model: mujoco.MjModel, data: mujoco.MjData, home_qpos: np.ndarray) -> None:
    n = min(model.nq, home_qpos.shape[0])
    data.qpos[:n] = home_qpos[:n]
    n_ctrl = min(model.nu, n)
    if n_ctrl:
        data.ctrl[:n_ctrl] = home_qpos[:n_ctrl]
    mujoco.mj_forward(model, data)


def _apply_initial_pose_kinova_wuji(model: mujoco.MjModel, data: mujoco.MjData, home_qpos: np.ndarray, num_arm_actuators: int) -> None:
    n = min(model.nq, home_qpos.shape[0])
    data.qpos[:n] = home_qpos[:n]
    n_ctrl = min(num_arm_actuators, model.nu)
    if n_ctrl:
        data.ctrl[:n_ctrl] = home_qpos[:n_ctrl]
    mujoco.mj_forward(model, data)


def _apply_initial_pose_aloha(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    left_qpos = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    right_qpos = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    for i, name in enumerate(
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ):
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left/{name}")
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right/{name}")
        if lid != -1:
            data.qpos[model.jnt_qposadr[lid]] = left_qpos[i]
        if rid != -1:
            data.qpos[model.jnt_qposadr[rid]] = right_qpos[i]
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Single-arm simulation loop
# ---------------------------------------------------------------------------


def _run_single_arm(config: dict, args: argparse.Namespace) -> None:
    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    home_qpos = config["home_qpos"]
    hand_type = config.get("hand_type")

    # Apply initial pose
    if hand_type == "wuji":
        _apply_initial_pose_kinova_wuji(model, data, home_qpos, config.get("num_arm_actuators", 7))
    else:
        _apply_initial_pose_single(model, data, home_qpos)

    # Find EE site
    site_name = args.site or config["site_name"]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1 and config.get("site_fallback"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config["site_fallback"])
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in model.")

    # Find gripper actuator (for gripper-type robots)
    gripper_actuator_id = -1
    gripper_joint_ids = []
    if hand_type == "gripper":
        gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, config["gripper_actuator_name"]
        )
        if gripper_actuator_id == -1 and config.get("gripper_actuator_fallback"):
            gripper_actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, config["gripper_actuator_fallback"]
            )
        for jname in config.get("gripper_joint_names", []):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                gripper_joint_ids.append(jid)

    # Load retargeter for wuji hand
    hand_retargeter = None
    if hand_type == "wuji":
        hand_retargeter = HandRetargeter(args.hand_config, args.hand_side)

    # Initial EE state
    mujoco.mj_forward(model, data)
    initial_site_pos = data.site_xpos[site_id].copy()
    initial_site_quat = matrix_to_quaternion(
        data.site_xmat[site_id].reshape(3, 3).copy()
    )

    # Base body transform
    base_body_name = config.get("base_body_name", "base_link")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_body_id == -1 and config.get("base_body_fallback"):
        base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, config["base_body_fallback"]
        )
    base_xmat = None
    if base_body_id != -1:
        base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

    # Input source
    sock = None
    avp_input = None
    if args.input_source == "avp":
        from util.avp_input import AVPInput
        avp_input = AVPInput(ip=args.avp_ip)
        print(f"  Input: Apple Vision Pro ({args.avp_ip})")
    else:
        sock = make_socket(args.port)
        print(f"  Input: Quest 3 (UDP port {args.port})")

    # Wrist tracker
    tracker = WristTracker(
        initial_site_pos,
        initial_site_quat,
        position_scale=args.position_scale,
        ema_alpha=args.ema_alpha,
        negate_rot_xy=config.get("negate_rot_xy", False) if args.input_source != "avp" else False,
        base_xmat=base_xmat,
    )

    # State variables
    last_log_time = time.time()
    latest_gripper_cmd = None
    latest_hand_qpos = None

    num_arm_actuators = config.get("num_arm_actuators", model.nu)
    num_hand_actuators = config.get("num_hand_actuators", 0)

    with viewer.launch_passive(model, data) as vis:
        vis.cam.azimuth = model.vis.global_.azimuth
        vis.cam.elevation = model.vis.global_.elevation
        vis.cam.distance = model.stat.extent * 1.5
        vis.cam.lookat[:] = model.stat.center

        while vis.is_running():
            loop_start = time.time()

            # --- Receive input data ---
            if avp_input is not None:
                # --- Apple Vision Pro path ---
                if avp_input.poll():
                    # Hand retargeting (wuji)
                    if hand_type == "wuji" and hand_retargeter is not None and hand_retargeter.available:
                        mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                        if mediapipe_pts is not None:
                            result = hand_retargeter._retargeter.retarget(mediapipe_pts)
                            if result is not None:
                                latest_hand_qpos = result

                    # Gripper from pinch distance
                    if hand_type == "gripper" and gripper_actuator_id != -1:
                        pinch_distance = avp_input.get_pinch_distance("right")
                        if pinch_distance is not None:
                            latest_gripper_cmd = _pinch_to_gripper(
                                pinch_distance,
                                config["pinch_max_distance"],
                                config["gripper_max"],
                            )

                    # Arm: wrist pose
                    wrist = avp_input.get_wrist_pose("right")
                    if wrist is not None:
                        robot_position, robot_quaternion = wrist
                        tracker.update(robot_position, robot_quaternion)
            else:
                # --- Quest 3 path (unchanged) ---
                packet = recv_latest_packet(sock)

                if packet is not None:
                    message = packet.decode("utf-8", errors="ignore")

                    # Hand retargeting from landmarks (wuji)
                    if hand_type == "wuji" and hand_retargeter is not None:
                        landmarks = parse_right_landmarks(message)
                        if landmarks is not None:
                            result = hand_retargeter.retarget(landmarks)
                            if result is not None:
                                latest_hand_qpos = result

                    # Gripper from landmarks (gripper type)
                    if hand_type == "gripper" and gripper_actuator_id != -1:
                        landmarks = parse_right_landmarks(message)
                        if landmarks is not None:
                            pinch_distance = pinch_distance_from_landmarks(landmarks)
                            if pinch_distance is not None:
                                latest_gripper_cmd = _pinch_to_gripper(
                                    pinch_distance,
                                    config["pinch_max_distance"],
                                    config["gripper_max"],
                                )

                    # Arm: wrist pose residuals
                    wrist_pose = parse_right_wrist_pose(message)
                    if wrist_pose is not None:
                        wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
                        wrist_quaternion = (
                            wrist_pose[3], wrist_pose[4], wrist_pose[5], wrist_pose[6]
                        )
                        robot_position, robot_quaternion = transform_vr_to_robot_pose(
                            wrist_position, wrist_quaternion
                        )
                        tracker.update(robot_position, robot_quaternion)

            # --- Logging ---
            now = time.time()
            if (
                tracker.residual is not None
                and tracker.euler_residual is not None
                and now - last_log_time > 1.0
            ):
                print(
                    f"Wrist residual (xyz): {tracker.residual.tolist()} "
                    f"euler: {list(tracker.euler_residual)}"
                )
                last_log_time = now

            # --- IK for arm ---
            q_sol = solve_pose_ik(
                model,
                ik_data,
                site_id,
                tracker.target_position,
                tracker.target_quaternion,
                data.qpos[: model.nq],
                rot_weight=args.rot_weight,
                damping=args.ik_damping,
                current_q_weight=args.ik_current_weight,
                home_qpos=home_qpos,
                skip_tail_joints=config.get("skip_tail_joints", 0),
            )

            # --- Apply arm controls ---
            if hand_type == "wuji":
                # Wuji: write arm ctrl directly (indices 0..num_arm-1)
                for i in range(num_arm_actuators):
                    joint_id = model.actuator_trnid[i, 0]
                    qadr = model.jnt_qposadr[joint_id]
                    data.ctrl[i] = q_sol[qadr]
                # Write hand ctrl
                if latest_hand_qpos is not None:
                    data.ctrl[num_arm_actuators : num_arm_actuators + num_hand_actuators] = (
                        latest_hand_qpos
                    )
            else:
                # Piper / Kinova gripper: skip gripper joints in IK output
                if model.nu:
                    ctrl = data.ctrl.copy()
                    for act_id in range(model.nu):
                        if model.actuator_trntype[act_id] != 0:  # mjTRN_JOINT = 0
                            continue
                        joint_id = model.actuator_trnid[act_id, 0]
                        if joint_id < 0 or joint_id in gripper_joint_ids:
                            continue
                        qadr = model.jnt_qposadr[joint_id]
                        if qadr < q_sol.shape[0]:
                            ctrl[act_id] = q_sol[qadr]
                    data.ctrl[:] = ctrl
                if latest_gripper_cmd is not None and gripper_actuator_id != -1:
                    data.ctrl[gripper_actuator_id] = latest_gripper_cmd

            # --- Step simulation ---
            mujoco.mj_step(model, data)
            vis.sync()

            sleep_time = model.opt.timestep - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# AVP helper for bimanual ArmController
# ---------------------------------------------------------------------------


def _update_arm_from_avp(arm: ArmController, avp_input, side: str) -> None:
    """Update an ArmController from AVP input (mirrors arm.update for Quest)."""
    # Gripper from pinch distance
    if arm.gripper_actuator_id != -1:
        pinch = avp_input.get_pinch_distance(side)
        if pinch is not None:
            arm.latest_gripper_cmd = pinch_to_gripper(pinch)

    # Wrist pose
    wrist = avp_input.get_wrist_pose(side)
    if wrist is not None:
        robot_position, robot_quaternion = wrist
        if arm.initial_wrist_position is None:
            arm.initial_wrist_position = robot_position
            arm.initial_wrist_quaternion = robot_quaternion
        else:
            residual = np.array(
                [
                    robot_position[0] - arm.initial_wrist_position[0],
                    robot_position[1] - arm.initial_wrist_position[1],
                    robot_position[2] - arm.initial_wrist_position[2],
                ],
                dtype=np.float64,
            )
            if arm.base_xmat is not None:
                residual = arm.base_xmat @ residual
            if arm.smoothed_residual is None:
                arm.smoothed_residual = residual
            else:
                arm.smoothed_residual = (
                    arm.args.ema_alpha * residual
                    + (1.0 - arm.args.ema_alpha) * arm.smoothed_residual
                )
            arm.target_position = (
                arm.initial_site_pos + arm.args.position_scale * arm.smoothed_residual
            )
            relative_quaternion = quaternion_multiply(
                robot_quaternion,
                quaternion_inverse(arm.initial_wrist_quaternion),
            )
            arm.target_quaternion = np.array(
                quaternion_multiply(relative_quaternion, arm.initial_site_quat),
                dtype=np.float64,
            )
            norm = np.linalg.norm(arm.target_quaternion)
            if norm > 0.0:
                arm.target_quaternion /= norm
            arm.latest_residual = arm.smoothed_residual
            arm.latest_euler_residual = quaternion_to_euler_xyz(
                relative_quaternion[0],
                relative_quaternion[1],
                relative_quaternion[2],
                relative_quaternion[3],
            )


# ---------------------------------------------------------------------------
# Aloha bimanual simulation loop
# ---------------------------------------------------------------------------


def _run_bimanual(config: dict, args: argparse.Namespace) -> None:
    xml_path = Path(args.scene).expanduser().resolve()
    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.004
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose_aloha(model, data)

    left_arm = ArmController(model, data, ik_data, "left", args)
    right_arm = ArmController(model, data, ik_data, "right", args)

    # Input source
    sock = None
    avp_input = None
    if args.input_source == "avp":
        from util.avp_input import AVPInput
        avp_input = AVPInput(ip=args.avp_ip)
        print(f"  Input: Apple Vision Pro ({args.avp_ip})")
    else:
        sock = make_socket(args.port)
        print(f"  Input: Quest 3 (UDP port {args.port})")

    last_log_time = time.time()

    with viewer.launch_passive(model, data) as vis:
        while vis.is_running():
            if avp_input is not None:
                # --- Apple Vision Pro path ---
                if avp_input.poll():
                    _update_arm_from_avp(left_arm, avp_input, "left")
                    _update_arm_from_avp(right_arm, avp_input, "right")
            else:
                # --- Quest 3 path (unchanged) ---
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

            mujoco.mj_step(model, data)
            vis.sync()

            now = time.time()
            if now - last_log_time > 1.0:
                print(f"L resid: {left_arm.latest_residual} R resid: {right_arm.latest_residual}")
                last_log_time = now

            time.sleep(0.0001)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate robot teleoperation with Quest 3 hand tracking.",
    )
    parser.add_argument(
        "--robot",
        required=True,
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot configuration to use.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Override scene XML path (default: per-robot default).",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--site",
        default=None,
        help="Override end-effector site name.",
    )
    parser.add_argument(
        "--hand-config",
        default=None,
        help="Path to retargeter YAML config (kinova_wuji only).",
    )
    parser.add_argument(
        "--hand-side",
        default="right",
        choices=["left", "right"],
        help="Hand side for retargeting (kinova_wuji only).",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=None,
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
    parser.add_argument(
        "--input-source",
        default="quest3",
        choices=["quest3", "avp"],
        help="Input device: quest3 (UDP, default) or avp (Vision Pro via avp_stream).",
    )
    parser.add_argument(
        "--avp-ip",
        default="192.168.1.100",
        help="Apple Vision Pro IP address (used with --input-source avp).",
    )
    args = parser.parse_args()

    config = ROBOT_CONFIGS[args.robot]

    # Apply defaults from config if user didn't override
    if args.scene is None:
        args.scene = config["scene_xml"]
    if args.position_scale is None:
        args.position_scale = config["position_scale"]

    if config.get("bimanual", False):
        _run_bimanual(config, args)
    else:
        _run_single_arm(config, args)


if __name__ == "__main__":
    main()

"""Teleoperate a simulated robot arm via VR hand tracking (MuJoCo).

Supports multiple robot configurations via --robot:
  piper          Piper single-arm with pinch gripper
  kinova_gripper Kinova Gen3 + Robotiq 2F-85 gripper
  kinova_wuji    Kinova Gen3 + Wuji dexterous hand (20 DOF)
  rm65           Realman RM65 6-DOF arm (no end-effector)
  dexforce       DexForce dual-arm + dual-hand humanoid (bimanual)
  aloha          Aloha bimanual (dual 6-DOF arms + grippers)

Input sources (--input-source):
  quest3         Meta Quest 3 via UDP (default)
  avp            Apple Vision Pro via avp_stream / Tracking Streamer

Examples:
    python example/teleop_sim.py --robot piper --port 9000
    python example/teleop_sim.py --robot kinova_gripper
    python example/teleop_sim.py --robot kinova_gripper --input-source avp --avp-ip 192.168.1.100
    python example/teleop_sim.py --robot kinova_wuji --hand-config path/to/config.yaml
    python example/teleop_sim.py --robot dexforce --input-source pico4
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
from util.ik import solve_body_pose_ik
from util.quaternion import (
    matrix_to_quaternion,
    transform_vr_to_robot_pose,
)
from util.udp_socket import (
    clamp_pinch_ratio,
    make_socket,
    recv_latest_packet,
    parse_left_landmarks,
    parse_left_wrist_pose,
    parse_right_landmarks,
    parse_right_wrist_pose,
    pinch_distance_from_landmarks,
)
from util.wrist_tracker import WristTracker
from util.hand_retarget import (
    HandRetargeter,
    default_inspire_config_path,
    default_linker_l20_config_path,
    default_pico4_config_path,
)

# ---------------------------------------------------------------------------
# Scene / config paths
# ---------------------------------------------------------------------------

_SCENE_DIR = Path(__file__).resolve().parent / "scene"
_ALOHA_SCENE = _SCENE_DIR / "aloha" / "scene.xml"
_DEXFORCE_SCENE = _SCENE_DIR / "scene_dexforce.xml"
_DEXFORCE_HAND_TYPE = "linker_l20"
_DEXFORCE_LEFT_ARM_HOME_QPOS = np.array(
    [0.0, -0.6, 0.0, -1.57, 0.0, -0.4, 0.0],
    dtype=np.float64,
)
_DEXFORCE_RIGHT_ARM_HOME_QPOS = np.array(
    [0.0, 0.6, 0.0, 1.57, 0.0, 0.4, 0.0],
    dtype=np.float64,
)


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
        "position_scale": 1.5,
        "hand_type": "wuji",
        "negate_rot_xy": True,
        "skip_tail_joints": 0,
        "num_arm_actuators": 7,
        "num_hand_actuators": 20,
        "bimanual": False,
    },
    "rm65": {
        "scene_xml": str(_SCENE_DIR / "scene_rm65.xml"),
        "site_name": "rm65_ee_site",
        "site_fallback": None,
        "base_body_name": "base_link",
        "base_body_fallback": None,
        "home_qpos": np.array(
            [0.0, -0.522, -1.81, 0.0099, 0.782, 0.0],
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
        "gripper_max": 0.82,
        "pinch_inverted": True,
        "bimanual": False,
    },
    "rm65_inspire": {
        "scene_xml": str(_SCENE_DIR / "scene_rm65_inspire.xml"),
        "site_name": "rm65_ee_site",
        "site_fallback": None,
        "base_body_name": "base_link",
        "base_body_fallback": None,
        "home_qpos": np.array(
            [0.0, -0.522, -1.81, 0.0099, 0.782, 0.0],
            dtype=np.float64,
        ),
        "position_scale": 1.5,
        "hand_type": "inspire",
        "negate_rot_xy": True,
        "skip_tail_joints": 0,
        "num_arm_actuators": 6,
        "num_hand_actuators": 12,
        "hand_qpos_mapping": [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5],
        "bimanual": False,
    },
    "dexforce": {
        "scene_xml": str(_DEXFORCE_SCENE),
        "position_scale": 3.0,
        "bimanual": True,
        "dexforce": True,
        "dexforce_hand_type": _DEXFORCE_HAND_TYPE,
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


def _pinch_to_gripper(
    pinch_distance: float, pinch_max: float, gripper_max: float, inverted: bool = False
) -> float:
    """Map pinch distance to gripper control value.

    ``inverted=False``: ctrl = gripper_max when pinched, 0 when open
    (normally-open grippers like Robotiq 2F-85).
    ``inverted=True``: reverse for normally-closed grippers like EG2-4C2.
    """
    ratio = clamp_pinch_ratio(pinch_distance, pinch_max)
    return gripper_max * ratio if inverted else gripper_max * (1.0 - ratio)


def _resolve_hand_config(args: argparse.Namespace, hand_type: str, side: str = "right") -> str | None:
    if args.hand_config:
        return args.hand_config
    if args.input_source == "pico4" and hand_type in ("wuji", "inspire", "linker_l20"):
        return str(default_pico4_config_path(hand_type, side=side))
    if hand_type == "inspire":
        return str(default_inspire_config_path())
    if hand_type == "linker_l20":
        return str(default_linker_l20_config_path(args.input_source, side=side))
    return None


# ---------------------------------------------------------------------------
# Aloha bimanual ArmController (kept from original teleop_bimanual.py)
# ---------------------------------------------------------------------------


def _wrist_pose_is_finite(wrist_pose) -> bool:
    return all(
        isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)
        for v in wrist_pose
    )


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

        initial_site_pos = data.site_xpos[self.site_id].copy()
        initial_site_quat = matrix_to_quaternion(
            data.site_xmat[self.site_id].reshape(3, 3).copy()
        )
        base_body_name = f"{side}/base_link"
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        base_xmat = None
        if base_body_id != -1:
            base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

        self.tracker = WristTracker(
            initial_site_pos,
            initial_site_quat,
            position_scale=args.position_scale,
            ema_alpha=args.ema_alpha,
            negate_rot_xy=False,
            base_xmat=base_xmat,
        )
        self.latest_gripper_cmd = None

    # -- pose state shims (keep existing call sites untouched) --------------

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

    # -- input decoupled from pose math -------------------------------------

    def update_from_pose(self, robot_position, robot_quaternion) -> None:
        self.tracker.update(robot_position, robot_quaternion)

    def update_gripper_from_pinch(self, pinch_distance) -> None:
        if self.gripper_actuator_id == -1 or pinch_distance is None:
            return
        self.latest_gripper_cmd = pinch_to_gripper(pinch_distance)

    def update(self, packet_msg: str) -> None:
        """Quest 3 path: decode UDP message, then update pose / gripper."""
        if self.side == "right":
            wrist_pose = parse_right_wrist_pose(packet_msg)
            landmarks = parse_right_landmarks(packet_msg)
        else:
            wrist_pose = parse_left_wrist_pose(packet_msg)
            landmarks = parse_left_landmarks(packet_msg)

        if landmarks is not None:
            self.update_gripper_from_pinch(pinch_distance_from_landmarks(landmarks))

        if wrist_pose is not None and _wrist_pose_is_finite(wrist_pose):
            wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
            wrist_quaternion = (wrist_pose[3], wrist_pose[4], wrist_pose[5], wrist_pose[6])
            robot_position, robot_quaternion = transform_vr_to_robot_pose(
                wrist_position, wrist_quaternion
            )
            self.update_from_pose(robot_position, robot_quaternion)

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


class DexForceArmController:
    """Per-arm controller for DexForce bimanual teleop."""

    _ARM_JOINT_SUFFIXES = [f"J{i}" for i in range(1, 8)]
    _HAND_JOINT_SUFFIXES = [
        "T_CMC_ROLL",
        "T_CMC_YAW",
        "T_CMC_PITCH",
        "T_MCP",
        "T_IP",
        "IF_MCP_ROLL",
        "IF_MCP_PITCH",
        "IF_PIP",
        "IF_DIP",
        "MF_MCP_ROLL",
        "MF_MCP_PITCH",
        "MF_PIP",
        "MF_DIP",
        "RF_MCP_ROLL",
        "RF_MCP_PITCH",
        "RF_PIP",
        "RF_DIP",
        "LF_MCP_ROLL",
        "LF_MCP_PITCH",
        "LF_PIP",
        "LF_DIP",
    ]

    # qpos_mapping: retarget URDF joint order -> MuJoCo joint order, keyed by hand type
    _HAND_QPOS_MAPPING = {
        "linker_l20": [16, 17, 18, 19, 20, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik_data: mujoco.MjData,
        side: str,
        args: argparse.Namespace,
        *,
        hand_config: str | None = None,
        hand_type: str = "linker_l20",
        home_qpos: np.ndarray | None = None,
    ):
        self.model = model
        self.data = data
        self.ik_data = ik_data
        self.side = side
        self.args = args

        prefix = side.upper()
        self.ee_body_name = f"{side}_j7"
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        if self.ee_body_id == -1:
            raise ValueError(f"Body '{self.ee_body_name}' not found.")

        self.arm_joint_ids = []
        self.arm_dof_indices = []
        for suffix in self._ARM_JOINT_SUFFIXES:
            jname = f"{prefix}_{suffix}"
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
                f"DexForce {side} home qpos size {self.arm_home_qpos.shape[0]} "
                f"!= arm DoF count {len(self.arm_dof_indices)}."
            )

        self.hand_joint_ids = []
        self.hand_dof_indices = []
        self.hand_joint_names = []
        for suffix in self._HAND_JOINT_SUFFIXES:
            jname = f"{prefix}_{suffix}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.hand_joint_ids.append(jid)
                self.hand_dof_indices.append(model.jnt_qposadr[jid])
                self.hand_joint_names.append(jname)
        self.hand_dof_indices = np.array(self.hand_dof_indices, dtype=int)

        # Build actuator indices for arm and hand (for ctrl-based control)
        self.arm_actuator_ids = []
        for suffix in self._ARM_JOINT_SUFFIXES:
            aname = f"{prefix}_J{suffix[1:]}"  # e.g. LEFT_J1
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid != -1:
                self.arm_actuator_ids.append(aid)
        self.hand_actuator_ids = []
        for suffix in self._HAND_JOINT_SUFFIXES:
            aname = f"{prefix}_{suffix}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid != -1:
                self.hand_actuator_ids.append(aid)

        self.hand_retargeter = None
        if hand_config:
            self.hand_retargeter = HandRetargeter(hand_config, side)
        self.hand_qpos_mapping = self._HAND_QPOS_MAPPING.get(hand_type)

        initial_body_pos = data.xpos[self.ee_body_id].copy()
        initial_body_quat = matrix_to_quaternion(
            data.xmat[self.ee_body_id].reshape(3, 3).copy()
        )
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        base_xmat = None
        if base_body_id != -1:
            base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

        self.tracker = WristTracker(
            initial_body_pos,
            initial_body_quat,
            position_scale=args.position_scale,
            ema_alpha=args.ema_alpha,
            negate_rot_xy=False,
            base_xmat=base_xmat,
        )
        self.latest_hand_qpos = None

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

    def update_from_pose(self, robot_position, robot_quaternion) -> None:
        self.tracker.update(robot_position, robot_quaternion)

    def update_hand_from_mediapipe(self, mediapipe_pts: np.ndarray | None) -> None:
        if mediapipe_pts is None or self.hand_retargeter is None or not self.hand_retargeter.available:
            return
        result = self.hand_retargeter.retarget_mediapipe(mediapipe_pts)
        if result is not None:
            self.latest_hand_qpos = result

    def step_ik(self) -> np.ndarray:
        return solve_body_pose_ik(
            self.model,
            self.ik_data,
            self.ee_body_id,
            self.target_position,
            self.target_quaternion,
            self.data.qpos[: self.model.nq],
            rot_weight=self.args.rot_weight,
            damping=self.args.ik_damping,
            current_q_weight=self.args.ik_current_weight,
            dof_indices=self.arm_dof_indices,
            home_qpos=self.arm_home_qpos,
        )

    def apply_qpos(self, q_sol: np.ndarray):
        for joint_id in self.arm_joint_ids:
            qadr = self.model.jnt_qposadr[joint_id]
            if qadr < q_sol.shape[0]:
                self.data.qpos[qadr] = q_sol[qadr]

        if self.latest_hand_qpos is not None:
            mapped = self.latest_hand_qpos[self.hand_qpos_mapping] if self.hand_qpos_mapping is not None else self.latest_hand_qpos
            if len(self.hand_dof_indices) == len(mapped):
                for qadr, value in zip(self.hand_dof_indices, mapped):
                    self.data.qpos[qadr] = value


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


def _apply_initial_pose_dexforce(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
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

    # Load retargeter for wuji or inspire hand
    hand_retargeter = None
    if hand_type == "wuji":
        hand_retargeter = HandRetargeter(_resolve_hand_config(args, hand_type), args.hand_side)
    elif hand_type == "inspire":
        hand_retargeter = HandRetargeter(_resolve_hand_config(args, hand_type), args.hand_side)

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
    pico4_input = None
    if args.input_source == "avp":
        from util.avp_input import AVPInput
        avp_input = AVPInput(ip=args.avp_ip)
        print(f"  Input: Apple Vision Pro ({args.avp_ip})")
    elif args.input_source == "pico4":
        from util.pico4_input import Pico4
        pico4_input = Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
        print(f"  Input: Pico 4 ({args.pico4_mode})")
    else:
        sock = make_socket(args.port)
        print(f"  Input: Quest 3 (UDP port {args.port})")

    # Wrist tracker
    tracker = WristTracker(
        initial_site_pos,
        initial_site_quat,
        position_scale=args.position_scale,
        ema_alpha=args.ema_alpha,
        negate_rot_xy=config.get("negate_rot_xy", False) if args.input_source == "quest3" else False,
        base_xmat=base_xmat,
    )

    # State variables
    last_log_time = time.time()
    latest_gripper_cmd = None
    latest_hand_qpos = None
    q_sol = data.qpos[:model.nq].copy()  # 初始化为当前关节角，避免未定义

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
                    # Hand retargeting (wuji / inspire)
                    if hand_type in ("wuji", "inspire") and hand_retargeter is not None and hand_retargeter.available:
                        mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                        if mediapipe_pts is not None:
                            result = hand_retargeter.retarget_mediapipe(mediapipe_pts)
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
                                inverted=config.get("pinch_inverted", False),
                            )

                    # Arm: wrist pose
                    wrist = avp_input.get_wrist_pose("right")
                    if wrist is not None:
                        robot_position, robot_quaternion = wrist
                        tracker.update(robot_position, robot_quaternion)
            elif pico4_input is not None:
                # --- Pico 4 path ---
                # Hand retargeting (wuji / inspire)
                if hand_type in ("wuji", "inspire") and hand_retargeter is not None and hand_retargeter.available:
                    mediapipe_pts = pico4_input.get_landmarks_mediapipe("right")
                    if mediapipe_pts is not None:
                        result = hand_retargeter.retarget_mediapipe(mediapipe_pts)
                        if result is not None:
                            latest_hand_qpos = result

                # Gripper from pinch distance
                if hand_type == "gripper" and gripper_actuator_id != -1:
                    pinch_distance = pico4_input.get_pinch_distance("right")
                    if pinch_distance is not None:
                        latest_gripper_cmd = _pinch_to_gripper(
                            pinch_distance,
                            config["pinch_max_distance"],
                            config["gripper_max"],
                            inverted=config.get("pinch_inverted", False),
                        )

                # Arm: wrist pose
                wrist = pico4_input.get_wrist_pose("right")
                if wrist is not None:
                    robot_position, robot_quaternion = wrist
                    tracker.update(robot_position, robot_quaternion)
            else:
                # --- Quest 3 path (unchanged) ---
                packet = recv_latest_packet(sock)

                if packet is not None:
                    message = packet.decode("utf-8", errors="ignore")

                    # Hand retargeting from landmarks (wuji / inspire)
                    if hand_type in ("wuji", "inspire") and hand_retargeter is not None:
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
                                    inverted=config.get("pinch_inverted", False),
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
            if tracker.initialized:
                q_sol_new = solve_pose_ik(
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
                # 丢弃跳变解：任意关节变化超过 30° 则保持上一帧
                if np.max(np.abs(np.rad2deg(q_sol_new[:6] - q_sol[:6]))) < 30.0:
                    q_sol = q_sol_new

            # --- Apply arm controls ---
            if tracker.initialized:
                if hand_type in ("wuji", "inspire"):
                    # Write arm ctrl (indices 0..num_arm-1)
                    for i in range(num_arm_actuators):
                        joint_id = model.actuator_trnid[i, 0]
                        qadr = model.jnt_qposadr[joint_id]
                        data.ctrl[i] = q_sol[qadr]
                    # Write hand ctrl
                    if latest_hand_qpos is not None:
                        hand_qpos_mapping = config.get("hand_qpos_mapping")
                        mapped = latest_hand_qpos[hand_qpos_mapping] if hand_qpos_mapping else latest_hand_qpos
                        data.ctrl[num_arm_actuators : num_arm_actuators + num_hand_actuators] = (
                            mapped[:num_hand_actuators]
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
    pico4_input = None
    if args.input_source == "avp":
        from util.avp_input import AVPInput
        avp_input = AVPInput(ip=args.avp_ip)
        print(f"  Input: Apple Vision Pro ({args.avp_ip})")
    elif args.input_source == "pico4":
        from util.pico4_input import Pico4
        pico4_input = Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
        print(f"  Input: Pico 4 ({args.pico4_mode})")
    else:
        sock = make_socket(args.port)
        print(f"  Input: Quest 3 (UDP port {args.port})")

    last_log_time = time.time()

    with viewer.launch_passive(model, data) as vis:
        while vis.is_running():
            if avp_input is not None:
                # --- Apple Vision Pro path ---
                if avp_input.poll():
                    for arm, side in ((left_arm, "left"), (right_arm, "right")):
                        arm.update_gripper_from_pinch(avp_input.get_pinch_distance(side))
                        wrist = avp_input.get_wrist_pose(side)
                        if wrist is not None:
                            arm.update_from_pose(*wrist)
            elif pico4_input is not None:
                # --- Pico 4 path ---
                for arm, side in ((left_arm, "left"), (right_arm, "right")):
                    arm.update_gripper_from_pinch(pico4_input.get_pinch_distance(side))
                    wrist = pico4_input.get_wrist_pose(side)
                    if wrist is not None:
                        arm.update_from_pose(*wrist)
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


def _run_dexforce_bimanual(config: dict, args: argparse.Namespace) -> None:
    if args.scene is None:
        xml_path = Path(config["scene_xml"]).expanduser().resolve()
    else:
        xml_path = Path(args.scene).expanduser().resolve()
    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose_dexforce(model, data)

    dexforce_hand_type = config.get("dexforce_hand_type", "linker_l20")
    left_hand_config = _resolve_hand_config(args, dexforce_hand_type, side="left")
    right_hand_config = _resolve_hand_config(args, dexforce_hand_type, side="right")

    left_arm = DexForceArmController(
        model,
        data,
        ik_data,
        "left",
        args,
        hand_config=left_hand_config,
        hand_type=dexforce_hand_type,
        home_qpos=_DEXFORCE_LEFT_ARM_HOME_QPOS,
    )
    right_arm = DexForceArmController(
        model,
        data,
        ik_data,
        "right",
        args,
        hand_config=right_hand_config,
        hand_type=dexforce_hand_type,
        home_qpos=_DEXFORCE_RIGHT_ARM_HOME_QPOS,
    )

    sock = None
    avp_input = None
    pico4_input = None
    if args.input_source == "avp":
        from util.avp_input import AVPInput

        avp_input = AVPInput(ip=args.avp_ip)
        print(f"  Input: Apple Vision Pro ({args.avp_ip})")
    elif args.input_source == "pico4":
        from util.pico4_input import Pico4

        pico4_input = Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
        print(f"  Input: Pico 4 ({args.pico4_mode})")
    else:
        sock = make_socket(args.port)
        print(f"  Input: Quest 3 (UDP port {args.port})")

    # 躯干/颈部关节索引
    _torso_joints = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2")
    }
    _head_zero_q = [None]  # 用列表包装，loop 内可赋值
    _head_zero_y = [None]  # HMD 初始高度（Pico Y 轴）

    last_log_time = time.time()

    with viewer.launch_passive(model, data) as vis:
        vis.cam.azimuth = model.vis.global_.azimuth
        vis.cam.elevation = model.vis.global_.elevation
        vis.cam.distance = model.stat.extent * 1.5
        vis.cam.lookat[:] = model.stat.center

        while vis.is_running():
            loop_start = time.time()

            if avp_input is not None:
                if avp_input.poll():
                    for arm, side in ((left_arm, "left"), (right_arm, "right")):
                        arm.update_hand_from_mediapipe(avp_input.get_landmarks_mediapipe(side))
                        wrist = avp_input.get_wrist_pose(side)
                        if wrist is not None:
                            arm.update_from_pose(*wrist)
            elif pico4_input is not None:
                for arm, side in ((left_arm, "left"), (right_arm, "right")):
                    arm.update_hand_from_mediapipe(pico4_input.get_landmarks_mediapipe(side))
                    wrist = pico4_input.get_wrist_pose(side)
                    if wrist is not None:
                        arm.update_from_pose(*wrist)
                # Head pose → 躯干/颈部
                head_raw = pico4_input.get_head_pose()
                if head_raw is not None:
                    qx, qy, qz, qw = head_raw[3], head_raw[4], head_raw[5], head_raw[6]
                    q_cur = np.array([qw, qx, qy, qz])  # [w,x,y,z]
                    head_y = float(head_raw[1])  # Pico Y 轴 = 高度（向上为正）
                    if _head_zero_q[0] is None:
                        _head_zero_q[0] = q_cur.copy()
                        _head_zero_y[0] = head_y
                    # q_cal = q_cur * q_zero^(-1)，得到相对初始姿态的旋转
                    q0 = _head_zero_q[0]
                    q_zero_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
                    def _qmul(a, b):
                        w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
                        x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
                        y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
                        z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
                        n = np.sqrt(w*w+x*x+y*y+z*z)
                        return np.array([w,x,y,z]) / n if n > 1e-8 else np.array([1.,0,0,0])
                    q = _qmul(q_cur, q_zero_inv)
                    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                    # Pico OpenXR: X右 Y上 Z后
                    # yaw:  绕 Y 轴，左转为正
                    yaw   =  np.arctan2(2*(qy*qw + qx*qz), 1 - 2*(qx*qx + qy*qy))
                    # pitch: 绕 X 轴，低头为负（Pico），取反使低头为正（和机器人一致）
                    pitch =  np.arctan2(2*(qx*qw - qy*qz), 1 - 2*(qx*qx + qz*qz))
                    # 高度：HMD Y 轴相对初始值的变化（站起为正，蹲下为负）
                    dy = float(np.clip((_head_zero_y[0] - head_y), -0.3, 0.3))
                    t = dy / 0.3  # -1~1，蹲下为正
                    def _clamp_joint(jname, val):
                        jid = _torso_joints[jname]
                        lo, hi = model.jnt_range[jid]
                        data.qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))
                    # 高度控制: ANKLE + KNEE 联动（参考官方实现）
                    _clamp_joint("ANKLE",  0.6 + t * 0.6)
                    _clamp_joint("KNEE",  -1.2 + t * (-0.8))
                    # 躯干俯仰: pitch → BUTTOCK（参考官方 x_rot → joint3）
                    _clamp_joint("BUTTOCK", 0.6 - pitch * 0.5)
                    # 腰部偏航
                    _clamp_joint("WAIST", yaw)
                    # 颈部
                    _clamp_joint("NECK1", yaw)
                    _clamp_joint("NECK2", pitch)  # 低头为正
            else:
                try:
                    sock_data, _ = sock.recvfrom(4096)
                    message = sock_data.decode("utf-8", errors="ignore")
                    for arm, side in ((left_arm, "left"), (right_arm, "right")):
                        if arm.hand_retargeter is not None:
                            landmarks = (
                                parse_left_landmarks(message)
                                if side == "left"
                                else parse_right_landmarks(message)
                            )
                            if landmarks is not None:
                                result = arm.hand_retargeter.retarget(landmarks)
                                if result is not None:
                                    arm.latest_hand_qpos = result
                        wrist_pose = (
                            parse_left_wrist_pose(message)
                            if side == "left"
                            else parse_right_wrist_pose(message)
                        )
                        if wrist_pose is not None:
                            wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
                            wrist_quaternion = (
                                wrist_pose[3],
                                wrist_pose[4],
                                wrist_pose[5],
                                wrist_pose[6],
                            )
                            robot_position, robot_quaternion = transform_vr_to_robot_pose(
                                wrist_position, wrist_quaternion
                            )
                            arm.update_from_pose(robot_position, robot_quaternion)
                except BlockingIOError:
                    pass

            q_left = left_arm.step_ik()
            q_right = right_arm.step_ik()
            left_arm.apply_qpos(q_left)
            right_arm.apply_qpos(q_right)
            mujoco.mj_forward(model, data)
            vis.sync()

            now = time.time()
            if now - last_log_time > 1.0:
                print(
                    f"L resid: {left_arm.latest_residual} "
                    f"R resid: {right_arm.latest_residual}"
                )
                last_log_time = now

            sleep_time = model.opt.timestep - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


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
        choices=["quest3", "avp", "pico4"],
        help="Input device: quest3 (UDP, default), avp (Vision Pro), or pico4.",
    )
    parser.add_argument(
        "--avp-ip",
        default="192.168.1.100",
        help="Apple Vision Pro IP address (used with --input-source avp).",
    )
    parser.add_argument(
        "--pico4-mode",
        default="relay",
        choices=["relay", "direct"],
        help="Pico 4 input mode: relay daemon (default) or direct TCP server.",
    )
    parser.add_argument(
        "--pico4-port",
        type=int,
        default=63901,
        help="Pico 4 direct-mode TCP listen port.",
    )
    parser.add_argument(
        "--pico4-relay-host",
        default="127.0.0.1",
        help="Pico 4 relay daemon host.",
    )
    parser.add_argument(
        "--pico4-relay-port",
        type=int,
        default=63902,
        help="Pico 4 relay daemon port.",
    )
    parser.add_argument(
        "--pico4-broadcast-port",
        type=int,
        default=29888,
        help="Pico 4 direct-mode UDP broadcast port.",
    )
    args = parser.parse_args()

    config = ROBOT_CONFIGS[args.robot]

    # Apply defaults from config if user didn't override
    if args.scene is None:
        args.scene = config["scene_xml"]
    if args.position_scale is None:
        args.position_scale = config["position_scale"]

    if config.get("dexforce", False):
        _run_dexforce_bimanual(config, args)
    elif config.get("bimanual", False):
        _run_bimanual(config, args)
    else:
        _run_single_arm(config, args)


if __name__ == "__main__":
    main()

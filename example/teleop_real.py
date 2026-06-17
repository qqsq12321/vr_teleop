"""Teleoperate a real robot arm via VR hand tracking.

Supports multiple end-effector configurations via --robot:
  kinova_gripper  Kinova Gen3 + Robotiq 2F-85 gripper
  kinova_wuji     Kinova Gen3 + Wuji dexterous hand (20 DOF)
  rm65            Realman RM65 6-DOF arm (no end-effector)
  rm65_gripper    Realman RM65 + EG2-4C2 two-finger gripper
  rm65_inspire    Realman RM65 + Inspire dexterous hand

Input sources (--input-source):
  quest3          Meta Quest 3 via UDP (default)
  avp             Apple Vision Pro via avp_stream / Tracking Streamer

The --disable-arm flag enables hand-only mode for kinova_wuji and rm65_inspire.

Examples:
    python example/teleop_real.py --robot kinova_gripper --kinova-ip 192.168.1.10
    python example/teleop_real.py --robot kinova_gripper --input-source avp --avp-ip 192.168.5.32
    python example/teleop_real.py --robot kinova_wuji --kinova-ip 192.168.1.10
    python example/teleop_real.py --robot kinova_wuji --disable-arm --input-source avp --avp-ip 192.168.5.32
    python example/teleop_real.py --robot rm65 --rm65-ip 192.168.1.18
    python example/teleop_real.py --robot rm65_gripper --rm65-ip 192.168.1.18 --input-source avp --avp-ip 192.168.5.32
    python example/teleop_real.py --robot rm65_inspire --rm65-ip 192.168.1.18 --inspire-port /dev/ttyUSB0
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"))

import argparse
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from util.ik import solve_pose_ik
from util.quaternion import (
    matrix_to_quaternion,
    transform_quest3_raw_to_robot_pose,
)
from util.udp_socket import (
    clamp_pinch_ratio,
    make_socket,
    recv_latest_packet,
    parse_right_landmarks,
    parse_right_wrist_pose,
    pinch_distance_from_landmarks,
)
from util.wrist_tracker import WristTracker
from util.hand_retarget import (
    HandRetargeter,
    default_inspire_config_path,
    default_pico4_config_path,
)

# ---------------------------------------------------------------------------
# RM_API2 SDK path
# ---------------------------------------------------------------------------

_RM_API2_DIR = _Path(__file__).resolve().parents[1] / "third_party" / "RM_API2" / "Python"
if str(_RM_API2_DIR) not in sys.path:
    sys.path.insert(0, str(_RM_API2_DIR))

# ---------------------------------------------------------------------------
# Kortex SDK import
# ---------------------------------------------------------------------------

_KORTEX_EXAMPLES_DIR = (
    _Path(__file__).resolve().parents[1]
    / "third_party"
    / "Kinova-kortex2_Gen3_G3L"
    / "api_python"
    / "examples"
)
if str(_KORTEX_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_KORTEX_EXAMPLES_DIR))

# Kortex SDK is imported lazily inside _run_arm_teleop() so that RM65 paths
# never trigger Kortex socket initialization (which breaks RM SDK connections).

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_ARM_JOINTS = 7
_NUM_HAND_JOINTS = 20

# Gen3 initial pose (7 arm joints) used as IK prior and startup target.
_INIT_QPOS = np.array(
    [1.57079633, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633],
    dtype=np.float64,
)

# Pinch distance normalization (meters)
_PINCH_MAX_DISTANCE = 0.06

# Fixed Kinova teleop gains/limits.
_POSITION_SCALE = 1.0
_WRIST_POS_DEADBAND = 0.03
_ARM_KP = 2.0
_ARM_MAX_SPEED_DEG = 50.0
_EMA_ALPHA = 0.8
_WRIST_ROT_DEADBAND_DEG = 8.0
_ROT_WEIGHT = 1.0
_IK_DAMPING = 1e-3
_IK_CURRENT_WEIGHT = 0.1
_ARM_DEADBAND_DEG = 0.5
_CONTROL_PERIOD_S = 0.02
_PACKET_TIMEOUT_S = 0.25
_HOME_TIMEOUT_S = 30.0
_DEFAULT_SITE = "kinova_ee_site"
_HAND_CUTOFF_FREQ = 5.0

# ---------------------------------------------------------------------------
# Scene / config paths
# ---------------------------------------------------------------------------

_SCENE_DIR = Path(__file__).resolve().parent / "scene"


ROBOT_CONFIGS = {
    "kinova_gripper": {
        "scene_xml": str(_SCENE_DIR / "scene_kinova_gen3.xml"),
        "hand_type": "gripper",
    },
    "kinova_wuji": {
        "scene_xml": str(_SCENE_DIR / "scene_kinova_gen3_wuji.xml"),
        "scene_xml_arm_only": str(_SCENE_DIR / "scene_kinova_gen3.xml"),
        "hand_type": "wuji",
    },
    "rm65": {
        "scene_xml": str(_SCENE_DIR / "scene_rm65.xml"),
        "hand_type": "none",
    },
    "rm65_gripper": {
        "scene_xml": str(_SCENE_DIR / "scene_rm65.xml"),
        "hand_type": "eg2",
    },
    "rm65_inspire": {
        "scene_xml": str(_SCENE_DIR / "scene_rm65_inspire.xml"),
        "hand_type": "inspire",
    },
}


def _make_hand_controller(args: argparse.Namespace):
    """Create wuji hand hardware controller. Returns (hand, controller) or (None, None)."""
    if args.disable_hand:
        return None, None
    try:
        import wujihandpy
    except ImportError:
        raise ImportError(
            "wujihandpy is not installed, but Wuji hand control is enabled. "
            "Install it or pass --disable-hand."
        )
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    controller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=_HAND_CUTOFF_FREQ),
    )
    time.sleep(0.5)
    return hand, controller


# ---------------------------------------------------------------------------
# Gripper helper
# ---------------------------------------------------------------------------


def _pinch_to_gripper_position(pinch_distance: float) -> float:
    """Map pinch distance to Kortex gripper position (binary: 0=open, 1=closed)."""
    ratio = clamp_pinch_ratio(pinch_distance, _PINCH_MAX_DISTANCE)
    return 1.0 if ratio < 0.5 else 0.0


def _resolve_hand_config(args: argparse.Namespace, hand_type: str) -> str | None:
    if args.hand_config:
        return args.hand_config
    if args.input_source == "pico4" and hand_type in ("wuji", "inspire"):
        return str(default_pico4_config_path(hand_type))
    if hand_type == "inspire":
        return str(default_inspire_config_path())
    return None


# ---------------------------------------------------------------------------
# Shared Kortex utilities
# ---------------------------------------------------------------------------


def _angle_error_deg(target_deg: np.ndarray, current_deg: np.ndarray) -> np.ndarray:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _get_measured_q_rad(base_cyclic) -> np.ndarray:
    feedback = base_cyclic.RefreshFeedback()
    q_deg = np.array(
        [feedback.actuators[i].position for i in range(_NUM_ARM_JOINTS)],
        dtype=np.float64,
    )
    q_deg = np.where(q_deg > 180.0, q_deg - 360.0, q_deg)
    return np.deg2rad(q_deg)


def _build_joint_speeds_command(speed_deg_s: np.ndarray):
    from kortex_api.autogen.messages import Base_pb2
    joint_speeds = Base_pb2.JointSpeeds()
    for i, speed in enumerate(speed_deg_s.tolist()):
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i
        joint_speed.value = float(speed)
        joint_speed.duration = 0
    return joint_speeds


def _stop_arm(base) -> None:
    try:
        base.Stop()
    except Exception as exc:
        print(f"Warning: failed to stop Kinova arm cleanly: {exc}")


def _send_gripper_command(base, position: float) -> None:
    from kortex_api.autogen.messages import Base_pb2
    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_SPEED
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 0
    if position >= 1.0:
        finger.value = -1.0
    else:
        finger.value = 1.0
    base.SendGripperCommand(gripper_command)


def _check_for_end_or_abort(event: threading.Event):
    from kortex_api.autogen.messages import Base_pb2
    def check(notification, event=event):
        if (
            notification.action_event == Base_pb2.ACTION_END
            or notification.action_event == Base_pb2.ACTION_ABORT
        ):
            event.set()
    return check


def _move_arm_home(base, timeout: float = 30.0) -> bool:
    from kortex_api.autogen.messages import Base_pb2
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)

    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle
            break

    if action_handle is None:
        print("Warning: Kinova Home action not found; skipping move-to-home.")
        return False

    finished = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        _check_for_end_or_abort(finished),
        Base_pb2.NotificationOptions(),
    )
    try:
        print("Moving Kinova arm to Home...")
        base.ExecuteActionFromReference(action_handle)
        ok = finished.wait(timeout)
        if ok:
            print("Kinova Home reached.")
        else:
            print("Warning: timeout while waiting for Kinova Home action.")
        return ok
    finally:
        base.Unsubscribe(notification_handle)


def _move_to_init_qpos(
    base,
    base_cyclic,
    timeout: float = 15.0,
    kp: float = 2.0,
    max_speed: float = 30.0,
    threshold_deg: float = 1.0,
    dt: float = 0.02,
) -> None:
    target_deg = np.rad2deg(_INIT_QPOS)
    print(f"Moving to _INIT_QPOS (deg): {target_deg.tolist()}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        current_rad = _get_measured_q_rad(base_cyclic)
        current_deg = np.rad2deg(current_rad)
        err_deg = _angle_error_deg(target_deg, current_deg)
        if np.all(np.abs(err_deg) < threshold_deg):
            print("Reached _INIT_QPOS.")
            base.SendJointSpeedsCommand(
                _build_joint_speeds_command(np.zeros(_NUM_ARM_JOINTS, dtype=np.float64))
            )
            return
        speed_deg_s = np.clip(kp * err_deg, -max_speed, max_speed)
        speed_deg_s[np.abs(err_deg) < _ARM_DEADBAND_DEG] = 0.0
        base.SendJointSpeedsCommand(_build_joint_speeds_command(speed_deg_s))
        time.sleep(dt)
    print("Warning: timeout moving to _INIT_QPOS.")
    base.SendJointSpeedsCommand(
        _build_joint_speeds_command(np.zeros(_NUM_ARM_JOINTS, dtype=np.float64))
    )


# ---------------------------------------------------------------------------
# RM65 constants
# ---------------------------------------------------------------------------

_RM65_NUM_JOINTS = 6
_RM65_INIT_QPOS_DEG = [0.0, -29.9, -103.7, 0.57, 44.8, 30.0]
_RM65_INIT_QPOS_RAD = np.deg2rad(_RM65_INIT_QPOS_DEG)
_RM65_POSITION_SCALE = 1.5
_RM65_WRIST_POS_DEADBAND = 0.03
_RM65_WRIST_ROT_DEADBAND_DEG = 8.0
_RM65_EMA_ALPHA = 0.5
_RM65_ROT_WEIGHT = 1.0
_RM65_IK_DAMPING = 1e-3
_RM65_IK_CURRENT_WEIGHT = 0.3
_RM65_CONTROL_PERIOD_S = 0.02
_RM65_PACKET_TIMEOUT_S = 0.25
_RM65_DEFAULT_SITE = "rm65_ee_site"
_RM65_PINCH_MAX_DISTANCE = 0.06
_RM65_POS_GAIN = 0.08
_RM65_ROT_GAIN = 0.08

# Inspire RH56DFX: retarget 12维输出（实际 dof 顺序）
#   0 index_proximal, 1 index_intermediate
#   2 middle_proximal, 3 middle_intermediate
#   4 pinky_proximal, 5 pinky_intermediate
#   6 ring_proximal, 7 ring_intermediate
#   8 thumb_proximal_yaw, 9 thumb_proximal_pitch
#   10 thumb_intermediate, 11 thumb_distal
# 真机/串口 6 通道顺序: 小拇指, 无名指, 中指, 食指, 大拇指弯曲, 大拇指旋转
# 这里只取 6 个独立关节；其余 intermediate/distal 是 mimic/耦合关节。
_INSPIRE_CHANNEL_INDICES = [4, 6, 2, 0, 9, 8]
_INSPIRE_CHANNEL_MAX_RAD = [1.47, 1.47, 1.47, 1.47, 0.6, 1.308]
# 当前实机反馈表现为“手弯曲时灵巧手打开”，因此默认对 6 个通道全部反向。
_INSPIRE_CHANNEL_INVERT = [True, True, True, True, True, True]
_INSPIRE_SERIAL_RESPONSE_TIMEOUT_S = 0.1

# ---------------------------------------------------------------------------
# RM65 SDK helpers
# ---------------------------------------------------------------------------


def _rm65_connect(ip: str):
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, 8080)
    if handle.id == -1:
        raise RuntimeError(f"Failed to connect to RM65 at {ip}:8080")
    print(f"Connected to RM65 at {ip}, handle id={handle.id}, dof={arm.arm_dof}")
    return arm


def _rm65_get_joint_deg(arm) -> np.ndarray:
    ret, q_deg = arm.rm_get_joint_degree()
    if ret != 0:
        raise RuntimeError(f"rm_get_joint_degree failed: {ret}")
    return np.array(q_deg[:_RM65_NUM_JOINTS], dtype=np.float64)


def _rm65_move_to_init(arm) -> None:
    print(f"Moving RM65 to init pose (deg): {_RM65_INIT_QPOS_DEG}")
    ret = arm.rm_movej(_RM65_INIT_QPOS_DEG, v=20, r=0, connect=0, block=1)
    if ret != 0:
        print(f"Warning: rm_movej to init returned {ret}")
    else:
        print("RM65 reached init pose.")


def _rm65_send_canfd(arm, q_rad: np.ndarray) -> None:
    q_deg = np.rad2deg(q_rad[:_RM65_NUM_JOINTS]).tolist()
    arm.rm_movej_canfd(q_deg, follow=True)


def _rm65_pinch_to_gripper(pinch_distance: float) -> int:
    """Map pinch distance to EG2-4C2 position (1=closed, 1000=open)."""
    ratio = clamp_pinch_ratio(pinch_distance, _RM65_PINCH_MAX_DISTANCE)
    return max(1, int(ratio * 1000))


def _slerp_step(current_quat: np.ndarray, target_quat: np.ndarray, gain: float) -> np.ndarray:
    """单步 SLERP，每帧按 gain 比例逼近目标姿态。"""
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
    return q / np.linalg.norm(q)


def _inspire_retarget_to_real(retarget_output: np.ndarray) -> list[int]:
    """将 retarget 12维弧度输出转换为因时 RH56DFX 6 通道控制量（0~2000）。"""
    result = []
    if not np.all(np.isfinite(retarget_output)):
        raise ValueError("Invalid Inspire retarget output: contains NaN/Inf")
    for i, max_rad, invert in zip(
        _INSPIRE_CHANNEL_INDICES,
        _INSPIRE_CHANNEL_MAX_RAD,
        _INSPIRE_CHANNEL_INVERT,
    ):
        val = float(retarget_output[i])
        norm = float(np.clip(val / max_rad, 0.0, 1.0))
        if invert:
            norm = 1.0 - norm
        angle_int = int(norm * 2000)
        result.append(angle_int)
    return result


class InspireSerialController:
    """Direct serial controller for Inspire RH56DFX hand."""

    def __init__(self, port_name: str, baudrate: int, hand_id: int):
        import serial

        self._port = serial.Serial(port_name, baudrate, timeout=0.01)
        self._hand_id = int(hand_id)
        self._port_name = port_name
        self._baudrate = int(baudrate)
        print(f"Connected to Inspire hand serial at {port_name} @ {baudrate} baud.")

    def _read_response(self) -> bytes:
        deadline = time.time() + _INSPIRE_SERIAL_RESPONSE_TIMEOUT_S
        input_bytes = bytearray()
        while time.time() < deadline:
            chunk = self._port.read(self._port.in_waiting or 1)
            if chunk:
                input_bytes += chunk
            else:
                break
        return bytes(input_bytes)

    def _encode_angles(self, angles: list[int]) -> list[int]:
        if len(angles) != 6:
            raise ValueError(f"Inspire hand expects 6 channels, got {len(angles)}")
        # Keep retarget output semantics unchanged (0~2000), only convert at the transport boundary.
        return [int(np.clip(round(angle / 2.0), 0, 1000)) for angle in angles]

    def send(self, angles: list[int]) -> None:
        encoded_angles = self._encode_angles(angles)
        output = bytearray([0xEB, 0x90, self._hand_id, 0x0F, 0x12, 0xCE, 0x05])
        for angle in encoded_angles:
            output.append(angle & 0xFF)
            output.append((angle >> 8) & 0xFF)
        check_num = sum(output[2:2 + 0x0F + 3])
        output.append(check_num & 0xFF)
        self._port.write(output)
        self._read_response()

    def close(self) -> None:
        if self._port.is_open:
            self._port.close()
            print(f"Closed Inspire hand serial at {self._port_name} @ {self._baudrate} baud.")


# ---------------------------------------------------------------------------
# RM65 teleop main loop
# ---------------------------------------------------------------------------


def _run_rm65_teleop(config: dict, args: argparse.Namespace) -> None:
    hand_type = config["hand_type"]
    has_gripper = hand_type == "eg2"
    has_inspire = hand_type == "inspire"
    hand_only = bool(args.disable_arm and has_inspire)

    # RM65 专属参数，覆盖 CLI 默认值（CLI 默认值是 Kinova 调的）
    ema_alpha = _RM65_EMA_ALPHA
    position_scale = _RM65_POSITION_SCALE
    rot_weight = _RM65_ROT_WEIGHT
    ik_damping = _RM65_IK_DAMPING
    ik_current_weight = _RM65_IK_CURRENT_WEIGHT

    # Inspire 手 retargeter
    inspire_retargeter = None
    inspire_controller = None
    if has_inspire:
        inspire_retargeter = HandRetargeter(_resolve_hand_config(args, "inspire"), "right")
        inspire_controller = InspireSerialController(
            args.inspire_port,
            args.inspire_baudrate,
            args.inspire_hand_id,
        )

    model = None
    state_data = None
    ik_data = None
    site_id = -1
    base_body_id = -1
    arm = None
    tracker = None

    if not hand_only:
        # Load MuJoCo model (IK only, no viewer)
        xml_path = Path(config["scene_xml"]).resolve()
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        state_data = mujoco.MjData(model)
        ik_data = mujoco.MjData(model)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _RM65_DEFAULT_SITE)
        if site_id == -1:
            raise ValueError(f"Site '{_RM65_DEFAULT_SITE}' not found in {xml_path}")
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Connect RM65 first (before AVP init which may block)
        arm = _rm65_connect(args.rm65_ip)

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

    try:
        if not hand_only:
            _rm65_move_to_init(arm)

            # Read current joint angles and sync MuJoCo state
            current_q_deg = _rm65_get_joint_deg(arm)
            current_q_rad = np.deg2rad(current_q_deg)

            state_q = np.array(model.qpos0, dtype=np.float64)
            state_q[:_RM65_NUM_JOINTS] = current_q_rad
            state_data.qpos[:model.nq] = state_q
            state_data.qvel[:] = 0.0
            mujoco.mj_forward(model, state_data)

            initial_site_pos = state_data.site_xpos[site_id].copy()
            initial_site_quat = matrix_to_quaternion(
                state_data.site_xmat[site_id].reshape(3, 3).copy()
            )
            base_xmat = None
            if base_body_id != -1:
                base_xmat = state_data.xmat[base_body_id].reshape(3, 3).copy()

            tracker = WristTracker(
                initial_site_pos,
                initial_site_quat,
                position_scale=position_scale,
                ema_alpha=ema_alpha,
                negate_rot_xy=False,
                base_xmat=base_xmat,
                position_deadband=_RM65_WRIST_POS_DEADBAND,
                rotation_deadband_deg=_RM65_WRIST_ROT_DEADBAND_DEG,
            )

            q_sol = current_q_rad.copy()
            # 笛卡尔平滑层：维护当前命令位姿，每帧小步逼近 tracker 目标
            cmd_pos = initial_site_pos.copy()
            cmd_quat = np.array(initial_site_quat, dtype=np.float64)

            print(f"  Initial arm q (deg): {current_q_deg.tolist()}")
            print(f"  Init pose   (deg):   {_RM65_INIT_QPOS_DEG}")
            print(f"  Initial EE pos:      {initial_site_pos.tolist()}")
            print(f"  Packet timeout: {_RM65_PACKET_TIMEOUT_S:.3f} s")
        else:
            current_q_deg = None
            q_sol = None
            cmd_pos = None
            cmd_quat = None

        latest_gripper_pos = None
        latest_inspire_angles = None
        last_log_time = time.time()
        last_valid_packet_time = 0.0
        last_inspire_warn_time = 0.0
        print(f"  Gripper: {'EG2-4C2' if has_gripper else 'none'}")
        if hand_only:
            print("  Arm: disabled (Inspire hand-only mode)")

        # 激活灵巧手：先发一次阻塞指令，之后才能用 follow_angle 高频控制
        if has_inspire:
            print("Initializing Inspire hand...")
            inspire_controller.send([1000, 1000, 1000, 1000, 1000, 1000])

        print("Starting RM65 teleoperation loop..." if not hand_only else "Starting Inspire hand-only loop...")
        print("Press Ctrl+C to stop.")

        while True:
            loop_start = time.time()
            now = loop_start
            saw_valid_data = False

            if avp_input is not None:
                if avp_input.poll():
                    if avp_input.check_stop_gesture():
                        print("\nStopping teleoperation (stop gesture)...")
                        break

                    if has_gripper:
                        pinch_distance = avp_input.get_pinch_distance("right")
                        if pinch_distance is not None:
                            latest_gripper_pos = _rm65_pinch_to_gripper(pinch_distance)
                            saw_valid_data = True

                    wrist = avp_input.get_wrist_pose("right")
                    if wrist is not None and tracker is not None:
                        robot_position, robot_quaternion = wrist
                        saw_valid_data = True
                        if not tracker.initialized:
                            print("Captured initial wrist reference pose.")
                        tracker.update(robot_position, robot_quaternion)

                    if has_inspire and inspire_retargeter is not None:
                        mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                        if mediapipe_pts is not None:
                            result = inspire_retargeter.retarget_mediapipe(mediapipe_pts)
                            if result is not None and np.all(np.isfinite(result)):
                                latest_inspire_angles = _inspire_retarget_to_real(result)
                            elif result is not None and now - last_inspire_warn_time > 1.0:
                                print("Warning: invalid Inspire retarget frame; keeping previous hand command.")
                                last_inspire_warn_time = now
            elif pico4_input is not None:
                if has_gripper:
                    pinch_distance = pico4_input.get_pinch_distance("right")
                    if pinch_distance is not None:
                        latest_gripper_pos = _rm65_pinch_to_gripper(pinch_distance)
                        saw_valid_data = True

                wrist = pico4_input.get_wrist_pose("right")
                if wrist is not None and tracker is not None:
                    robot_position, robot_quaternion = wrist
                    saw_valid_data = True
                    if not tracker.initialized:
                        print("Captured initial wrist reference pose.")
                    tracker.update(robot_position, robot_quaternion)

                if has_inspire and inspire_retargeter is not None:
                    mediapipe_pts = pico4_input.get_landmarks_mediapipe("right")
                    if mediapipe_pts is not None:
                        result = inspire_retargeter.retarget_mediapipe(mediapipe_pts)
                        if result is not None and np.all(np.isfinite(result)):
                            latest_inspire_angles = _inspire_retarget_to_real(result)
                            saw_valid_data = True
                        elif result is not None and now - last_inspire_warn_time > 1.0:
                            print("Warning: invalid Inspire retarget frame; keeping previous hand command.")
                            last_inspire_warn_time = now
            else:
                packet = recv_latest_packet(sock)
                if packet is not None:
                    message = packet.decode("utf-8", errors="ignore")

                    if has_gripper:
                        landmarks = parse_right_landmarks(message)
                        if landmarks is not None:
                            pinch_distance = pinch_distance_from_landmarks(landmarks)
                            if pinch_distance is not None:
                                latest_gripper_pos = _rm65_pinch_to_gripper(pinch_distance)
                                saw_valid_data = True

                    wrist_pose = parse_right_wrist_pose(message)
                    if wrist_pose is not None and tracker is not None:
                        robot_position, robot_quaternion = transform_quest3_raw_to_robot_pose(wrist_pose)
                        saw_valid_data = True
                        if not tracker.initialized:
                            print("Captured initial wrist reference pose.")
                        tracker.update(robot_position, robot_quaternion)

                    if has_inspire and inspire_retargeter is not None:
                        landmarks = parse_right_landmarks(message)
                        if landmarks is not None:
                            result = inspire_retargeter.retarget(landmarks)
                            if result is not None and np.all(np.isfinite(result)):
                                latest_inspire_angles = _inspire_retarget_to_real(result)
                            elif result is not None and now - last_inspire_warn_time > 1.0:
                                print("Warning: invalid Inspire retarget frame; keeping previous hand command.")
                                last_inspire_warn_time = now

            if saw_valid_data:
                last_valid_packet_time = loop_start

            # Logging
            now = time.time()
            if (
                tracker is not None
                and
                tracker.residual is not None
                and tracker.euler_residual is not None
                and now - last_log_time > 1.0
            ):
                gripper_str = f"  gripper: {latest_gripper_pos}" if has_gripper and latest_gripper_pos is not None else ""
                print(
                    f"Wrist residual (xyz): {tracker.residual.tolist()} "
                    f"euler: {list(tracker.euler_residual)}{gripper_str}"
                )
                last_log_time = now

            # Send arm command
            if tracker is None:
                pass
            elif (
                not tracker.initialized
                or now - last_valid_packet_time > _RM65_PACKET_TIMEOUT_S
            ):
                pass  # 超时：不发送新指令，保持当前位置
            elif not hand_only:
                # 笛卡尔平滑层：每帧小步逼近 tracker 目标，IK 输入更稳定
                cmd_pos += (tracker.target_position - cmd_pos) * _RM65_POS_GAIN
                cmd_quat = _slerp_step(cmd_quat, tracker.target_quaternion, _RM65_ROT_GAIN)

                q_init = np.array(model.qpos0, dtype=np.float64)
                q_init[:_RM65_NUM_JOINTS] = q_sol[:_RM65_NUM_JOINTS]
                q_sol = solve_pose_ik(
                    model,
                    ik_data,
                    site_id,
                    cmd_pos,
                    cmd_quat,
                    q_init,
                    rot_weight=rot_weight,
                    damping=ik_damping,
                    current_q_weight=ik_current_weight,
                    home_qpos=np.array(_RM65_INIT_QPOS_RAD, dtype=np.float64),
                    skip_tail_joints=0,
                )
                if now - last_log_time < 0.1:
                    print(
                        f"  cmd_pos: {cmd_pos.tolist()}\n"
                        f"  q_target (deg): {np.rad2deg(q_sol[:_RM65_NUM_JOINTS]).tolist()}"
                    )
                _rm65_send_canfd(arm, q_sol)

            # Send gripper command
            if has_gripper and latest_gripper_pos is not None and arm is not None:
                arm.rm_set_gripper_position(latest_gripper_pos, block=False, timeout=0)

            # Send inspire hand command
            if has_inspire and latest_inspire_angles is not None and inspire_controller is not None:
                inspire_controller.send(latest_inspire_angles)
                if now - last_log_time < 0.1:
                    print(f"  inspire angles: {latest_inspire_angles}")

            elapsed = time.time() - loop_start
            sleep_time = _RM65_CONTROL_PERIOD_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        if arm is not None:
            try:
                arm.rm_set_arm_stop()
            except Exception as exc:
                print(f"Warning: rm_set_arm_stop failed: {exc}")
            try:
                arm.rm_delete_robot_arm()
                from Robotic_Arm.rm_robot_interface import RoboticArm
                RoboticArm.rm_destroy()
                time.sleep(1.0)  # 等待控制器端 TCP 连接完全释放
            except Exception as exc:
                print(f"Warning: RM65 disconnect failed: {exc}")
        if sock is not None:
            sock.close()
        if pico4_input is not None:
            pico4_input.stop()
        if inspire_controller is not None:
            try:
                inspire_controller.close()
            except Exception as exc:
                print(f"Warning: Inspire serial close failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate supported real robot arms with Quest 3 or Apple Vision Pro hand tracking."
    )
    parser.add_argument(
        "--robot",
        required=True,
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot configuration to use.",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument("--kinova-ip", default="192.168.1.10", help="Kinova robot IP.")
    parser.add_argument("--kinova-username", default="admin", help="Kinova username.")
    parser.add_argument("--kinova-password", default="admin", help="Kinova password.")
    parser.add_argument("--rm65-ip", default="192.168.1.18", help="Realman RM65 robot IP.")
    parser.add_argument(
        "--position-scale", type=float, default=_POSITION_SCALE,
        help="Scale for wrist position residuals.",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=_EMA_ALPHA,
        help="EMA smoothing factor for wrist residuals (0-1).",
    )
    parser.add_argument(
        "--rot-weight", type=float, default=_ROT_WEIGHT,
        help="Weight for orientation error in IK.",
    )
    parser.add_argument(
        "--ik-damping", type=float, default=_IK_DAMPING,
        help="Damping factor for IK solver.",
    )
    parser.add_argument(
        "--ik-current-weight", type=float, default=_IK_CURRENT_WEIGHT,
        help="Weight for penalizing deviation from current pose in IK.",
    )
    parser.add_argument(
        "--hand-config",
        default=None,
        help="Path to retargeter YAML config (primarily used for kinova_wuji).",
    )
    parser.add_argument(
        "--disable-arm",
        action="store_true",
        help="Do not send arm commands; supported for kinova_wuji and rm65_inspire hand-only mode.",
    )
    parser.add_argument(
        "--disable-hand",
        action="store_true",
        help="Do not send dexterous-hand commands (currently mainly used for kinova_wuji).",
    )
    parser.add_argument("--input-source", default="quest3", choices=["quest3", "avp", "pico4"],
                        help="Input device: quest3 (UDP, default), avp, or pico4.")
    parser.add_argument("--avp-ip", default="192.168.1.100",
                        help="Apple Vision Pro IP address (used with --input-source avp).")
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"],
                        help="Pico 4 input mode: relay daemon (default) or direct TCP server.")
    parser.add_argument("--pico4-port", type=int, default=63901,
                        help="Pico 4 direct-mode TCP listen port.")
    parser.add_argument("--pico4-relay-host", default="127.0.0.1",
                        help="Pico 4 relay daemon host.")
    parser.add_argument("--pico4-relay-port", type=int, default=63902,
                        help="Pico 4 relay daemon port.")
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888,
                        help="Pico 4 direct-mode UDP broadcast port.")
    parser.add_argument("--inspire-port", default="/dev/ttyUSB0", help="Serial port for Inspire hand direct control.")
    parser.add_argument("--inspire-baudrate", type=int, default=115200, help="Baudrate for Inspire hand serial control.")
    parser.add_argument("--inspire-hand-id", type=int, default=1, help="Hand ID used in Inspire serial protocol.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hand-only mode (wuji, --disable-arm)
# ---------------------------------------------------------------------------


def _run_hand_only(args: argparse.Namespace) -> None:
    """Run pure hand retargeting without arm control."""
    hand_retargeter = HandRetargeter(_resolve_hand_config(args, "wuji"), "right")
    hand, hand_controller = _make_hand_controller(args)

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

    latest_hand_qpos = None

    print("Starting hand-only teleoperation loop (arm disabled)...")
    print("  Hand side: right")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            loop_start = time.time()

            if avp_input is not None:
                # --- Apple Vision Pro path ---
                if avp_input.poll():
                    # Check dual pinch stop gesture
                    if avp_input.check_stop_gesture():
                        print("\nStopping teleoperation (dual pinch)...")
                        break

                    if hand_retargeter.available:
                        mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                        if mediapipe_pts is not None:
                            result = hand_retargeter.retarget_mediapipe(mediapipe_pts)
                        if result is not None:
                            latest_hand_qpos = result
            elif pico4_input is not None:
                # --- Pico 4 path ---
                if hand_retargeter.available:
                    mediapipe_pts = pico4_input.get_landmarks_mediapipe("right")
                    if mediapipe_pts is not None:
                        result = hand_retargeter.retarget_mediapipe(mediapipe_pts)
                        if result is not None:
                            latest_hand_qpos = result
            else:
                # --- Quest 3 path (unchanged) ---
                packet = recv_latest_packet(sock)
                if packet is not None:
                    message = packet.decode("utf-8", errors="ignore")
                    if hand_retargeter.available:
                        landmarks = parse_right_landmarks(message)
                        if landmarks is not None:
                            result = hand_retargeter.retarget(landmarks)
                            if result is not None:
                                latest_hand_qpos = result

            if latest_hand_qpos is not None and hand_controller is not None:
                hand_controller.set_joint_target_position(
                    np.asarray(latest_hand_qpos, dtype=np.float64).reshape(5, 4)
                )

            elapsed = time.time() - loop_start
            sleep_time = _CONTROL_PERIOD_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        if hand is not None:
            try:
                hand.write_joint_enabled(False)
            except Exception as exc:
                print(f"Warning: failed to disable Wuji hand cleanly: {exc}")
        if sock is not None:
            sock.close()
        if pico4_input is not None:
            pico4_input.stop()


# ---------------------------------------------------------------------------
# Full arm + end-effector mode
# ---------------------------------------------------------------------------


def _run_arm_teleop(config: dict, args: argparse.Namespace) -> None:
    # Lazy import: keep Kortex SDK away from RM65 code paths
    import utilities as kortex_utilities
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2

    hand_type = config["hand_type"]
    is_wuji = hand_type == "wuji"

    # Load MuJoCo model
    if is_wuji and args.disable_hand:
        xml_path = Path(config["scene_xml_arm_only"]).resolve()
        print(f"Arm-only mode: using arm scene {xml_path}")
    else:
        xml_path = Path(config["scene_xml"]).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    state_data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _DEFAULT_SITE)
    if site_id == -1:
        raise ValueError(f"Site '{_DEFAULT_SITE}' not found in model.")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

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

    # Wuji-specific setup
    hand_retargeter = None
    hand = None
    hand_controller = None
    if is_wuji:
        hand_retargeter = HandRetargeter(_resolve_hand_config(args, "wuji"), "right")
        hand, hand_controller = _make_hand_controller(args)

    kinova_args = SimpleNamespace(
        ip=args.kinova_ip,
        username=args.kinova_username,
        password=args.kinova_password,
    )

    with kortex_utilities.DeviceConnection.createTcpConnection(kinova_args) as router, \
         kortex_utilities.DeviceConnection.createUdpConnection(kinova_args) as router_rt:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router_rt)

        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo_mode)

        _move_to_init_qpos(base, base_cyclic)

        current_q_rad = _get_measured_q_rad(base_cyclic)
        if current_q_rad.shape[0] != _NUM_ARM_JOINTS:
            raise RuntimeError("Expected 7 Kinova joint readings.")

        state_q = np.array(model.qpos0, dtype=np.float64)
        state_q[:_NUM_ARM_JOINTS] = current_q_rad
        state_data.qpos[: model.nq] = state_q
        state_data.qvel[:] = 0.0
        mujoco.mj_forward(model, state_data)

        initial_site_pos = state_data.site_xpos[site_id].copy()
        initial_site_quat = matrix_to_quaternion(
            state_data.site_xmat[site_id].reshape(3, 3).copy()
        )
        base_xmat = None
        if base_body_id != -1:
            base_xmat = state_data.xmat[base_body_id].reshape(3, 3).copy()

        # Wrist tracker
        tracker = WristTracker(
            initial_site_pos,
            initial_site_quat,
            position_scale=args.position_scale,
            ema_alpha=args.ema_alpha,
            negate_rot_xy=True if args.input_source == "quest3" else False,
            base_xmat=base_xmat,
            position_deadband=_WRIST_POS_DEADBAND,
            rotation_deadband_deg=_WRIST_ROT_DEADBAND_DEG,
        )

        # State variables
        latest_hand_qpos = None
        latest_gripper_pos = None
        last_log_time = time.time()
        last_valid_packet_time = 0.0

        print(f"  Initial arm q (deg): {np.rad2deg(current_q_rad).tolist()}")
        print(f"  HOME_QPOS (deg):     {np.rad2deg(_INIT_QPOS).tolist()}")
        print(f"  Initial EE pos:      {initial_site_pos.tolist()}")
        print(f"  model.nq={model.nq}  model.nv={model.nv}")
        print("Starting real teleoperation loop...")
        print(f"  Kinova IP: {args.kinova_ip}")
        print(f"  Quest UDP port: {args.port}")
        if is_wuji:
            print("  Hand side: right")
        else:
            print("  Gripper: Robotiq 2F-85 (Kortex API)")
        print(f"  Arm speed limit: +/-{_ARM_MAX_SPEED_DEG:.1f} deg/s")
        print(f"  Packet timeout: {_PACKET_TIMEOUT_S:.3f} s")
        print(f"  Wrist position deadband: {_WRIST_POS_DEADBAND:.3f} m")
        print(f"  Wrist rotation deadband: {_WRIST_ROT_DEADBAND_DEG:.1f} deg")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                loop_start = time.time()
                saw_valid_data = False

                if avp_input is not None:
                    # --- Apple Vision Pro path ---
                    if avp_input.poll():
                        # Check dual pinch stop gesture
                        if avp_input.check_stop_gesture():
                            print("\nStopping teleoperation (dual pinch)...")
                            break

                        # Hand control
                        if is_wuji and hand_retargeter is not None and hand_retargeter.available:
                            mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                            if mediapipe_pts is not None:
                                result = hand_retargeter.retarget_mediapipe(mediapipe_pts)
                                if result is not None:
                                    latest_hand_qpos = result
                                    saw_valid_data = True
                        elif not is_wuji:
                            pinch_distance = avp_input.get_pinch_distance("right")
                            if pinch_distance is not None:
                                latest_gripper_pos = _pinch_to_gripper_position(pinch_distance)
                                saw_valid_data = True

                        # Arm: wrist pose
                        # Rz(-90°) from get_wrist_pose + Rz(+90°) CW for
                        # real _INIT_QPOS[0]=90° base rotation.
                        wrist = avp_input.get_wrist_pose("right")
                        if wrist is not None:
                            from util.avp_input import apply_real_kinova_base_correction
                            robot_position, robot_quaternion = apply_real_kinova_base_correction(*wrist)
                            saw_valid_data = True
                            if not tracker.initialized:
                                print("Captured initial wrist reference pose.")
                            tracker.update(robot_position, robot_quaternion)
                elif pico4_input is not None:
                    # --- Pico 4 path ---
                    if is_wuji and hand_retargeter is not None and hand_retargeter.available:
                        mediapipe_pts = pico4_input.get_landmarks_mediapipe("right")
                        if mediapipe_pts is not None:
                            result = hand_retargeter.retarget_mediapipe(mediapipe_pts)
                            if result is not None:
                                latest_hand_qpos = result
                                saw_valid_data = True
                    elif not is_wuji:
                        pinch_distance = pico4_input.get_pinch_distance("right")
                        if pinch_distance is not None:
                            latest_gripper_pos = _pinch_to_gripper_position(pinch_distance)
                            saw_valid_data = True

                    wrist = pico4_input.get_wrist_pose("right")
                    if wrist is not None:
                        from util.avp_input import apply_real_kinova_base_correction
                        robot_position, robot_quaternion = apply_real_kinova_base_correction(*wrist)
                        saw_valid_data = True
                        if not tracker.initialized:
                            print("Captured initial wrist reference pose.")
                        tracker.update(robot_position, robot_quaternion)
                else:
                    # --- Quest 3 path (unchanged) ---
                    packet = recv_latest_packet(sock)

                    if packet is not None:
                        message = packet.decode("utf-8", errors="ignore")

                        # Hand control
                        if is_wuji and hand_retargeter is not None and hand_retargeter.available:
                            landmarks = parse_right_landmarks(message)
                            if landmarks is not None:
                                result = hand_retargeter.retarget(landmarks)
                                if result is not None:
                                    latest_hand_qpos = result
                                    saw_valid_data = True
                        elif not is_wuji:
                            landmarks = parse_right_landmarks(message)
                            if landmarks is not None:
                                pinch_distance = pinch_distance_from_landmarks(landmarks)
                                if pinch_distance is not None:
                                    latest_gripper_pos = _pinch_to_gripper_position(pinch_distance)
                                    saw_valid_data = True

                        # Arm: wrist pose residuals
                        wrist_pose = parse_right_wrist_pose(message)
                        if wrist_pose is not None:
                            robot_position, robot_quaternion = transform_quest3_raw_to_robot_pose(wrist_pose)
                            saw_valid_data = True
                            if not tracker.initialized:
                                print("Captured initial wrist reference pose.")
                            tracker.update(robot_position, robot_quaternion)

                if saw_valid_data:
                    last_valid_packet_time = loop_start

                # --- Logging ---
                now = time.time()
                if (
                    tracker.residual is not None
                    and tracker.euler_residual is not None
                    and now - last_log_time > 1.0
                ):
                    gripper_str = ""
                    if not is_wuji and latest_gripper_pos is not None:
                        gripper_str = f"  gripper: {latest_gripper_pos:.2f}"
                    print(
                        f"Wrist residual (xyz): {tracker.residual.tolist()} "
                        f"euler: {list(tracker.euler_residual)}{gripper_str}"
                    )
                    last_log_time = now

                # --- Send hand commands (wuji) ---
                if is_wuji and latest_hand_qpos is not None and hand_controller is not None:
                    hand_controller.set_joint_target_position(
                        np.asarray(latest_hand_qpos, dtype=np.float64).reshape(5, 4)
                    )

                # --- Send arm command ---
                current_q_rad = _get_measured_q_rad(base_cyclic)
                current_q_deg = np.rad2deg(current_q_rad)

                if (
                    not tracker.initialized
                    or now - last_valid_packet_time > _PACKET_TIMEOUT_S
                ):
                    base.SendJointSpeedsCommand(
                        _build_joint_speeds_command(np.zeros(_NUM_ARM_JOINTS, dtype=np.float64))
                    )
                else:
                    q_init = np.array(model.qpos0, dtype=np.float64)
                    q_init[:_NUM_ARM_JOINTS] = current_q_rad
                    q_sol = solve_pose_ik(
                        model,
                        ik_data,
                        site_id,
                        tracker.target_position,
                        tracker.target_quaternion,
                        q_init,
                        rot_weight=args.rot_weight,
                        damping=args.ik_damping,
                        current_q_weight=args.ik_current_weight,
                        home_qpos=_INIT_QPOS,
                        skip_tail_joints=0,
                    )
                    q_target_deg = np.rad2deg(q_sol[:_NUM_ARM_JOINTS])
                    q_err_deg = _angle_error_deg(q_target_deg, current_q_deg)
                    speed_deg_s = _ARM_KP * q_err_deg
                    speed_deg_s[np.abs(q_err_deg) < _ARM_DEADBAND_DEG] = 0.0
                    speed_deg_s = np.clip(speed_deg_s, -_ARM_MAX_SPEED_DEG, _ARM_MAX_SPEED_DEG)
                    if now - last_log_time < 0.1:
                        print(
                            f"  IK target pos: {tracker.target_position.tolist()}\n"
                            f"  q_current(deg): {current_q_deg.tolist()}\n"
                            f"  q_target (deg): {q_target_deg.tolist()}\n"
                            f"  q_err    (deg): {q_err_deg.tolist()}\n"
                            f"  speed  (deg/s): {speed_deg_s.tolist()}"
                        )
                    base.SendJointSpeedsCommand(_build_joint_speeds_command(speed_deg_s))

                # --- Send gripper command (gripper mode) ---
                if not is_wuji and latest_gripper_pos is not None:
                    _send_gripper_command(base, latest_gripper_pos)

                elapsed = time.time() - loop_start
                sleep_time = _CONTROL_PERIOD_S - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping teleoperation...")
        finally:
            _stop_arm(base)
            if hand is not None:
                try:
                    hand.write_joint_enabled(False)
                except Exception as exc:
                    print(f"Warning: failed to disable Wuji hand cleanly: {exc}")
            if sock is not None:
                sock.close()
            if pico4_input is not None:
                pico4_input.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    config = ROBOT_CONFIGS[args.robot]

    if args.robot in ("rm65", "rm65_gripper", "rm65_inspire"):
        _run_rm65_teleop(config, args)
    elif args.disable_arm and config["hand_type"] == "wuji":
        _run_hand_only(args)
    else:
        _run_arm_teleop(config, args)


if __name__ == "__main__":
    main()

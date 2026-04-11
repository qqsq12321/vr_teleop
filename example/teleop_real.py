"""Teleoperate a real Kinova Gen3 arm via VR hand tracking.

Supports multiple end-effector configurations via --robot:
  kinova_gripper  Kinova Gen3 + Robotiq 2F-85 gripper
  kinova_wuji     Kinova Gen3 + Wuji dexterous hand (20 DOF)

Input sources (--input-source):
  quest3          Meta Quest 3 via UDP (default)
  avp             Apple Vision Pro via avp_stream / Tracking Streamer

The --disable-arm flag enables hand-only mode (kinova_wuji only).

Examples:
    python example/teleop_real.py --robot kinova_gripper --kinova-ip 192.168.1.10
    python example/teleop_real.py --robot kinova_gripper --input-source avp --avp-ip 192.168.5.32
    python example/teleop_real.py --robot kinova_wuji --kinova-ip 192.168.1.10
    python example/teleop_real.py --robot kinova_wuji --disable-arm --input-source avp --avp-ip 192.168.5.32
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
    transform_vr_to_robot_pose,
)
from util.udp_socket import (
    make_socket,
    recv_latest_packet,
    parse_right_landmarks,
    parse_right_wrist_pose,
    pinch_distance_from_landmarks,
)
from util.wrist_tracker import WristTracker
from util.hand_retarget import HandRetargeter

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

import utilities as kortex_utilities  # noqa: E402
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient  # noqa: E402
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient  # noqa: E402
from kortex_api.autogen.messages import Base_pb2  # noqa: E402

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
    if pinch_distance < 0.03:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Shared Kortex utilities
# ---------------------------------------------------------------------------


def _angle_error_deg(target_deg: np.ndarray, current_deg: np.ndarray) -> np.ndarray:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _get_measured_q_rad(base_cyclic: BaseCyclicClient) -> np.ndarray:
    feedback = base_cyclic.RefreshFeedback()
    q_deg = np.array(
        [feedback.actuators[i].position for i in range(_NUM_ARM_JOINTS)],
        dtype=np.float64,
    )
    q_deg = np.where(q_deg > 180.0, q_deg - 360.0, q_deg)
    return np.deg2rad(q_deg)


def _build_joint_speeds_command(speed_deg_s: np.ndarray):
    joint_speeds = Base_pb2.JointSpeeds()
    for i, speed in enumerate(speed_deg_s.tolist()):
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i
        joint_speed.value = float(speed)
        joint_speed.duration = 0
    return joint_speeds


def _stop_arm(base: BaseClient) -> None:
    try:
        base.Stop()
    except Exception as exc:
        print(f"Warning: failed to stop Kinova arm cleanly: {exc}")


def _send_gripper_command(base: BaseClient, position: float) -> None:
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
    def check(notification, event=event):
        if (
            notification.action_event == Base_pb2.ACTION_END
            or notification.action_event == Base_pb2.ACTION_ABORT
        ):
            event.set()
    return check


def _move_arm_home(base: BaseClient, timeout: float = 30.0) -> bool:
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
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
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
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleop real Kinova Gen3 with Quest 3 hand tracking."
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
    # Wuji-specific
    parser.add_argument("--hand-config", default=None, help="Path to retargeter YAML config.")
    parser.add_argument("--disable-arm", action="store_true", help="Do not send commands to Kinova arm.")
    parser.add_argument("--disable-hand", action="store_true", help="Do not send commands to Wuji hand.")
    parser.add_argument("--input-source", default="quest3", choices=["quest3", "avp"],
                        help="Input device: quest3 (UDP, default) or avp (Vision Pro via avp_stream).")
    parser.add_argument("--avp-ip", default="192.168.1.100",
                        help="Apple Vision Pro IP address (used with --input-source avp).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hand-only mode (wuji, --disable-arm)
# ---------------------------------------------------------------------------


def _run_hand_only(args: argparse.Namespace) -> None:
    """Run pure hand retargeting without arm control."""
    hand_retargeter = HandRetargeter(args.hand_config, "right")
    hand, hand_controller = _make_hand_controller(args)

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
                    if avp_input.check_dual_pinch_stop():
                        print("\nStopping teleoperation (dual pinch)...")
                        break

                    if hand_retargeter.available:
                        mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                        if mediapipe_pts is not None:
                            result = hand_retargeter._retargeter.retarget(mediapipe_pts)
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
        sock.close()


# ---------------------------------------------------------------------------
# Full arm + end-effector mode
# ---------------------------------------------------------------------------


def _run_arm_teleop(config: dict, args: argparse.Namespace) -> None:
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
    if args.input_source == "avp":
        from util.avp_input import AVPInput
        avp_input = AVPInput(ip=args.avp_ip)
        print(f"  Input: Apple Vision Pro ({args.avp_ip})")
    else:
        sock = make_socket(args.port)
        print(f"  Input: Quest 3 (UDP port {args.port})")

    # Wuji-specific setup
    hand_retargeter = None
    hand = None
    hand_controller = None
    if is_wuji:
        hand_retargeter = HandRetargeter(args.hand_config, "right")
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

        if is_wuji:
            _move_arm_home(base, timeout=_HOME_TIMEOUT_S)
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
            negate_rot_xy=True if args.input_source != "avp" else False,
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
                        if avp_input.check_dual_pinch_stop():
                            print("\nStopping teleoperation (dual pinch)...")
                            break

                        # Hand control
                        if is_wuji and hand_retargeter is not None and hand_retargeter.available:
                            mediapipe_pts = avp_input.get_landmarks_mediapipe("right")
                            if mediapipe_pts is not None:
                                result = hand_retargeter._retargeter.retarget(mediapipe_pts)
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
                            from util.avp_input import apply_rz90cw
                            robot_position, robot_quaternion = apply_rz90cw(*wrist)
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
                            raw_x, raw_y, raw_z = wrist_pose[0], wrist_pose[1], wrist_pose[2]
                            wrist_position = (raw_z, raw_y, -raw_x)
                            wrist_quaternion = (
                                wrist_pose[5], wrist_pose[4], -wrist_pose[3], wrist_pose[6]
                            )
                            robot_position, robot_quaternion = transform_vr_to_robot_pose(
                                wrist_position, wrist_quaternion
                            )
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
            sock.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    config = ROBOT_CONFIGS[args.robot]

    if args.disable_arm and config["hand_type"] == "wuji":
        _run_hand_only(args)
    else:
        _run_arm_teleop(config, args)


if __name__ == "__main__":
    main()

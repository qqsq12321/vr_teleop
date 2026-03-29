"""Teleoperate a real Kinova Gen3 arm + Robotiq 2F-85 gripper via Quest 3 hand tracking.

Design:
- Quest wrist pose -> residual target pose -> MuJoCo IK -> Kinova joint-speed commands
- Quest pinch distance -> Kortex gripper position command
- MuJoCo is used only as a kinematic / IK model, not as a simulator

Example:
    python teleop_env/teleop_kinova_gripper_real.py --port 9000 --kinova-ip 192.168.1.10
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import math
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from util.ik import solve_pose_ik
from util.quaternion import (
    matrix_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    transform_vr_to_robot_pose,
)
from util.udp_socket import (
    parse_right_landmarks,
    parse_right_wrist_pose,
    pinch_distance_from_landmarks,
)

_KORTEX_EXAMPLES_DIR = (
    _Path(__file__).resolve().parents[1]
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

# Gen3 initial pose (7 arm joints) used as IK prior and startup target.
_INIT_QPOS = np.array(
    [1.57079633, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633],
    dtype=np.float64,
)

_NUM_ARM_JOINTS = 7

# Pinch distance normalization (meters)
_PINCH_MAX_DISTANCE = 0.1

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
_GRIPPER_EMA_ALPHA = 0.3
_GRIPPER_COMMAND_THRESHOLD = 0.02
_HOME_TIMEOUT_S = 30.0
_DEFAULT_SITE = "kinova_ee_site"


def _default_scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene" / "scene_kinova_gen3.xml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleop real Kinova Gen3 + Robotiq gripper with Quest wrist residuals + pinch."
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--kinova-ip",
        default="192.168.1.10",
        help="Kinova robot IP.",
    )
    parser.add_argument(
        "--kinova-username",
        default="admin",
        help="Kinova username.",
    )
    parser.add_argument(
        "--kinova-password",
        default="admin",
        help="Kinova password.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=_POSITION_SCALE,
        help="Scale for wrist position residuals.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=_EMA_ALPHA,
        help="EMA smoothing factor for wrist residuals (0-1).",
    )
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=_ROT_WEIGHT,
        help="Weight for orientation error in IK.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=_IK_DAMPING,
        help="Damping factor for IK solver.",
    )
    parser.add_argument(
        "--ik-current-weight",
        type=float,
        default=_IK_CURRENT_WEIGHT,
        help="Weight for penalizing deviation from current pose in IK.",
    )
    return parser.parse_args()


def _pinch_to_gripper_position(pinch_distance: float) -> float:
    """Map pinch distance to Kortex gripper position (0.0=open, 1.0=closed).

    Small distance (pinched) -> large value (closed).
    Large distance (open hand) -> small value (open).
    """
    scaled = pinch_distance / _PINCH_MAX_DISTANCE
    clamped = min(1.0, max(0.0, scaled))
    return 1.0 - clamped


def _angle_error_deg(target_deg: np.ndarray, current_deg: np.ndarray) -> np.ndarray:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _recv_latest_packet(sock: socket.socket) -> bytes | None:
    latest = None
    while True:
        try:
            latest, _ = sock.recvfrom(65536)
        except BlockingIOError:
            break
    return latest


def _make_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    return sock


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
    """Send gripper position command via Kortex high-level API.

    Args:
        base: Kortex BaseClient.
        position: Gripper target position, 0.0 (open) to 1.0 (closed).
    """
    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 0
    finger.value = float(position)
    base.SendGripperCommand(gripper_command)


def _apply_position_deadband(vec: np.ndarray, deadband: float) -> np.ndarray:
    if deadband <= 0.0:
        return vec
    if np.linalg.norm(vec) < deadband:
        return np.zeros_like(vec)
    return vec


def _quaternion_rotation_angle_rad(q: tuple[float, float, float, float]) -> float:
    x, y, z, w = q
    sin_half = math.sqrt(x * x + y * y + z * z)
    return 2.0 * math.atan2(sin_half, abs(w))


def _apply_rotation_deadband(
    q: tuple[float, float, float, float], deadband_deg: float
) -> tuple[float, float, float, float]:
    if deadband_deg <= 0.0:
        return q
    deadband_rad = math.radians(deadband_deg)
    if _quaternion_rotation_angle_rad(q) < deadband_rad:
        return (0.0, 0.0, 0.0, 1.0)
    return q


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
    """Drive arm from current pose to _INIT_QPOS using joint speed commands."""
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


def main() -> None:
    args = _parse_args()

    xml_path = _default_scene_path().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    state_data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _DEFAULT_SITE)
    if site_id == -1:
        raise ValueError(f"Site '{_DEFAULT_SITE}' not found in model.")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    sock = _make_socket(args.port)

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

        initial_wrist_position = None
        initial_wrist_quaternion = None
        target_position = initial_site_pos.copy()
        target_quaternion = np.array(initial_site_quat, dtype=np.float64)
        latest_residual = None
        latest_euler_residual = None
        smoothed_residual = None
        latest_gripper_pos = None
        smoothed_gripper_pos = None
        last_sent_gripper_pos = None
        last_log_time = time.time()
        last_valid_packet_time = 0.0

        print(f"  Initial arm q (deg): {np.rad2deg(current_q_rad).tolist()}")
        print(f"  HOME_QPOS (deg):     {np.rad2deg(_INIT_QPOS).tolist()}")
        print(f"  Initial EE pos:      {initial_site_pos.tolist()}")
        print(f"  model.nq={model.nq}  model.nv={model.nv}")
        print("Starting real teleoperation loop...")
        print(f"  Kinova IP: {args.kinova_ip}")
        print(f"  Quest UDP port: {args.port}")
        print(f"  Arm speed limit: ±{_ARM_MAX_SPEED_DEG:.1f} deg/s")
        print(f"  Packet timeout: {_PACKET_TIMEOUT_S:.3f} s")
        print(f"  Wrist position deadband: {_WRIST_POS_DEADBAND:.3f} m")
        print(f"  Wrist rotation deadband: {_WRIST_ROT_DEADBAND_DEG:.1f} deg")
        print("  Gripper: Robotiq 2F-85 (Kortex API)")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                loop_start = time.time()
                packet = _recv_latest_packet(sock)

                if packet is not None:
                    message = packet.decode("utf-8", errors="ignore")
                    saw_valid_data = False

                    # --- Gripper: pinch distance from landmarks ---
                    landmarks = parse_right_landmarks(message)
                    if landmarks is not None:
                        pinch_distance = pinch_distance_from_landmarks(landmarks)
                        if pinch_distance is not None:
                            raw_gripper_pos = _pinch_to_gripper_position(pinch_distance)
                            if smoothed_gripper_pos is None:
                                smoothed_gripper_pos = raw_gripper_pos
                            else:
                                smoothed_gripper_pos = (
                                    _GRIPPER_EMA_ALPHA * raw_gripper_pos
                                    + (1.0 - _GRIPPER_EMA_ALPHA) * smoothed_gripper_pos
                                )
                            latest_gripper_pos = smoothed_gripper_pos
                            saw_valid_data = True

                    # --- Arm: wrist pose residuals ---
                    wrist_pose = parse_right_wrist_pose(message)
                    if wrist_pose is not None:
                        # Rotate camera coordinate system +90° around Y (clockwise when looking down +Y):
                        # x' = z, y' = y, z' = -x
                        raw_x, raw_y, raw_z = wrist_pose[0], wrist_pose[1], wrist_pose[2]
                        wrist_position = (raw_z, raw_y, -raw_x)
                        wrist_quaternion = (
                            wrist_pose[5],
                            wrist_pose[4],
                            -wrist_pose[3],
                            wrist_pose[6],
                        )
                        robot_position, robot_quaternion = transform_vr_to_robot_pose(
                            wrist_position, wrist_quaternion
                        )
                        saw_valid_data = True
                        if initial_wrist_position is None:
                            initial_wrist_position = robot_position
                            initial_wrist_quaternion = robot_quaternion
                            print("Captured initial wrist reference pose.")
                        else:
                            residual = np.array(
                                [
                                    robot_position[0] - initial_wrist_position[0],
                                    robot_position[1] - initial_wrist_position[1],
                                    robot_position[2] - initial_wrist_position[2],
                                ],
                                dtype=np.float64,
                            )
                            if base_xmat is not None:
                                residual = base_xmat @ residual
                            if smoothed_residual is None:
                                smoothed_residual = residual
                            else:
                                smoothed_residual = (
                                    args.ema_alpha * residual
                                    + (1.0 - args.ema_alpha) * smoothed_residual
                                )
                            smoothed_residual = _apply_position_deadband(
                                smoothed_residual, _WRIST_POS_DEADBAND
                            )
                            target_position = (
                                initial_site_pos + args.position_scale * smoothed_residual
                            )

                            relative_quaternion = quaternion_multiply(
                                robot_quaternion,
                                quaternion_inverse(initial_wrist_quaternion),
                            )
                            relative_quaternion = (
                                -relative_quaternion[0],
                                -relative_quaternion[1],
                                relative_quaternion[2],
                                relative_quaternion[3],
                            )
                            relative_quaternion = _apply_rotation_deadband(
                                relative_quaternion, _WRIST_ROT_DEADBAND_DEG
                            )
                            target_quaternion = np.array(
                                quaternion_multiply(relative_quaternion, initial_site_quat),
                                dtype=np.float64,
                            )
                            norm = np.linalg.norm(target_quaternion)
                            if norm > 0.0:
                                target_quaternion /= norm
                            latest_residual = smoothed_residual
                            latest_euler_residual = quaternion_to_euler_xyz(
                                relative_quaternion[0],
                                relative_quaternion[1],
                                relative_quaternion[2],
                                relative_quaternion[3],
                            )

                    if saw_valid_data:
                        last_valid_packet_time = loop_start

                now = time.time()
                if (
                    latest_residual is not None
                    and latest_euler_residual is not None
                    and now - last_log_time > 1.0
                ):
                    gripper_str = f"  gripper: {latest_gripper_pos:.2f}" if latest_gripper_pos is not None else ""
                    print(
                        f"Wrist residual (xyz): {latest_residual.tolist()} "
                        f"euler: {list(latest_euler_residual)}{gripper_str}"
                    )
                    last_log_time = now

                # --- Send gripper command ---
                if latest_gripper_pos is not None and (
                    last_sent_gripper_pos is None
                    or abs(latest_gripper_pos - last_sent_gripper_pos) > _GRIPPER_COMMAND_THRESHOLD
                ):
                    _send_gripper_command(base, latest_gripper_pos)
                    last_sent_gripper_pos = latest_gripper_pos

                # --- Send arm command ---
                current_q_rad = _get_measured_q_rad(base_cyclic)
                current_q_deg = np.rad2deg(current_q_rad)

                if (
                    initial_wrist_position is None
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
                        target_position,
                        target_quaternion,
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
                    speed_deg_s = np.clip(
                        speed_deg_s,
                        -_ARM_MAX_SPEED_DEG,
                        _ARM_MAX_SPEED_DEG,
                    )
                    if now - last_log_time < 0.1:
                        print(
                            f"  IK target pos: {target_position.tolist()}\n"
                            f"  q_current(deg): {current_q_deg.tolist()}\n"
                            f"  q_target (deg): {q_target_deg.tolist()}\n"
                            f"  q_err    (deg): {q_err_deg.tolist()}\n"
                            f"  speed  (deg/s): {speed_deg_s.tolist()}"
                        )
                    base.SendJointSpeedsCommand(_build_joint_speeds_command(speed_deg_s))

                elapsed = time.time() - loop_start
                sleep_time = _CONTROL_PERIOD_S - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping teleoperation...")
        finally:
            _stop_arm(base)
            sock.close()


if __name__ == "__main__":
    main()

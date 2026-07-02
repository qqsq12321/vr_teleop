"""Teleoperate a real Galbot G1 dual arm + dual Inspire hand setup.

Outputs:
  - Galbot SDK: left/right arm joint commands.
  - Serial: left/right Inspire RH56DFX hand commands.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(_Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"))

import argparse
import time
from pathlib import Path

import mujoco

from utils.adapters.output.galbot_real import GalbotRealRobotOutput
from utils.adapters.output.inspire_real import InspireSerialOutput
from utils.core.arm_controller import TeleopArmController
from utils.core.input_stream import TeleopInputStream
from utils.core.quaternion import transform_quest3_raw_to_robot_pose
from utils.robots.galbot import GalbotInspireSpec


_SPEC = GalbotInspireSpec()

_CONTROL_PERIOD_S = 0.02
_PACKET_TIMEOUT_S = 0.25
_POSITION_SCALE = 3.0
_EMA_ALPHA = 0.8
_ROT_WEIGHT = 1.0
_IK_DAMPING = 1e-3
_IK_CURRENT_WEIGHT = 0.1
_WRIST_POS_DEADBAND = 0.03
_WRIST_ROT_DEADBAND_DEG = 8.0
_POS_GAIN = 0.08
_ROT_GAIN = 0.08


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        return
    data.qpos[model.jnt_qposadr[jid]] = value


def _apply_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for side in ("left", "right"):
        for joint_name, value in zip(_SPEC.arm_joint_names(side), _SPEC.arm_home_qpos(side)):
            _set_joint_qpos(model, data, joint_name, float(value))
    for joint_name, value in zip(_SPEC.torso_joint_names, _SPEC.torso_home_qpos):
        _set_joint_qpos(model, data, joint_name, float(value))
    for joint_name, value in zip(_SPEC.head_joint_names, _SPEC.head_home_qpos):
        _set_joint_qpos(model, data, joint_name, float(value))
    mujoco.mj_forward(model, data)


def _controller_kwargs(side: str) -> dict:
    return {
        "robot_label": "Galbot",
        "home_qpos": _SPEC.arm_home_qpos(side),
        "arm_joint_names": _SPEC.arm_joint_names(side),
        "hand_joint_names": _SPEC.hand_joint_names(side),
        "ee_body_name": _SPEC.ee_body_name(side),
        "base_body_name": _SPEC.base_body_name,
        "hand_qpos_mapping": _SPEC.hand_qpos_mapping,
        "position_deadband": _WRIST_POS_DEADBAND,
        "rotation_deadband_deg": _WRIST_ROT_DEADBAND_DEG,
        "pos_gain": _POS_GAIN,
        "rot_gain": _ROT_GAIN,
    }


def _make_arm_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_data: mujoco.MjData,
    side: str,
    args: argparse.Namespace,
) -> TeleopArmController:
    hand_config = args.hand_config or str(_SPEC.hand_config_path(args.input_source, side))
    return TeleopArmController(
        model,
        data,
        ik_data,
        side,
        args,
        hand_config=hand_config,
        hand_type=_SPEC.hand_type,
        **_controller_kwargs(side),
    )


def _open_hand_output(port: str | None, baudrate: int, hand_id: int, side: str):
    if not port:
        raise ValueError(f"--{side}-hand-port is required unless --disable-hands is set.")
    print(f"  {side.capitalize()} hand serial: {port} @ {baudrate}, id={hand_id}")
    return InspireSerialOutput(port, baudrate=baudrate, hand_id=hand_id)


def _send_hand(side: str, output: InspireSerialOutput | None, controller: TeleopArmController) -> InspireSerialOutput | None:
    if output is None:
        return None
    mapped = controller.mapped_hand_qpos()
    if mapped is None:
        return output
    try:
        output.send_hand_qpos(mapped)
    except Exception as exc:
        print(f"Warning: disabling {side} Inspire hand after serial send failure: {exc}")
        try:
            output.close()
        except Exception:
            pass
        return None
    return output


def _run_galbot_real(args: argparse.Namespace) -> None:
    xml_path = Path(args.scene or _SPEC.scene_path).expanduser().resolve()
    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    _apply_initial_pose(model, data)

    galbot = GalbotRealRobotOutput(
        _SPEC,
        skip_controller_switch=args.skip_controller_switch,
        strict_joint_names=not args.no_strict_joint_names,
        command_time_from_start_s=args.command_time_from_start,
    )
    left_hand = None
    right_hand = None
    input_stream = None

    try:
        galbot.connect()

        left_arm = _make_arm_controller(model, data, ik_data, "left", args)
        right_arm = _make_arm_controller(model, data, ik_data, "right", args)

        left_arm.sync_from_joint_positions(galbot.get_joint_positions_rad("left"))
        right_arm.sync_from_joint_positions(galbot.get_joint_positions_rad("right"))

        if not args.disable_hands:
            left_hand = _open_hand_output(args.left_hand_port, args.hand_baudrate, args.left_hand_id, "left")
            right_hand = _open_hand_output(args.right_hand_port, args.hand_baudrate, args.right_hand_id, "right")

        input_stream = TeleopInputStream(args)

        hand_period_s = 1.0 / max(args.hand_rate_hz, 1.0)
        last_hand_send_time = 0.0
        last_log_time = time.time()
        last_valid_packet_time = 0.0

        print("Starting Galbot real teleoperation loop...")
        print("Press Ctrl+C to stop.")

        while True:
            loop_start = time.time()
            now = loop_start

            poll_result = input_stream.poll_arms(
                left_arm,
                right_arm,
                transform_quest3_raw_to_robot_pose,
                allow_stop_gesture=True,
            )
            if poll_result.stop_requested:
                print("\nStopping teleoperation (stop gesture)...")
                break
            if poll_result.saw_valid_data:
                last_valid_packet_time = loop_start

            q_left = left_arm.step_smoothed_ik(last_valid_packet_time, now, _PACKET_TIMEOUT_S)
            q_right = right_arm.step_smoothed_ik(last_valid_packet_time, now, _PACKET_TIMEOUT_S)

            if q_left is not None:
                galbot.send_arm_qpos("left", q_left[left_arm.arm_dof_indices])
                left_arm.write_solution_to_data(q_left, include_hand=False)
            if q_right is not None:
                galbot.send_arm_qpos("right", q_right[right_arm.arm_dof_indices])
                right_arm.write_solution_to_data(q_right, include_hand=False)

            if time.time() - last_hand_send_time >= hand_period_s:
                left_hand = _send_hand("left", left_hand, left_arm)
                right_hand = _send_hand("right", right_hand, right_arm)
                last_hand_send_time = time.time()

            now = time.time()
            if left_arm.latest_residual is not None and now - last_log_time > 1.0:
                print(f"L resid: {left_arm.latest_residual} R resid: {right_arm.latest_residual}")
                last_log_time = now

            sleep_time = _CONTROL_PERIOD_S - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        for hand in (left_hand, right_hand):
            if hand is not None:
                try:
                    hand.close()
                except Exception as exc:
                    print(f"Warning: Inspire hand close failed: {exc}")
        if input_stream is not None:
            input_stream.close()
        galbot.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate real Galbot G1 arms and serial Inspire hands with VR hand tracking."
    )
    parser.add_argument("--scene", default=None, help="Override Galbot Inspire scene XML path.")
    parser.add_argument("--port", type=int, default=9000, help="Quest 3 UDP port to listen on.")
    parser.add_argument("--hand-config", default=None, help="Override AnyDex retargeter YAML config.")
    parser.add_argument("--position-scale", type=float, default=_POSITION_SCALE)
    parser.add_argument("--ema-alpha", type=float, default=_EMA_ALPHA)
    parser.add_argument("--rot-weight", type=float, default=_ROT_WEIGHT)
    parser.add_argument("--ik-damping", type=float, default=_IK_DAMPING)
    parser.add_argument("--ik-current-weight", type=float, default=_IK_CURRENT_WEIGHT)
    parser.add_argument("--command-time-from-start", type=float, default=0.0)
    parser.add_argument("--skip-controller-switch", action="store_true")
    parser.add_argument("--no-strict-joint-names", action="store_true")

    parser.add_argument("--disable-hands", action="store_true")
    parser.add_argument("--left-hand-port", default=None, help="Serial port for left Inspire hand.")
    parser.add_argument("--right-hand-port", default=None, help="Serial port for right Inspire hand.")
    parser.add_argument("--left-hand-id", type=int, default=1)
    parser.add_argument("--right-hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-rate-hz", type=float, default=25.0)

    parser.add_argument(
        "--input-source",
        default="quest3",
        choices=["quest3", "avp", "pico4"],
        help="Input device: quest3 (UDP, default), avp, or pico4.",
    )
    parser.add_argument("--avp-ip", default="192.168.1.100")
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"])
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-relay-host", default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_galbot_real(args)


if __name__ == "__main__":
    main()

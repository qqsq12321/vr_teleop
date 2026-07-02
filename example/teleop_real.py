"""Teleoperate the real DexForce dual-arm + dual-hand humanoid via VR hand tracking.

This is a real-robot skeleton adapted from the simulation path. The VR input
pipeline, WristTracker + Cartesian smoothing, SLERP, and bimanual control loop
are preserved. The DexForce real-robot SDK calls are left as TODO placeholders
to be filled in during adaptation.

Input sources (--input-source):
  quest3          Meta Quest 3 via UDP (default)
  avp             Apple Vision Pro via avp_stream / Tracking Streamer
  pico4           Pico 4 via relay / direct TCP

Examples:
    python example/teleop_real.py --dexforce-ip 192.168.1.50
    python example/teleop_real.py --input-source avp --avp-ip 192.168.5.32
    python example/teleop_real.py --input-source pico4
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

from utils.core.arm_controller import TeleopArmController
from utils.core.input_stream import TeleopInputStream
from utils.core.hand.retarget import (
    default_linker_l20_config_path,
    default_pico4_config_path,
)
from utils.core.quaternion import (
    transform_quest3_raw_to_robot_pose,
)
from utils.adapters.output.dexforce_real import DexForceRealArm
from utils.robots.dexforce import DexForceSpec

# ---------------------------------------------------------------------------
# Scene / config paths
# ---------------------------------------------------------------------------

_DEXFORCE_SPEC = DexForceSpec()
_DEXFORCE_SCENE = _DEXFORCE_SPEC.scene_path
_DEXFORCE_HAND_TYPE = _DEXFORCE_SPEC.hand_type

# ---------------------------------------------------------------------------
# Real-robot tuning constants (adapt during DexForce integration)
# ---------------------------------------------------------------------------

_CONTROL_PERIOD_S = 0.02
_PACKET_TIMEOUT_S = 0.25
_POSITION_SCALE = 3.0
_EMA_ALPHA = 0.8
_ROT_WEIGHT = 1.0
_IK_DAMPING = 1e-3
_IK_CURRENT_WEIGHT = 0.1
_WRIST_POS_DEADBAND = 0.03
_WRIST_ROT_DEADBAND_DEG = 8.0
# Cartesian smoothing gains (per-frame fraction toward tracker target)
_POS_GAIN = 0.08
_ROT_GAIN = 0.08


def _resolve_hand_config(args: argparse.Namespace, side: str = "right") -> str:
    """Resolve linker_l20 retarget config path for the given input source / side."""
    if args.hand_config:
        return args.hand_config
    if args.input_source == "pico4":
        return str(default_pico4_config_path(side=side))
    return str(default_linker_l20_config_path(args.input_source, side=side))


# ---------------------------------------------------------------------------
# Initial pose helpers
# ---------------------------------------------------------------------------


def _apply_initial_pose_dexforce(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def _controller_kwargs(side: str) -> dict:
    return {
        "robot_label": "DexForce",
        "home_qpos": _DEXFORCE_SPEC.arm_home_qpos(side),
        "arm_joint_names": _DEXFORCE_SPEC.arm_joint_names(side),
        "hand_joint_names": _DEXFORCE_SPEC.hand_joint_names(side),
        "ee_body_name": _DEXFORCE_SPEC.ee_body_name(side),
        "base_body_name": _DEXFORCE_SPEC.base_body_name,
        "hand_qpos_mapping": _DEXFORCE_SPEC.hand_qpos_mapping,
        "position_deadband": _WRIST_POS_DEADBAND,
        "rotation_deadband_deg": _WRIST_ROT_DEADBAND_DEG,
        "pos_gain": _POS_GAIN,
        "rot_gain": _ROT_GAIN,
    }


def _dispatch_real_arm(controller: TeleopArmController, real_arm: DexForceRealArm, q_sol) -> None:
    if q_sol is None:
        return
    real_arm.send_arm_qpos(q_sol[controller.arm_dof_indices])
    mapped_hand_qpos = controller.mapped_hand_qpos()
    if mapped_hand_qpos is not None:
        real_arm.send_hand_qpos(mapped_hand_qpos)
    controller.write_solution_to_data(q_sol, include_hand=False)


# ---------------------------------------------------------------------------
# DexForce real-robot bimanual teleop loop
# ---------------------------------------------------------------------------


def _run_dexforce_real(args: argparse.Namespace) -> None:
    xml_path = Path(args.scene or _DEXFORCE_SCENE).expanduser().resolve()
    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose_dexforce(model, data)

    dexforce_hand_type = _DEXFORCE_HAND_TYPE
    left_hand_config = _resolve_hand_config(args, side="left")
    right_hand_config = _resolve_hand_config(args, side="right")

    # Connect to real DexForce arms/hands
    left_real = DexForceRealArm("left", args.dexforce_ip)
    right_real = DexForceRealArm("right", args.dexforce_ip)
    left_real.connect()
    right_real.connect()

    left_arm = TeleopArmController(
        model, data, ik_data, "left", args,
        hand_config=left_hand_config,
        hand_type=dexforce_hand_type,
        **_controller_kwargs("left"),
    )
    right_arm = TeleopArmController(
        model, data, ik_data, "right", args,
        hand_config=right_hand_config,
        hand_type=dexforce_hand_type,
        **_controller_kwargs("right"),
    )

    # Sync MuJoCo state with real joint readings + reset tracker baseline
    left_arm.sync_from_joint_positions(left_real.get_joint_positions_rad())
    right_arm.sync_from_joint_positions(right_real.get_joint_positions_rad())

    input_stream = TeleopInputStream(args)

    print(f"  DexForce IP: {args.dexforce_ip}")
    print("Starting DexForce real teleoperation loop...")
    print("Press Ctrl+C to stop.")

    last_log_time = time.time()
    last_valid_packet_time = 0.0

    try:
        while True:
            loop_start = time.time()
            now = loop_start
            saw_valid_data = False

            poll_result = input_stream.poll_arms(
                left_arm,
                right_arm,
                transform_quest3_raw_to_robot_pose,
                allow_stop_gesture=True,
            )
            if poll_result.stop_requested:
                print("\nStopping teleoperation (stop gesture)...")
                break
            saw_valid_data = poll_result.saw_valid_data

            if saw_valid_data:
                last_valid_packet_time = loop_start

            # --- Dispatch to real arms/hands ---
            q_left = left_arm.step_smoothed_ik(last_valid_packet_time, now, _PACKET_TIMEOUT_S)
            q_right = right_arm.step_smoothed_ik(last_valid_packet_time, now, _PACKET_TIMEOUT_S)
            _dispatch_real_arm(left_arm, left_real, q_left)
            _dispatch_real_arm(right_arm, right_real, q_right)

            # --- Logging ---
            now = time.time()
            if (
                left_arm.latest_residual is not None
                and now - last_log_time > 1.0
            ):
                print(
                    f"L resid: {left_arm.latest_residual} "
                    f"R resid: {right_arm.latest_residual}"
                )
                last_log_time = now

            elapsed = time.time() - loop_start
            sleep_time = _CONTROL_PERIOD_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        for arm in (left_real, right_real):
            try:
                arm.close()
            except Exception as exc:
                print(f"Warning: DexForce {arm.side} close failed: {exc}")
        input_stream.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate the real DexForce dual-arm humanoid with VR hand tracking."
    )
    parser.add_argument("--dexforce-ip", default="192.168.1.50", help="DexForce robot IP.")
    parser.add_argument(
        "--scene",
        default=None,
        help="Override scene XML path (default: scene_dexforce.xml).",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--hand-config",
        default=None,
        help="Path to linker_l20 retargeter YAML config.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_dexforce_real(args)


if __name__ == "__main__":
    main()

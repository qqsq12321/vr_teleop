#!/usr/bin/env python3
"""Teleoperate a real CVTE CARM D3 dual arm (+ parallel grippers) from VR.

Pipeline: VR wrist tracking -> MuJoCo pose IK -> SDK ``track_*_joint`` at
--publish-hz, with the thumb-index pinch distance opening/closing the gripper.

The arms teleoperate from wherever they currently are; --home first performs a
smooth --home-duration ramp to the "forearms forward" pose. Commands are clamped to
the joint limits and rate-limited (--max-joint-speed). Any stale feedback,
controller error, or persistent joint tracking error stops the session.

The D3 joints have NO brakes: on exit this stops the tracking task and leaves
the servos enabled so the arms hold. Keep the hardware E-stop in reach, and
remember that pressing it (or powering off) drops the arms.

Requires the vendor SDK (``pip install carm``) and a PC address on 10.42.0.x.
Bring-up order:

    python example/teleop_cvte_real.py --read-only --report      # nothing is sent
    python example/teleop_cvte_real.py --arm left --input-source avp --avp-ip <ip>
    python example/teleop_cvte_real.py --input-source avp --avp-ip <ip>
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.adapters.output.cvte_real import SIDES, CVTEError, CVTERealOutput, sim_gripper_to_opening
from utils.core.arm_controller import TeleopArmController
from utils.core.quaternion import matrix_to_quaternion, transform_vr_to_robot_pose
from utils.robots.cvte_carm_d3 import CVTECArmD3Spec

_SPEC = CVTECArmD3Spec()
LOG = logging.getLogger(__name__)

PACKET_TIMEOUT_S = 0.25
POSITION_SCALE = 1.0  # 1:1 hand-to-gripper translation.
EMA_ALPHA = 0.8
ROT_WEIGHT = 1.0
IK_DAMPING = 1e-3
IK_CURRENT_WEIGHT = 0.1
POS_GAIN = 0.3  # ~0.13 s to cover 90% of a step; 1.0 = no smoothing, as in simulation.
ROT_GAIN = 0.3
POSITION_DEADBAND = 0.0  # Off, as in simulation: small motions are followed.
ROTATION_DEADBAND_DEG = 0.0
HOME_TOLERANCE_RAD = 0.02
HOME_SETTLE_TIMEOUT_S = 5.0


def _quest3_pose_converter(wrist_pose):
    return transform_vr_to_robot_pose(wrist_pose[:3], wrist_pose[3:7])


def _make_controller(model, data, ik_data, side, args):
    return TeleopArmController(
        model,
        data,
        ik_data,
        side,
        args,
        hand_config=None,  # The gripper is driven by pinch distance, not AnyDex.
        hand_type=_SPEC.hand_type,
        robot_label="CVTE D3",
        home_qpos=_SPEC.arm_home_qpos(side),
        arm_joint_names=_SPEC.arm_joint_names(side),
        hand_joint_names=_SPEC.hand_joint_names(side),
        ee_body_name=_SPEC.ee_body_name(side),
        base_body_name=_SPEC.base_body_name,
        hand_qpos_mapping=_SPEC.hand_qpos_mapping,
        position_deadband=args.position_deadband,
        rotation_deadband_deg=args.rotation_deadband,
        pos_gain=args.pos_gain,
        rot_gain=args.rot_gain,
    )


def _sync_measured(model, data, arms, output) -> None:
    import mujoco

    for side, controller in arms.items():
        controller.sync_from_joint_positions(output.read_arm_positions(side))
    mujoco.mj_forward(model, data)


def _flange_in_base(model, data, side):
    """Flange position/quaternion (xyzw) in base_link, plus position in the shoulder frame."""
    base = model.body(_SPEC.base_body_name).id
    flange = model.body(_SPEC.flange_body_name(side)).id
    base_rot = data.xmat[base].reshape(3, 3)
    position = base_rot.T @ (data.xpos[flange] - data.xpos[base])
    quat_xyzw = matrix_to_quaternion(base_rot.T @ data.xmat[flange].reshape(3, 3))
    shoulder = model.body_pos[model.body(f"link1-{side.upper()}").id]
    return position, np.asarray(quat_xyzw), position - shoulder


def _quat_angle_deg(qa, qb) -> float:
    dot = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def _print_report(model, data, output, gripper_sides) -> None:
    print()
    print("=" * 72)
    print("REPORT (read-only queries; nothing is sent by this report)")
    print("=" * 72)
    for side in SIDES:
        status = output.status(side)
        print(f"[{side}] arm_name={status.get('arm_name')!r} dof={status.get('arm_dof')} "
              f"state={status.get('state')} servo={status.get('servo')} "
              f"fsm={status.get('fsm_state')!r} speed%={status.get('speed_percentage')}")
        q = output.read_arm_positions(side)
        print(f"  measured joints (deg): {np.round(np.degrees(q), 2).tolist()}")
        if side in gripper_sides:
            print(f"  gripper: detected, opening range {output.gripper_range(side)} m")
        else:
            print("  gripper: not detected (pinch control disabled for this side)")
        position, quat, shoulder = _flange_in_base(model, data, side)
        cart = output.read_cart_pose(side)
        print(f"  model flange in base_link (m): {np.round(position, 4).tolist()}")
        print(f"  model flange in shoulder (m):  {np.round(shoulder, 4).tolist()}")
        if cart is None:
            print("  controller flange pose: unavailable")
            continue
        # SDK order is [x, y, z, qx, qy, qz, qw], same quaternion order as the model's.
        print(f"  controller flange pose (m):    {np.round(cart[:3], 4).tolist()}")
        print(f"  -> distance vs base_link frame: {np.linalg.norm(cart[:3] - position) * 1000:.1f} mm, "
              f"vs shoulder frame: {np.linalg.norm(cart[:3] - shoulder) * 1000:.1f} mm, "
              f"orientation diff {_quat_angle_deg(cart[3:], quat):.1f} deg")
    print("A few mm in one of the frames confirms the URDF joint conventions match the controller.")
    print("=" * 72)
    print()


def _viewer_qpos(model, data, output, sides_moving, show_measured):
    """Copy for the display: measured arm joints on hardware, plus mirrored gripper fingers."""
    qpos = data.qpos.copy()
    for side in SIDES:
        suffix = side.upper()
        if show_measured or side not in sides_moving:
            for name, value in zip(_SPEC.arm_joint_names(side), output.read_arm_positions(side)):
                qpos[model.jnt_qposadr[model.joint(name).id]] = value
        driven = qpos[model.jnt_qposadr[model.joint(f"gripper_right_joint-{suffix}").id]]
        qpos[model.jnt_qposadr[model.joint(f"gripper_left_joint-{suffix}").id]] = driven
    return qpos


def _motion_line(model, fk_data, controller, output, side, ee_start) -> str:
    """Hand vs IK target vs measured gripper translation since teleop start (cm, base axes)."""
    import mujoco

    def cm(vector):
        return "[" + ", ".join(f"{v * 100:+5.1f}" for v in vector) + "]"

    tracker = controller.tracker
    hand = tracker.raw_residual if tracker.raw_residual is not None else np.zeros(3)
    fk_data.qpos[:] = controller.data.qpos
    fk_data.qpos[controller.arm_dof_indices] = output.read_arm_positions(side)
    mujoco.mj_kinematics(model, fk_data)
    measured = fk_data.xpos[controller.ee_body_id] - ee_start
    rotation = tracker.euler_residual or (0.0, 0.0, 0.0)
    return (f"{side[0].upper()} hand {cm(hand)}  target {cm(tracker.target_position - ee_start)}  "
            f"measured {cm(measured)} cm  wrist rot(deg) {np.round(np.degrees(rotation), 1).tolist()}")


def _ramp_to_home(output, sides, duration_s, period_s) -> float:
    """Smoothstep every selected arm from its measured pose to the home pose.

    Streams through the same clamped, rate-limited ``send_joints`` path as
    teleop (the controller rejects the SDK's timed TASK_PVT move), then holds
    the target until the measured joints arrive. Returns the final max error.
    """
    start = {side: output.read_arm_positions(side) for side in sides}
    target = {side: _SPEC.arm_home_qpos(side) for side in sides}
    t0 = time.monotonic()
    settle_deadline = None
    while True:
        tick = time.monotonic()
        u = min(1.0, (tick - t0) / duration_s)
        blend = u * u * (3.0 - 2.0 * u)
        for side in sides:
            output.send_joints(side, start[side] + (target[side] - start[side]) * blend)
            output.check_tracking(side)
        if output.fault:
            raise CVTEError(output.fault)
        if u >= 1.0:
            error = max(float(np.max(np.abs(output.read_arm_positions(s) - target[s]))) for s in sides)
            if error <= HOME_TOLERANCE_RAD:
                return error
            settle_deadline = settle_deadline or tick + HOME_SETTLE_TIMEOUT_S
            if tick > settle_deadline:
                raise CVTEError(f"Arms still {np.degrees(error):.1f} deg from the startup pose; stopping")
        remaining = period_s - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def run(args, *, output_factory=CVTERealOutput, input_factory=None, viewer_factory=None) -> int:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SPEC.scene_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    # TeleopArmController reads base_link's xmat at construction to rotate hand
    # translations into the base frame. Before any forward pass xmat is all
    # zeros, which silently zeroes every translation (rotation still works).
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    sides = SIDES if args.arm == "both" else (args.arm,)

    output = output_factory(
        args.ip,
        sides=sides,
        read_only=args.read_only,
        feedback_timeout_s=args.feedback_timeout,
        joint_margin_rad=args.joint_margin,
    )
    input_stream = None
    viewer = None
    prev_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        print(f"Connecting to the D3 controller at {args.ip} ...", flush=True)
        output.connect()
        if not output.wait_for_feedback(args.state_timeout):
            print("ERROR: no joint feedback from both arms. Check the network and the web UI "
                  f"(http://{args.ip}).")
            return 1
        output.load_limits_from_model(model, {side: _SPEC.arm_joint_names(side) for side in SIDES})

        gripper_sides = () if args.no_gripper else tuple(s for s in sides if output.gripper_available(s))
        arms = {side: _make_controller(model, data, ik_data, side, args) for side in SIDES}
        _sync_measured(model, data, arms, output)
        sim_gripper_range = {
            side: model.jnt_range[model.joint(_SPEC.hand_joint_names(side)[0]).id].copy() for side in SIDES
        }

        if args.report:
            _print_report(model, data, output, gripper_sides)

        problems = output.check_startup_state()
        for problem in problems:
            print(f"ERROR: {problem}")
        if problems:
            print("Refusing to start. Use the web UI (复位) and re-run with --read-only --report.")
            return 1

        if not args.read_only:
            print("Enabling motion (clear errors, servo on, POSITION mode) ...", flush=True)
            output.enable_motion()
            if args.home:
                print(f"Moving {', '.join(sides)} to the startup pose over {args.home_duration:.1f}s ...", flush=True)
                error = _ramp_to_home(output, sides, args.home_duration, 1.0 / args.publish_hz)
                print(f"  At the startup pose (max joint error {np.degrees(error):.2f} deg).", flush=True)
                # Never build the IK baseline from the pre-move or assumed pose.
                _sync_measured(model, data, arms, output)
                if args.home_only:
                    for side in sides:
                        print(f"  {side} at (deg): {np.round(np.degrees(output.read_arm_positions(side)), 2).tolist()}")
                    print("Homed. Exiting without teleoperation (--home-only).")
                    return 0

        if args.viewer:
            from utils.core.process_viewer import ProcessViewer

            viewer = (viewer_factory or ProcessViewer)(
                _SPEC.scene_path, _viewer_qpos(model, data, output, sides, not args.read_only)
            )

        if input_factory is None:
            from utils.core.input_stream import TeleopInputStream

            input_factory = TeleopInputStream
        input_stream = input_factory(args)

        mode = "READ-ONLY (no commands are sent)" if args.read_only else "MOTION ENABLED"
        print(f"{mode}; arm={args.arm}; grippers={list(gripper_sides) or 'off'}", flush=True)
        print("First wrist frame aligns to the current pose. Left fist 3s (AVP) or Ctrl+C stops.", flush=True)

        # Gripper position at teleop start, for the hand/target/measured log.
        ee_start = {side: data.xpos[arms[side].ee_body_id].copy() for side in sides}
        fk_data = mujoco.MjData(model)
        period = 1.0 / args.publish_hz
        start = time.monotonic()
        last_valid_packet = 0.0
        last_log = start
        while True:
            loop_start = time.monotonic()
            if args.duration and loop_start - start >= args.duration:
                break
            if viewer is not None and not viewer.is_running():
                break

            poll = input_stream.poll_arms(
                arms["left"], arms["right"], _quest3_pose_converter, allow_stop_gesture=True
            )
            if poll.stop_requested:
                print("\nStop gesture detected.", flush=True)
                break
            if poll.saw_valid_data:
                last_valid_packet = loop_start
            if output.fault:
                raise CVTEError(output.fault)

            for side in SIDES:
                measured = output.read_arm_positions(side)  # Raises on stale feedback.
                if side not in sides:
                    data.qpos[arms[side].arm_dof_indices] = measured

            for side in sides:
                controller = arms[side]
                solution = controller.step_smoothed_ik(last_valid_packet, loop_start, PACKET_TIMEOUT_S)
                if solution is None:
                    continue  # No fresh tracking: send nothing; the controller holds.
                controller.write_solution_to_data(solution, include_hand=True)
                if args.read_only:
                    continue
                gripper = None
                if side in gripper_sides and controller.latest_hand_qpos is not None:
                    gripper = sim_gripper_to_opening(
                        controller.latest_hand_qpos[0], sim_gripper_range[side], output.gripper_range(side)
                    )
                output.send_joints(side, solution[controller.arm_dof_indices], gripper_m=gripper)

            if not args.read_only:
                for side in sides:
                    output.check_tracking(side)
            mujoco.mj_forward(model, data)

            if loop_start - last_log > 1.0:
                stale = not last_valid_packet or loop_start - last_valid_packet > PACKET_TIMEOUT_S
                print(f"cmd#{output.commands_sent} vr_stale={stale}", flush=True)
                for side in sides:
                    print("  " + _motion_line(model, fk_data, arms[side], output, side, ee_start[side]), flush=True)
                    command = output.last_command(side)
                    if command is not None:
                        lag = np.degrees(command - output.read_arm_positions(side))
                        print(f"    {side[0].upper()} command - measured J1..J7 (deg): "
                              f"{np.round(lag, 1).tolist()}", flush=True)
                last_log = loop_start
            if viewer is not None:
                viewer.publish(_viewer_qpos(model, data, output, sides, not args.read_only))

            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by operator.", flush=True)
    finally:
        try:
            output.close()  # Stops the tracking task; servos stay enabled (no brakes).
        finally:
            try:
                if input_stream is not None:
                    input_stream.close()
            finally:
                if viewer is not None:
                    viewer.close()
                signal.signal(signal.SIGTERM, prev_sigterm)
    print(f"Session ended; commands sent={output.commands_sent}.")
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default=_SPEC.real_ip, help="D3 controller address")
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--read-only", action="store_true",
                        help="Connect and solve IK but never enable servos or send a command")
    parser.add_argument("--report", action="store_true",
                        help="Print controller state and an FK cross-check, then continue")
    parser.add_argument("--home", action="store_true",
                        help="Smoothly ramp to the forearms-forward pose before teleop "
                             "(default: teleoperate from the current pose)")
    parser.add_argument("--home-only", action="store_true",
                        help="Move to the startup pose and exit without VR. Implies --home")
    parser.add_argument("--home-duration", type=float, default=6.0, help="Seconds for the --home move")
    parser.add_argument("--no-gripper", action="store_true", help="Never command the grippers")
    parser.add_argument("--viewer", action="store_true",
                        help="Display-only MuJoCo viewer (measured joints; IK target in --read-only)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Exit automatically after this many seconds (0 = run until stopped)")

    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--joint-margin", type=float, default=0.01,
                        help="Keep commands this far inside the joint limits (radians)")
    parser.add_argument("--feedback-timeout", type=float, default=0.5,
                        help="Seconds without joint feedback before the session stops")
    parser.add_argument("--state-timeout", type=float, default=5.0,
                        help="Seconds to wait for the first joint feedback")

    parser.add_argument("--position-scale", type=float, default=POSITION_SCALE)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--rot-weight", type=float, default=ROT_WEIGHT)
    parser.add_argument("--ik-damping", type=float, default=IK_DAMPING)
    parser.add_argument("--ik-current-weight", type=float, default=IK_CURRENT_WEIGHT)
    parser.add_argument("--position-deadband", type=float, default=POSITION_DEADBAND,
                        help="Ignore hand translations smaller than this (m); 0 = off, as in simulation")
    parser.add_argument("--rotation-deadband", type=float, default=ROTATION_DEADBAND_DEG,
                        help="Ignore wrist rotations smaller than this (deg); 0 = off, as in simulation")
    parser.add_argument("--pos-gain", type=float, default=POS_GAIN,
                        help="Per-tick fraction of the way to the translation target; 1 = no smoothing")
    parser.add_argument("--rot-gain", type=float, default=ROT_GAIN,
                        help="Per-tick fraction of the way to the rotation target; 1 = no smoothing")

    parser.add_argument("--input-source", default="avp", choices=["avp", "pico4", "quest3"])
    parser.add_argument("--avp-ip", default="192.168.5.32")
    parser.add_argument("--port", type=int, default=9000, help="Quest 3 UDP port")
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"])
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-relay-host", default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    args = parser.parse_args(argv)
    if args.home_only:
        args.home = True
    if args.read_only and args.home:
        parser.error("--home/--home-only move the arms and cannot be combined with --read-only")
    for name in ("publish_hz", "feedback_timeout", "state_timeout", "home_duration"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("position_deadband", "rotation_deadband"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("pos_gain", "rot_gain"):
        if not 0 < getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        LOG.error("Session aborted: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

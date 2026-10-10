#!/usr/bin/env python3
"""Teleoperate a real DexForce W1 (dual arm + dexterous hands) from VR.

Pipeline: AVP wrist tracking -> MuJoCo pose IK -> /control/joint_position,
with AnyDex hand retargeting -> /control/ee/left|right when --enable-hands is set.

The arms teleoperate from wherever they currently are; --home performs the
startup motion to the configured pose first.

Nothing is published in --read-only mode: it installs no publishers at all.

The verified hand on the demonstrator robot is a Linker L6 (6 joints), while the
hand mapping here targets the L20 (16 joints). The arm path is unaffected; leave
--enable-hands off until the mapping matches the installed hand.

Run it under Python 3.10 with ROS 2 sourced, e.g.:

    conda activate vr_teleop
    source /opt/ros/humble/setup.bash
    source ~/dexforce_msgs_ws/install/setup.bash
    export ROS_DOMAIN_ID=20 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI='...'

    python example/teleop_dexforce_real.py --avp-ip 192.168.5.32
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"))

import numpy as np

from utils.adapters.output.dexforce_ros import DexForceFeedbackError, DexForceRealOutput
from utils.adapters.output.linker_l20_mapping import (
    load_sim_hand_limits,
    open_hand_percent,
    retarget_indices,
    retarget_to_dexforce_percent,
)
from utils.core.arm_controller import TeleopArmController
from utils.core.input_stream import TeleopInputStream
from utils.core.quaternion import transform_vr_to_robot_pose
from utils.robots.dexforce import DexForceSpec

_SPEC = DexForceSpec()

CONTROL_PERIOD_S = 0.02
PACKET_TIMEOUT_S = 0.25
POSITION_SCALE = 3.0
EMA_ALPHA = 0.8
ROT_WEIGHT = 1.0
IK_DAMPING = 1e-3
IK_CURRENT_WEIGHT = 0.1
POS_GAIN = 0.08
ROT_GAIN = 0.08
POSITION_DEADBAND = 0.03
ROTATION_DEADBAND_DEG = 8.0
HOME_SETTLE_TIMEOUT_S = 3.0
HOME_TOLERANCE_RAD = 0.05


def _joint_qpos_addresses(model, names) -> dict[str, int]:
    import mujoco

    addresses = {}
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found in scene {_SPEC.scene_path}")
        addresses[name] = int(model.jnt_qposadr[joint_id])
    return addresses


def _write_body_qpos(model, data, addresses, body_qpos) -> None:
    import mujoco

    for name, value in zip(_SPEC.real_body_joint_names(), body_qpos):
        data.qpos[addresses[name]] = float(value)
    mujoco.mj_forward(model, data)


def _make_controller(model, data, ik_data, side, args, home_qpos):
    hand_config = args.hand_config or str(_SPEC.hand_config_path(args.input_source, side))
    return TeleopArmController(
        model,
        data,
        ik_data,
        side,
        args,
        hand_config=hand_config,
        hand_type=_SPEC.hand_type,
        robot_label="DexForce",
        home_qpos=home_qpos,
        arm_joint_names=_SPEC.arm_joint_names(side),
        hand_joint_names=_SPEC.hand_joint_names(side),
        ee_body_name=_SPEC.ee_body_name(side),
        base_body_name=_SPEC.base_body_name,
        hand_qpos_mapping=_SPEC.hand_qpos_mapping,
        position_deadband=POSITION_DEADBAND,
        rotation_deadband_deg=ROTATION_DEADBAND_DEG,
        pos_gain=POS_GAIN,
        rot_gain=ROT_GAIN,
    )


def _wait_for_state(output: DexForceRealOutput, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        output.spin_once(timeout_sec=0.1)
        if output.has_state():
            return True
    return False


def _print_read_only_report(output: DexForceRealOutput, spec: DexForceSpec) -> None:
    state = output._latest_state or {}
    print()
    print("=" * 68)
    print("READ-ONLY REPORT (nothing has been published to the robot)")
    print("=" * 68)
    print(f"  server status : {output.server_status()!r}")
    print(f"  run_mode      : {output.run_mode()!r}   (must be 'Position' to command)")
    names = state.get("joint_name") or []
    print(f"  reported joints ({len(names)}):")
    for i in range(0, len(names), 4):
        print("     " + "  ".join(f"{n:<10}" for n in names[i : i + 4]))
    try:
        positions = output.read_joint_positions()
    except DexForceFeedbackError as exc:
        print(f"  joint positions unavailable: {exc}")
        positions = {}
    print("  measured angles (rad):")
    for group in (spec.arm_joint_names("right"), spec.arm_joint_names("left"),
                  spec.real_body_joint_names()):
        for name in group:
            if name in positions:
                print(f"     {name:<12} {positions[name]: .4f}")
    for side in ("left", "right"):
        hand = output.hand_state(side)
        if hand is None:
            print(f"  {side} hand: no /feedback/ee/{side} yet")
            continue
        joints = list(getattr(hand, "joint_states", []))
        print(f"  {side} hand: ee_name={getattr(hand, 'ee_name', '?')!r} status={getattr(hand, 'status', '?')} "
              f"({len(joints)} joints)")
        for joint in joints:
            print(f"     {joint.name:<16} pos={joint.position:7.2f} state={joint.status}")
    print("=" * 68)
    print()


def _spin_until(output, deadline: float) -> None:
    """Process robot feedback until ``deadline`` instead of sleeping through it.

    rclpy only runs subscription callbacks inside spin, so a bare sleep leaves
    the cached robot state frozen at whatever arrived before it.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if output.node is None:
            time.sleep(remaining)
            return
        output.spin_once(timeout_sec=remaining)


def _ramp_to_home(output, current_command, target_command, duration_s, publish_period_s) -> None:
    """Smoothstep from the measured pose to the startup pose."""
    if duration_s <= 0.0:
        output.send_command(target_command)
        return
    steps = max(1, int(duration_s / publish_period_s))
    start_time = time.monotonic()
    for step in range(1, steps + 1):
        u = step / steps
        blend = u * u * (3.0 - 2.0 * u)  # smoothstep
        output.send_command(current_command + (target_command - current_command) * blend)
        _spin_until(output, start_time + step * publish_period_s)
    print(f"  Startup ramp commanded in {time.monotonic() - start_time:.1f}s", flush=True)


def _wait_until_reached(output, target_command, sides, timeout_s, tolerance_rad, publish_period_s) -> float:
    """Hold the target while reading fresh feedback until the arms get there.

    Returns the largest measured arm-joint error in radians; it exceeds
    ``tolerance_rad`` when the timeout ran out first.
    """
    target = output.clamp_command(target_command)
    indices = {side: [output.joint_index[n] for n in _SPEC.arm_joint_names(side)] for side in sides}
    start_time = time.monotonic()
    tick = start_time
    while True:
        output.send_command(target_command)
        tick += publish_period_s
        _spin_until(output, tick)
        error = max(
            float(np.max(np.abs(output.read_arm_positions(side) - target[idx])))
            for side, idx in indices.items()
        )
        if error <= tolerance_rad or time.monotonic() - start_time >= timeout_s:
            return error


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def run(args) -> int:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SPEC.scene_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    output = DexForceRealOutput(
        _SPEC,
        read_only=args.read_only,
        publish_hz=args.publish_hz,
        joint_margin_rad=args.joint_margin,
        watchdog_timeout_s=args.watchdog_timeout,
    )
    input_stream = None
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    try:
        output.connect()
        output.load_limits_from_model(model)

        print("Waiting for /feedback/robot_server_state ...", flush=True)
        if not _wait_for_state(output, args.state_timeout):
            print("ERROR: no robot feedback received.")
            print("Check: robot powered and enabled, ROS_DOMAIN_ID, RMW_IMPLEMENTATION and "
                  "CYCLONEDDS_INTERFACE match the robot, and the driver is running.")
            return 1

        if args.report:
            _print_read_only_report(output, _SPEC)

        problems = output.check_startup_state()
        for problem in problems:
            print(f"ERROR: {problem}")
        if problems:
            print("Refusing to start. Re-run with --read-only to inspect the robot.")
            return 1

        # Mirror the real body and arms into the model so IK shares the robot's
        # actual base pose. The arms hang off the torso, so this matters.
        body_addresses = _joint_qpos_addresses(model, _SPEC.real_body_joint_names())
        _write_body_qpos(model, data, body_addresses, output.read_body_positions())

        arms = {}
        for side in ("left", "right"):
            controller = _make_controller(model, data, ik_data, side, args, _SPEC.real_home_qpos(side))
            controller.sync_from_joint_positions(output.read_arm_positions(side))
            arms[side] = controller
        mujoco.mj_forward(model, data)

        hand_limits = {}
        hand_indices = {}
        for side in ("left", "right"):
            hand_limits[side] = load_sim_hand_limits(model, _SPEC, side)
            if arms[side].hand_retargeter is not None and arms[side].hand_retargeter.available:
                raw_names = getattr(arms[side].hand_retargeter._retargeter.optimizer.robot,
                                    "dof_joint_names", None)
                if raw_names is None:
                    raise RuntimeError("Retargeter does not expose dof_joint_names")
                hand_indices[side] = retarget_indices(raw_names)

        # The arms teleoperate from wherever they currently are. A startup move
        # to the configured pose is opt-in (--home): with the robot already in a
        # sane pose there is no reason to move it before the operator is ready.
        body_hold = output.read_body_positions()
        start_command = output.build_command(
            {side: output.read_arm_positions(side) for side in arms},
            body_hold,
            body_from_feedback=False,
        )
        if args.home:
            print(f"Moving arms to the configured startup pose over {args.home_duration}s ...",
                  flush=True)
            target = output.build_command(
                {side: _SPEC.real_home_qpos(side) for side in arms},
                body_hold,
                body_from_feedback=False,
            )
            _ramp_to_home(output, start_command, target, args.home_duration, 1.0 / args.publish_hz)
            start_command = target
            if not args.read_only:
                error = _wait_until_reached(output, target, arms, HOME_SETTLE_TIMEOUT_S,
                                            HOME_TOLERANCE_RAD, 1.0 / args.publish_hz)
                if error <= HOME_TOLERANCE_RAD:
                    print(f"  Arms at the startup pose (max joint error {error:.3f} rad).", flush=True)
                else:
                    # Hold where the arms are, so that hold and IK baseline agree.
                    print(f"WARNING: arms still {error:.3f} rad from the startup pose after "
                          f"{HOME_SETTLE_TIMEOUT_S:.0f}s; teleoperating from where they are.",
                          flush=True)
                    start_command = output.build_command(
                        {side: output.read_arm_positions(side) for side in arms},
                        body_hold,
                        body_from_feedback=False,
                    )
            # Re-read after the startup move: never build the IK baseline from
            # the pre-move pose or assume the commanded pose was reached.
            _write_body_qpos(model, data, body_addresses, output.read_body_positions())
            for side, controller in arms.items():
                controller.sync_from_joint_positions(output.read_arm_positions(side))
            mujoco.mj_forward(model, data)
            if args.home_only:
                # Park at the startup pose without opening a VR connection, so
                # homing can be done as its own step before any teleoperation.
                for side in arms:
                    reached = output.read_arm_positions(side)
                    print(f"  {side} at: {np.round(reached, 4).tolist()}", flush=True)
                print("Homed. Exiting without teleoperation (--home-only).", flush=True)
                return 0
        else:
            print("Starting from the arms' current pose (no startup motion).", flush=True)

        hold_command = start_command.copy()

        if args.enable_hands:
            for side in arms:
                output.send_hand(side, open_hand_percent())
            print("Hands enabled; sending retargeted finger targets.", flush=True)

        input_stream = TeleopInputStream(args)
        publish_period = 1.0 / args.publish_hz
        last_publish = 0.0
        last_log = time.monotonic()
        last_valid_packet = 0.0

        print("Teleoperation loop running.", flush=True)
        print("Left fist for 3s, or Ctrl+C, to stop.", flush=True)
        while True:
            loop_start = time.monotonic()

            poll = input_stream.poll_arms(
                arms["left"], arms["right"], transform_vr_to_robot_pose, allow_stop_gesture=True
            )
            if poll.stop_requested:
                print("\nStop gesture detected.", flush=True)
                break
            if poll.saw_valid_data:
                last_valid_packet = loop_start

            # Keep the model body pose tracking the real robot.
            try:
                _write_body_qpos(model, data, body_addresses, output.read_body_positions())
            except DexForceFeedbackError:
                pass

            stale = (loop_start - last_valid_packet) > PACKET_TIMEOUT_S if last_valid_packet else True

            if args.read_only:
                for side, controller in arms.items():
                    solution = controller.step_smoothed_ik(last_valid_packet, loop_start, PACKET_TIMEOUT_S)
                    if solution is not None:
                        controller.write_solution_to_data(solution, include_hand=False)
                mujoco.mj_forward(model, data)
                if loop_start - last_log > 1.0:
                    residuals = " ".join(
                        f"{s[0].upper()}:{arms[s].latest_residual}" for s in ("left", "right")
                    )
                    print(f"[read-only] {residuals} stale={stale}", flush=True)
                    last_log = loop_start
                continue

            if loop_start - last_publish >= publish_period:
                last_publish = loop_start
                arm_targets = {}
                hand_targets = {}
                for side, controller in arms.items():
                    solution = controller.step_smoothed_ik(
                        last_valid_packet, loop_start, PACKET_TIMEOUT_S
                    )
                    if solution is None:
                        # No fresh VR data for this arm: keep commanding the last
                        # pose we drove to. Falling back to the startup pose would
                        # snap the arms across the workspace on every dropout.
                        arm_targets[side] = hold_command[
                            [output.joint_index[n] for n in _SPEC.arm_joint_names(side)]
                        ]
                    else:
                        arm_targets[side] = solution[controller.arm_dof_indices]
                        controller.write_solution_to_data(solution, include_hand=False)

                    if args.enable_hands and side in hand_indices:
                        raw = controller.latest_hand_qpos
                        if raw is not None:
                            hand_targets[side] = retarget_to_dexforce_percent(
                                raw, hand_limits[side], indices=hand_indices[side]
                            )

                command = output.build_command(arm_targets, body_hold, body_from_feedback=True)
                try:
                    output.send_command(command)
                except ValueError as exc:
                    print(f"Warning: rejecting command ({exc})")
                    continue
                # Advance the dropout fallback only after a command goes out, so
                # the held pose is always one we actually commanded.
                hold_command = command.copy()

                for side, values in hand_targets.items():
                    output.send_hand(side, values)

                if loop_start - last_log > 1.0:
                    print(
                        f"L resid:{arms['left'].latest_residual} "
                        f"R resid:{arms['right'].latest_residual} "
                        f"cmd#{output.commands_sent} stale={stale}",
                        flush=True,
                    )
                    last_log = loop_start

            output.spin_once(timeout_sec=0.0)
            remaining = CONTROL_PERIOD_S - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopping by operator.", flush=True)
    finally:
        if input_stream is not None:
            input_stream.close()
        if args.enable_hands and not args.read_only and output.node is not None:
            for side in ("left", "right"):
                try:
                    output.send_hand(side, open_hand_percent())
                except Exception:
                    pass
        output.close()
        signal.signal(signal.SIGTERM, prev_sigterm)
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--read-only", action="store_true",
                        help="Subscribe and solve IK but never publish a command")
    parser.add_argument("--report", action="store_true",
                        help="Print joints, angles, run mode and hand state, then continue")
    parser.add_argument("--enable-hands", action="store_true",
                        help="Also drive the Linker L20 hands on /control/ee/*")
    parser.add_argument("--home", action="store_true",
                        help="Move to the configured startup pose before teleop "
                             "(default: teleoperate from the arms' current pose)")
    parser.add_argument("--home-only", action="store_true",
                        help="Move to the startup pose and exit, without connecting to VR "
                             "or teleoperating. Implies --home")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Exit automatically after this many seconds (0 = run until stopped)")

    parser.add_argument("--publish-hz", type=float, default=50.0,
                        help="Command publish rate (the vendor controller uses 90 Hz internally)")
    parser.add_argument("--joint-margin", type=float, default=0.05,
                        help="Keep commands this far inside the model joint limits (radians)")
    parser.add_argument("--watchdog-timeout", type=float, default=0.5,
                        help="Seconds without robot feedback before the loop reports a stall")
    parser.add_argument("--state-timeout", type=float, default=10.0,
                        help="Seconds to wait for the first robot feedback message")
    parser.add_argument("--home-duration", type=float, default=3.0,
                        help="Seconds for the smoothstep startup motion")

    parser.add_argument("--position-scale", type=float, default=POSITION_SCALE)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--rot-weight", type=float, default=ROT_WEIGHT)
    parser.add_argument("--ik-damping", type=float, default=IK_DAMPING)
    parser.add_argument("--ik-current-weight", type=float, default=IK_CURRENT_WEIGHT)
    parser.add_argument("--hand-config", default=None, help="Override the AnyDex retargeter YAML")

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
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

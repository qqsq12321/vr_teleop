#!/usr/bin/env python3
"""Pico -> RealMan RM75-B + Inspire, legacy Python SDK (no ROS).

Default: offline dry-run. --check performs a socket-free model/config check.
--read-only connects but sends no actuator commands. --enable-motion allows hardware writes without a prompt; never homes, enables power, or reconnects on a fault.
"""
from __future__ import annotations

import argparse
import ipaddress
import logging
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "AnyDexRetarget"))

import mujoco
import numpy as np
import yaml

from utils.adapters.output.inspire_mapping import raw_qpos_to_channels
from utils.adapters.output.realman_real import ArmFeedback, RealManOutput
from utils.core.realman_arm_controller import RealManArmController
from utils.core.realman_input_stream import RealManInputStream
from utils.core.realman_collision import RealManCollisionGuard
from utils.core.process_viewer import ProcessViewer
from utils.core.safety import InputTrackingFault, SafetyFault, SafetyLimits, SafetySupervisor
from utils.robots.realman_rm75b import RealManRM75BInspireSpec

SPEC = RealManRM75BInspireSpec()
LOG = logging.getLogger(__name__)


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    expected = {
        "network": {"left_ip", "right_ip"},
        "input": {"mode", "relay_host", "relay_port", "direct_port", "broadcast_port"},
        "control": {"position_scale", "ema_alpha", "rot_weight", "ik_damping", "ik_current_weight", "pos_gain", "rot_gain", "follow"},
        "safety": set(SafetyLimits.__dataclass_fields__),
        "hands": {"backend", "rate_hz", "baudrate", "timeout_100ms", "address", "device_id"},
    }
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError("Configuration sections must be network/input/control/safety/hands")
    for section, keys in expected.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"Incorrect or missing keys in {section}; expected {sorted(keys)}")
    for value in config["network"].values():
        ipaddress.IPv4Address(value)
    if len(set(config["network"].values())) != 2:
        raise ValueError("Arm IPs must differ")
    inp, ctrl, hand = config["input"], config["control"], config["hands"]
    if inp["mode"] not in {"relay", "direct"} or not isinstance(inp["relay_host"], str) or not inp["relay_host"]:
        raise ValueError("Invalid Pico input configuration")
    for key in ("relay_port", "direct_port", "broadcast_port"):
        if type(inp[key]) is not int or not 1 <= inp[key] <= 65535:
            raise ValueError(f"Invalid {key}")
    if type(ctrl["follow"]) is not bool:
        raise ValueError("control.follow must be boolean")
    for key, value in ctrl.items():
        if key == "follow":
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"control.{key} must be finite and positive")
    for key in ("ema_alpha", "pos_gain", "rot_gain"):
        if ctrl[key] > 1:
            raise ValueError(f"control.{key} must be <= 1")
    limits = SafetyLimits(**config["safety"])
    if hand["backend"] not in {"native", "modbus"}:
        raise ValueError("hands.backend must be native or modbus")
    if type(hand["rate_hz"]) not in (int, float) or not math.isfinite(hand["rate_hz"]) or not 0 < hand["rate_hz"] <= limits.rate_hz:
        raise ValueError("Hand rate must be positive and <= arm rate")
    for key, low, high in (("baudrate", 9600, 115200), ("timeout_100ms", 1, 10),
                           ("address", 0, 65530), ("device_id", 1, 247)):
        if type(hand[key]) is not int or not low <= hand[key] <= high:
            raise ValueError(f"Invalid hands.{key}")
    return config, limits


def load_model():
    model = mujoco.MjModel.from_xml_path(str(SPEC.scene_path))
    data = mujoco.MjData(model)
    addresses, ranges = {}, {}
    for side in ("left", "right"):
        ids = [model.joint(name).id for name in SPEC.arm_joint_names(side)]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j] for j in ids):
            raise ValueError("Expected seven bounded hinge joints per arm")
        addresses[side] = model.jnt_qposadr[ids].astype(int)
        ranges[side] = model.jnt_range[ids].copy()
        data.qpos[addresses[side]] = SPEC.arm_home_qpos(side)  # Offline only; replaced before hardware control.
    mujoco.mj_forward(model, data)
    return model, data, addresses, ranges


def _feedback_from_model(data, addresses):
    now = time.monotonic()
    return {s: ArmFeedback(data.qpos[ids].copy(), now) for s, ids in addresses.items()}


def run_read_only(args, config, *, output_factory=RealManOutput):
    output = output_factory(**config["network"])
    try:
        states = output.connect()
        start = time.monotonic()
        while True:
            for side, state in states.items():
                print(f"{side}: deg={np.round(np.rad2deg(state.q_rad), 3).tolist()}", flush=True)
            if args.duration and time.monotonic() - start >= args.duration:
                break
            time.sleep(0.5)
            states = output.read_feedback()
    finally:
        output.close()  # Read-only close does NOT issue Move_Stop_Cmd.


def run(args, *, output_factory=RealManOutput, input_factory=RealManInputStream,
        viewer_factory=ProcessViewer):
    pause_tracking = getattr(args, "pause_on_tracking_loss", False)
    if pause_tracking and (args.enable_motion or args.read_only or args.check):
        raise ValueError("--pause-on-tracking-loss is only allowed in offline dry-run")
    config, limits = load_config(args.config)
    if args.read_only:
        return run_read_only(args, config, output_factory=output_factory)
    model, data, addresses, ranges = load_model()
    collision = RealManCollisionGuard(SPEC.scene_path, margin=limits.collision_margin_m)
    sides = ("left", "right") if args.arm == "both" else (args.arm,)
    safety = SafetySupervisor(limits, ranges, sides, hands=args.enable_hands,
                              recover_tracking=args.enable_motion)
    if args.check:
        states = _feedback_from_model(data, addresses)
        safety.observe_feedback(states, time.monotonic())
        collision.check(data.qpos)
        print(f"CHECK OK: RM75 7+7 joints; {len(collision.pairs)} explicit collision pairs; config valid.")
        print("Offline simulation pose only. No SDK loaded, no sockets opened, no hardware verified.")
        return
    inp = config["input"]
    ctrl_args = SimpleNamespace(**config["control"], input_source="pico4",
        pico4_mode=inp["mode"], pico4_relay_host=inp["relay_host"],
        pico4_relay_port=inp["relay_port"], pico4_port=inp["direct_port"],
        pico4_broadcast_port=inp["broadcast_port"])
    output, stream, viewer = None, None, None
    running = False
    tracking_paused = False
    command_count = 0
    try:
        if args.enable_motion:
            # Heavy scene loading happens before any SDK connection.
            output = output_factory(**config["network"])
            states = output.connect()
            safety.observe_feedback(states, time.monotonic())
            for side in addresses:
                data.qpos[addresses[side]] = states[side].q_rad
            mujoco.mj_forward(model, data)
            collision.check(data.qpos)  # Never validate only the simulation home.
        arms = {}
        for side in ("left", "right"):
            hand_config = str(SPEC.hand_config_path("pico4", side)) if args.enable_hands and side in sides else None
            arms[side] = RealManArmController(model, data, mujoco.MjData(model), side, ctrl_args,
                hand_config=hand_config, hand_type=SPEC.hand_type,
                home_qpos=data.qpos[addresses[side]].copy(), robot_label="RealMan RM75-B",
                arm_joint_names=SPEC.arm_joint_names(side), hand_joint_names=SPEC.hand_joint_names(side),
                ee_body_name=SPEC.ee_body_name(side), base_body_name=SPEC.base_body_name,
                hand_qpos_mapping=SPEC.hand_qpos_mapping, position_deadband=0.003,
                rotation_deadband_deg=2.0, pos_gain=ctrl_args.pos_gain, rot_gain=ctrl_args.rot_gain)
            if hand_config and not arms[side].hand_retargeter.available:
                raise SafetyFault(f"{side}: hand retargeter unavailable")
        if args.viewer:
            viewer = viewer_factory(SPEC.scene_path, data.qpos)
        if output:
            # Refresh feedback after controller/model initialization.
            states = output.read_feedback()
            safety.observe_feedback(states, time.monotonic())
            for side in addresses:
                data.qpos[addresses[side]] = states[side].q_rad
            collision.check(data.qpos)
            output.enable_motion()
        stream = input_factory(ctrl_args)
        print(f"{'MOTION ENABLED' if output else 'DRY RUN (NO ROBOT CONNECTION)'}; arm={args.arm}; hands={args.enable_hands}")
        print("Waiting for fresh tracking. First frame aligns to current measured pose; Ctrl+C exits.")
        if pause_tracking:
            print("OFFLINE PREVIEW: tracking loss holds the model; fresh tracking re-aligns before resuming.")
        start = time.monotonic()
        last_hand_sent = float('-inf')
        hand_sequences = {}
        previous_hand = {}
        while not args.duration or time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            if viewer and not viewer.is_running():
                break
            if not running:
                states = output.read_feedback() if output else _feedback_from_model(data, addresses)
                safety.observe_feedback(states, time.monotonic())
                for side in addresses:
                    data.qpos[addresses[side]] = states[side].q_rad
                mujoco.mj_forward(model, data)
                for side, arm in arms.items():
                    arm.arm_home_qpos = states[side].q_rad.copy()
                    arm.reset_baseline_from_current_pose(reset_input_reference=True)
            result = stream.poll_arms(arms["left"], arms["right"], None)
            safety.observe_input(result)
            if result.stop_requested:
                raise SafetyFault("Input requested stop")
            now = time.monotonic()
            try:
                if running:
                    # Refresh real feedback even while input is paused. Do not
                    # treat time spent waiting for tracking as SDK feedback loss.
                    if output:
                        states = output.read_feedback()
                        safety.observe_feedback(states, time.monotonic())
                        safety.check_live(time.monotonic())
                    else:
                        safety.check_live(now)
                        states = _feedback_from_model(data, addresses)
                        safety.observe_feedback(states, time.monotonic())
                    if tracking_paused:
                        LOG.info("Fresh tracking restored; continuing command loop.")
                        tracking_paused = False
                    for side in addresses:
                        data.qpos[addresses[side]] = states[side].q_rad
                    mujoco.mj_forward(model, data)
                    measured = data.qpos.copy()
                    previous = measured.copy()
                    for side in sides:
                        previous[addresses[side]] = safety.q_command[side]
                    generated_at = time.monotonic()
                    targets = {}
                    for side in sides:
                        q = arms[side].step_smoothed_ik(safety.wrist_times[side], generated_at, limits.input_timeout_s)
                        if q is None:
                            raise SafetyFault(f"{side}: IK target unavailable")
                        # Shared IK returns the FULL model qpos, not seven arm joints.
                        targets[side] = q[addresses[side]].copy()
                    last_sent = {side: q.copy() for side, q in safety.q_command.items()}
                    commands = safety.validate_targets(targets, time.monotonic())
                    # Feedback error compares against sent commands, not a target
                    # that may be skipped when tracking expires before SDK I/O.
                    safety.q_command = last_sent
                    candidate = measured.copy()
                    for side, q in commands.items():
                        candidate[addresses[side]] = q
                    collision.check_transition(measured, previous, candidate)
                    def guard(at):
                        safety.check_live(at, generated_at=generated_at)
                    guard(time.monotonic())
                    for side in sides:
                        if output:
                            output.send_joints(side, commands[side], guard=guard, follow=config["control"]["follow"])
                        else:
                            data.qpos[addresses[side]] = commands[side]
                        safety.q_command[side] = commands[side].copy()
                    command_count += 1
                    now = time.monotonic()
                    if args.enable_hands and now - last_hand_sent >= 1 / config["hands"]["rate_hz"]:
                        for side in sides:
                            stamp = safety.hand_times[side]
                            if hand_sequences.get(side) == stamp:
                                continue  # Never resend a cached hand sample as a new command.
                            channels = np.array(raw_qpos_to_channels(arms[side].latest_hand_qpos, SPEC))
                            # Only subsequent targets can be slew-limited: the first
                            # physical hand pose is unknown. Operator preflight is mandatory.
                            if side in previous_hand:
                                channels = np.clip(channels, previous_hand[side] - 20, previous_hand[side] + 20)
                            guard(time.monotonic())
                            if output:
                                output.send_hand(side, channels.tolist(), guard=guard)
                            previous_hand[side] = channels
                            hand_sequences[side] = stamp
                        last_hand_sent = time.monotonic()
                    guard(time.monotonic())
                elif safety.input_ready(now) and all(arms[s].tracker.initialized for s in sides):
                    collision.check(data.qpos)
                    safety.start(now)
                    if output and args.enable_hands:
                        output.configure_hands(dict(config["hands"], sides=sides), guard=safety.check_live)
                    running = True
                    print("Tracking aligned; command loop started.")
            except InputTrackingFault as exc:
                if output is not None:
                    # Match master's timeout path: skip new commands, retain
                    # the session and VR reference, resume on valid tracking.
                    if not tracking_paused:
                        LOG.warning("Tracking paused: %s. Skipping arm/hand commands; waiting for fresh data.", exc)
                    tracking_paused = True
                elif pause_tracking:
                    LOG.warning("DRY RUN paused: %s. Hold model; waiting to re-align.", exc)
                    running = False
                    safety = SafetySupervisor(limits, ranges, sides, hands=args.enable_hands)
                    hand_sequences.clear()
                    previous_hand.clear()
                    last_hand_sent = float('-inf')
                else:
                    raise
            if viewer:
                viewer.publish(data.qpos)
            # Match master's RM65 loop: fill the remaining period only.
            # Slow iterations do not incur an extra sleep or queued catch-up work.
            remaining = 1 / limits.rate_hz - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        # Stop transport BEFORE potentially slow viewer/input shutdown.
        try:
            if output:
                output.close()
        finally:
            try:
                if stream:
                    stream.close()
            finally:
                if viewer:
                    viewer.close()
        print(f"Session ended; command cycles={command_count}. No automatic restart.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "realman_real.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: no robot SDK or connection")
    mode.add_argument("--check", action="store_true", help="Offline config/model check and exit; no input sockets")
    mode.add_argument("--read-only", action="store_true", help="Connect both arms and print state; NO actuator writes")
    mode.add_argument("--enable-motion", action="store_true", help="Allow hardware writes without an interactive prompt")
    parser.add_argument("--arm", choices=("left", "right", "both"), default="left")
    parser.add_argument("--enable-hands", action="store_true", help="Explicit opt-in; verify hardware backend and low speed/force first")
    parser.add_argument("--viewer", action="store_true", help="Display-only MuJoCo viewer in a separate process (measured joints on hardware)")
    parser.add_argument("--pause-on-tracking-loss", action="store_true",
                        help="Offline dry-run only: hold on tracking loss and re-align on recovery; other faults still exit")
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 runs until Ctrl+C")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if args.enable_hands and (args.check or args.read_only):
        parser.error("--enable-hands is only valid with dry-run or motion mode")
    if args.pause_on_tracking_loss and (args.enable_motion or args.read_only or args.check):
        parser.error("--pause-on-tracking-loss is only allowed in offline dry-run")
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # SIGTERM follows the same best-effort cleanup as Ctrl+C. SIGKILL/native
    # deadlock cannot be handled; use a verified controller-side watchdog.
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        print("Stopped by operator.")
        return 130
    except Exception as exc:
        LOG.error("Session aborted: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

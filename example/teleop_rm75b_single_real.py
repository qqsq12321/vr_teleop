"""Teleoperate a single floor-mounted RealMan RM75-B arm over RM_API2.

Three modes, safest first:

    --check        Offline. Loads the model and configs, validates the home
                   pose against the joint limits, prints the SDK angles that
                   WOULD be sent, and opens no sockets. This is the default.
    --read-only    Connects and prints measured joint angles. Never enables
                   motion and never sends a motion command.
    --enable-motion  Full teleoperation. The arm WILL move.

Run --read-only first on a new installation to confirm the measured joint
angles look right before the arm is allowed to move.

Examples:
    python example/teleop_rm75b_single_real.py --check
    python example/teleop_rm75b_single_real.py --read-only --duration 10
    python example/teleop_rm75b_single_real.py --enable-motion --input-source pico4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTROL_PERIOD_S = 0.02
PACKET_TIMEOUT_S = 0.25
POSITION_DEADBAND = 0.03
ROTATION_DEADBAND_DEG = 8.0

# Countdown after the homing move completes, before the anchor is taken. The
# daemon streams continuously, so without this the first tracked frame becomes
# the anchor -- typically with a hand still on the keyboard after a restart.
COUNTDOWN_S = 3

DEFAULT_CONFIG = ROOT / "config" / "realman_rm75b_single_real.yaml"
# The lone arm is addressed as the right side throughout the stack.
SIDE = "right"

LOG = logging.getLogger(__name__)


def _transport_guard(_now):
    """No extra host deadline policy on this direct-send path."""


def load_config(config_path):
    """Validate the single-arm config. Rejects a dual-arm file outright."""
    import yaml

    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict) or "network" not in config:
        raise ValueError("Configuration must have a network section")
    network = config["network"]
    if not isinstance(network, dict) or set(network) != {"right_ip"}:
        raise ValueError(
            "Single-arm network must contain exactly right_ip; "
            "config/realman_real.yaml describes the dual-arm robot and is not usable here"
        )
    import ipaddress

    ipaddress.IPv4Address(network["right_ip"])

    for section in ("input", "control"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid {section} section")
    required_control = {
        "position_scale", "ema_alpha", "rot_weight", "ik_damping",
        "ik_current_weight", "pos_gain", "rot_gain", "follow",
    }
    missing = required_control - set(config["control"])
    if missing:
        raise ValueError(f"control section is missing {sorted(missing)}")
    return config


def _load_model(spec):
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(spec.scene_path))
    data = mujoco.MjData(model)
    addresses = np.array(
        [model.jnt_qposadr[model.joint(name).id] for name in spec.arm_joint_names(SIDE)]
    )
    return model, data, addresses


def run_check(args) -> int:
    """Offline validation. Opens no sockets and loads no vendor SDK."""
    import numpy as np

    from utils.robots.realman_rm75b_single import RealManRM75BSingleSpec

    config = load_config(args.config)
    spec = RealManRM75BSingleSpec()
    model, _data, addresses = _load_model(spec)

    home = spec.arm_home_qpos(SIDE)
    print(f"Arm IP (not contacted): {config['network']['right_ip']}")
    print(f"Scene: {spec.scene_path.relative_to(ROOT)}")
    print(f"Model DoF: nq={model.nq} nv={model.nv} nu={model.nu}")

    import mujoco

    ok = True
    print("Home pose per joint:")
    for index, name in enumerate(spec.arm_joint_names(SIDE)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = model.jnt_range[joint_id]
        value = float(home[index])
        inside = lower <= value <= upper
        ok = ok and inside
        marker = "ok" if inside else "OUT OF RANGE"
        print(
            f"  {name}: {value:+.4f} rad ({np.rad2deg(value):+7.2f} deg) "
            f"limits [{np.rad2deg(lower):+7.2f}, {np.rad2deg(upper):+7.2f}] deg  {marker}"
        )

    sdk_degrees = np.rad2deg(home)
    print(f"Would send to the controller (deg): {sdk_degrees.round(2).tolist()}")
    print(f"IK joint addresses: {addresses.tolist()}")

    if not ok:
        print("FAILED: home pose is outside the joint limits", file=sys.stderr)
        return 1
    print("Check passed. No sockets were opened.")
    return 0


def run_read_only(args, *, output_factory=None) -> int:
    """Connect and print measured joint angles. Sends no motion commands."""
    import numpy as np

    from utils.robots.realman_rm75b_single import RealManRM75BSingleSpec

    config = load_config(args.config)
    spec = RealManRM75BSingleSpec()

    if output_factory is None:
        from utils.adapters.output.realman_rm75b_single_real import RM75BSingleOutput

        output_factory = RM75BSingleOutput

    output = output_factory(config["network"]["right_ip"])
    home = spec.arm_home_qpos(SIDE)
    deadline = time.monotonic() + args.duration if args.duration else None
    try:
        feedback = output.connect()
        print("READ-ONLY. Motion is not enabled; no motion command will be sent.", flush=True)
        print(f"Home pose for reference (deg): {np.rad2deg(home).round(2).tolist()}", flush=True)
        while True:
            degrees = np.rad2deg(feedback.q_rad)
            print(f"measured (deg): {degrees.round(2).tolist()}", flush=True)
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.5)
            feedback = output.read_feedback()
    finally:
        # Read-only close must not issue a stop: motion was never enabled.
        output.close()
    return 0


def run_motion(args, *, output_factory=None, input_factory=None, clock=None) -> int:
    """Full teleoperation. The arm moves."""
    sys.path.insert(1, str(ROOT / "third_party" / "AnyDexRetarget"))
    import mujoco
    import numpy as np

    from utils.core.arm_controller import slerp_step
    from utils.core.body.ik import solve_pose_ik
    from utils.core.body.wrist_tracker import WristTracker
    from utils.core.input_stream import TeleopInputStream
    from utils.core.quaternion import matrix_to_quaternion
    from utils.robots.realman_rm75b_single import RealManRM75BSingleSpec

    clock = time if clock is None else clock
    if output_factory is None:
        from utils.adapters.output.realman_rm75b_single_real import RM75BSingleOutput

        output_factory = RM75BSingleOutput

    config = load_config(args.config)
    ctrl, inp = config["control"], config["input"]
    spec = RealManRM75BSingleSpec()
    model, data, addresses = _load_model(spec)

    output, input_stream = None, None
    try:
        output = output_factory(config["network"]["right_ip"])
        output.connect()
        output.enable_motion()

        home = spec.arm_home_qpos(SIDE)
        print(
            f"Moving to the initial pose (deg): "
            f"{output.model_to_sdk_degrees(home).round(2).tolist()}",
            flush=True,
        )
        output.move_to_initial_pose(home, guard=_transport_guard)

        # Re-read AFTER the move. Never assume the commanded pose was reached
        # when establishing the FK/IK baseline.
        feedback = output.read_feedback()
        data.qpos[addresses] = feedback.q_rad
        mujoco.mj_forward(model, data)

        site_id = model.site(f"{SIDE}_tcp").id
        position = data.site_xpos[site_id].copy()
        quaternion = matrix_to_quaternion(data.site_xmat[site_id].reshape(3, 3))
        tracker = WristTracker(
            position, quaternion,
            position_scale=ctrl["position_scale"], ema_alpha=ctrl["ema_alpha"],
            negate_rot_xy=False, base_xmat=None,
            position_deadband=POSITION_DEADBAND,
            rotation_deadband_deg=ROTATION_DEADBAND_DEG,
        )
        arm = _ArmState(
            tracker=tracker, site_id=site_id, ik_data=mujoco.MjData(model),
            cmd_pos=position.copy(), cmd_quat=np.array(quaternion),
            home=data.qpos[addresses].copy(),
        )

        stream_args = argparse.Namespace(
            input_source=args.input_source, avp_ip=args.avp_ip, port=args.port,
            pico4_mode=inp.get("mode", "relay"),
            pico4_relay_host=inp.get("relay_host", "127.0.0.1"),
            pico4_relay_port=inp.get("relay_port", 63902),
            pico4_port=inp.get("direct_port", 63901),
            pico4_broadcast_port=inp.get("broadcast_port", 29888),
        )
        input_stream = (TeleopInputStream if input_factory is None else input_factory)(stream_args)

        print("MOTION ENABLED. Ctrl+C stops.", flush=True)
        print("Homing complete. Get into your start pose.", flush=True)
        _wait_for_tracking(input_stream, arm, clock=clock)
        _countdown()
        # Capture which way the operator is facing right now, so "forward"
        # means forward regardless of where they chose to stand. This also
        # absorbs the arm's own mounting yaw: an operator naturally squares up
        # to the arm, so the two offsets cancel without anyone measuring them.
        yaw = _heading_yaw_rad(input_stream.get_pico4_head_pose())
        if yaw is None:
            print(
                "No headset pose; using the raw VR frame. Stand square to the arm.",
                flush=True,
            )
        else:
            print(f"Operator heading: {np.rad2deg(yaw):+.1f} deg; aligning to it.", flush=True)
        arm.set_heading(yaw)
        # The wait above already fed a sample into the tracker, which would
        # have anchored on the pose held before the countdown. Drop it so the
        # next sample -- the pose being held right now -- becomes the anchor.
        arm.tracker.reset_input_reference()
        arm.cmd_pos = data.site_xpos[arm.site_id].copy()
        arm.cmd_quat = matrix_to_quaternion(data.site_xmat[arm.site_id].reshape(3, 3))
        print("Tracking.", flush=True)
        while True:
            loop_start = clock.monotonic()
            # The absent left side is passed as None; poll_arms skips it.
            result = input_stream.poll_arms(None, arm, None, allow_stop_gesture=True)
            if result.stop_requested:
                raise KeyboardInterrupt
            if result.saw_valid_data:
                arm.last_valid = loop_start

            if (arm.tracker.initialized
                    and clock.monotonic() - arm.last_valid <= PACKET_TIMEOUT_S):
                arm.cmd_pos += (arm.tracker.target_position - arm.cmd_pos) * ctrl["pos_gain"]
                arm.cmd_quat = slerp_step(arm.cmd_quat, arm.tracker.target_quaternion, ctrl["rot_gain"])
                q_sol = solve_pose_ik(
                    model, arm.ik_data, arm.site_id, arm.cmd_pos, arm.cmd_quat,
                    data.qpos.copy(), rot_weight=ctrl["rot_weight"],
                    damping=ctrl["ik_damping"], current_q_weight=ctrl["ik_current_weight"],
                    home_qpos=arm.home, skip_tail_joints=0, dof_indices=addresses,
                )
                output.send_joints(q_sol[addresses], guard=_transport_guard, follow=ctrl["follow"])
                data.qpos[addresses] = q_sol[addresses]

            remaining = CONTROL_PERIOD_S - (clock.monotonic() - loop_start)
            if remaining > 0:
                clock.sleep(remaining)
    finally:
        try:
            if output is not None:
                output.close()
        finally:
            if input_stream is not None:
                input_stream.close()
    return 0


def _heading_yaw_rad(head_pose):
    """Yaw of the headset about the robot-frame Z axis, or None.

    ``get_head_pose`` reports the raw Pico frame, while wrist samples are
    already converted to the robot frame, so the head pose has to go through
    the same transform before its yaw is comparable.
    """
    if head_pose is None or len(head_pose) < 7:
        return None
    import numpy as np

    from utils.adapters.input.pico4 import transform_pico4_to_robot_pose
    from utils.core.quaternion import quaternion_to_matrix

    _position, quaternion = transform_pico4_to_robot_pose(
        np.asarray(head_pose[:3], dtype=np.float64),
        np.asarray(head_pose[3:7], dtype=np.float64),
    )
    rotation = np.asarray(quaternion_to_matrix(quaternion), dtype=np.float64)
    # Only the heading matters: looking down at the bench must not roll the
    # whole mapping.
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _yaw_matrix(yaw_rad):
    import numpy as np

    cos, sin = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _wait_for_tracking(input_stream, arm, *, clock=time, emit=print, timeout_s=0.0):
    """Block until wrist samples are actually arriving.

    Everything else is ready by now -- the arm is homed and the stream is
    open -- but the countdown should not start against dead air, so wait for
    the headset to produce a real frame first.
    """
    emit("Waiting for tracking data...")
    deadline = clock.monotonic() + timeout_s if timeout_s else None
    while True:
        result = input_stream.poll_arms(None, arm, None, allow_stop_gesture=True)
        if result.stop_requested:
            raise KeyboardInterrupt
        if result.saw_valid_data:
            return
        if deadline is not None and clock.monotonic() >= deadline:
            raise TimeoutError("No tracking data; is the headset app running?")
        clock.sleep(CONTROL_PERIOD_S)


def _countdown(seconds=COUNTDOWN_S, *, sleep=time.sleep, emit=print):
    """Count down out loud so the operator can settle into a start pose.

    Runs once everything is ready: the arm is homed and tracking data is
    flowing. The Pico daemon streams continuously, so without this pause the
    anchor would be whatever pose arrived first -- usually a hand still on the
    keyboard right after a restart.
    """
    for remaining in range(int(seconds), 0, -1):
        emit(f"{remaining}...")
        sleep(1.0)
    emit("Anchoring now; hold your start pose.")


class _ArmState:
    """Per-arm teleoperation state, shaped for TeleopInputStream.poll_arms."""

    def __init__(self, *, tracker, site_id, ik_data, cmd_pos, cmd_quat, home):
        self.tracker = tracker
        self.site_id = site_id
        self.ik_data = ik_data
        self.cmd_pos = cmd_pos
        self.cmd_quat = cmd_quat
        self.home = home
        self.last_valid = 0.0
        # Rotation from the VR world frame into the operator's own heading,
        # captured from the headset at anchor time. Until it is set, samples
        # pass through unchanged.
        self._heading_rotation = None
        self._heading_quat = None

    def set_heading(self, yaw_rad):
        """Align the VR world frame with the direction the operator faces.

        Translation residuals are differences of world-frame positions, and
        rotation residuals are left-multiplied, so both are expressed in the
        VR world frame. Without this, turning to stand somewhere else rotates
        the whole mapping -- pushing "forward" sends the arm off at an angle.
        """
        import numpy as np

        from utils.core.quaternion import matrix_to_quaternion

        if yaw_rad is None:
            self._heading_rotation = None
            self._heading_quat = None
            return
        rotation = _yaw_matrix(-yaw_rad)
        self._heading_rotation = rotation
        self._heading_quat = matrix_to_quaternion(
            tuple(tuple(float(v) for v in row) for row in rotation)
        )

    def update_from_pose(self, position, quaternion):
        if self._heading_rotation is not None:
            import numpy as np

            from utils.core.quaternion import quaternion_multiply

            position = tuple(
                self._heading_rotation @ np.asarray(position, dtype=np.float64)
            )
            quaternion = quaternion_multiply(self._heading_quat, tuple(quaternion))
        self.tracker.update(position, quaternion)

    def update_hand_from_mediapipe(self, _landmarks):
        return False  # No end effector.

    def update_hand_from_pinch(self, _distance):
        return False

    def update_hand_from_raw_landmarks(self, _landmarks):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true",
        help="Offline config/model check and exit; opens no sockets (default)",
    )
    mode.add_argument(
        "--read-only", action="store_true",
        help="Connect and print measured joint angles; never moves the arm",
    )
    mode.add_argument(
        "--enable-motion", action="store_true",
        help="Full teleoperation. THE ARM WILL MOVE.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input-source", choices=("pico4", "avp"), default="pico4",
        help="Tracking input source for --enable-motion (default: pico4)",
    )
    parser.add_argument("--avp-ip", default=None, help="Vision Pro IP for --input-source avp")
    parser.add_argument("--port", type=int, default=9000, help="Quest 3 UDP port")
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to run in --read-only mode; 0 runs until interrupted",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.input_source == "avp" and args.enable_motion and not args.avp_ip:
        parser.error("--avp-ip is required with --input-source avp")

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.read_only:
            return run_read_only(args)
        if args.enable_motion:
            return run_motion(args)
        return run_check(args)  # Default: the safest mode.
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        LOG.error("%s", exc)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())

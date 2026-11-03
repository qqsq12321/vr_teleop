"""Teleoperate a single floor-mounted RealMan RM75-B arm over RM_API2.

Three modes, safest first:

    --check        Offline. Loads the model and configs, validates the home
                   pose against the joint limits, prints the SDK angles that
                   WOULD be sent, and opens no sockets. This is the default.
    --read-only    Connects and prints measured joint angles. Never enables
                   motion and never sends a motion command.
    --enable-motion  Full teleoperation. The arm WILL move.
    --hand-only    Drives only the --hand on the tool port; the arm is
                   connected but never enabled.

Run --read-only first on a new installation to confirm the measured joint
angles look right before the arm is allowed to move, and --hand-only before a
hand is driven together with the arm.

Examples:
    python example/teleop_rm75b_single_real.py --check
    python example/teleop_rm75b_single_real.py --read-only --duration 10
    python example/teleop_rm75b_single_real.py --enable-motion --input-source pico4
    python example/teleop_rm75b_single_real.py --hand-only --hand inspire
    python example/teleop_rm75b_single_real.py --enable-motion --hand inspire
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
POSITION_DEADBAND = 0.02
ROTATION_DEADBAND_DEG = 5.0

# Countdown after the homing move completes, before the anchor is taken. The
# daemon streams continuously, so without this the first tracked frame becomes
# the anchor -- typically with a hand still on the keyboard after a restart.
COUNTDOWN_S = 3

# Grip travel past which the controller clutch counts as held. Matches the
# dual-arm entrypoint so the two feel the same.
GRIP_ENGAGE = 0.1

# Hand commands go through the controller's tool port, and every write is a
# blocking round trip. Sending one per 20 ms arm cycle would stretch the arm
# loop and bring back the jitter, so hand targets are rate limited and skipped
# when they have not moved by more than a small fraction of full scale.
HAND_RATE_HZ = 25.0
HAND_MIN_CHANGE_FRACTION = 0.005

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

    # Optional per-hand transport settings. Supply voltage is deliberately not
    # configurable here: it is read, never set.
    hands = config.setdefault("hands", {})
    if not isinstance(hands, dict) or not set(hands) <= {"inspire"}:
        raise ValueError("hands section may only contain inspire")
    for name, settings in hands.items():
        if not isinstance(settings, dict):
            raise ValueError(f"hands.{name} must be a mapping")
        if settings.get("baudrate", 115200) not in (9600, 115200, 460800):
            raise ValueError(f"hands.{name}.baudrate must be 9600, 115200 or 460800")
        rate = settings.get("rate_hz", HAND_RATE_HZ)
        if type(rate) not in (int, float) or not 0 < rate <= 100:
            raise ValueError(f"hands.{name}.rate_hz must be in (0, 100]")
    return config


def _hand_spec(args):
    from utils.robots.realman_rm75b_single import HAND_SPECS

    return HAND_SPECS[getattr(args, "hand", "none")]()


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

    config = load_config(args.config)
    spec = _hand_spec(args)
    model, _data, addresses = _load_model(spec)

    home = spec.arm_home_qpos(SIDE)
    print(f"Arm IP (not contacted): {config['network']['right_ip']}")
    print(f"Scene: {spec.scene_path.relative_to(ROOT)}")
    print(f"Model DoF: nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"Hand: {args.hand} ({len(spec.hand_joint_names(SIDE))} joints)")

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
    """Connect and print measured state. Sends no motion and no hand target."""
    import numpy as np

    config = load_config(args.config)
    spec = _hand_spec(args)
    hand = getattr(args, "hand", "none")

    if output_factory is None:
        from utils.adapters.output.realman_rm75b_single_real import RM75BSingleOutput

        output_factory = RM75BSingleOutput

    output = output_factory(config["network"]["right_ip"])
    home = spec.arm_home_qpos(SIDE)
    deadline = time.monotonic() + args.duration if args.duration else None
    try:
        feedback = output.connect()
        print("READ-ONLY. Motion is not enabled; no motion command will be sent.", flush=True)
        print(f"Controller generation: {output.controller_version}", flush=True)
        print(f"Home pose for reference (deg): {np.rad2deg(home).round(2).tolist()}", flush=True)
        if hand != "none":
            print(f"Tool port supply: {output.tool_voltage()}", flush=True)
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


def _open_input_stream(args, inp, input_factory=None):
    if input_factory is None:
        from utils.core.input_stream import TeleopInputStream

        input_factory = TeleopInputStream
    return input_factory(argparse.Namespace(
        input_source=args.input_source, avp_ip=args.avp_ip, port=args.port,
        pico4_mode=inp.get("mode", "relay"),
        pico4_relay_host=inp.get("relay_host", "127.0.0.1"),
        pico4_relay_port=inp.get("relay_port", 63902),
        pico4_port=inp.get("direct_port", 63901),
        pico4_broadcast_port=inp.get("broadcast_port", 29888),
    ))


def run_hand_only(args, *, output_factory=None, input_factory=None, clock=None,
                  hand_driver_factory=None) -> int:
    """Drive only the hand. The arm is connected but never enabled.

    The safe first power-up for a hand: its open/close direction and finger
    order can be checked without the arm being able to move at all.
    """
    clock = time if clock is None else clock
    if output_factory is None:
        from utils.adapters.output.realman_rm75b_single_real import RM75BSingleOutput

        output_factory = RM75BSingleOutput

    config = load_config(args.config)
    spec = _hand_spec(args)
    output, input_stream = None, None
    try:
        output = output_factory(config["network"]["right_ip"])
        output.connect()
        hand_driver = (hand_driver_factory or _setup_hand)(output, args, spec, config)
        input_stream = _open_input_stream(args, config["input"], input_factory)
        # Only landmarks are used; no tracker, so the wrist pose is ignored.
        state = _ArmState(tracker=None, site_id=None, ik_data=None,
                          cmd_pos=None, cmd_quat=None, home=None)
        print("HAND ONLY. The arm is not enabled and will not move. Ctrl+C stops.", flush=True)
        while True:
            loop_start = clock.monotonic()
            result = input_stream.poll_arms(None, state, None, allow_stop_gesture=True)
            if result.stop_requested:
                raise KeyboardInterrupt
            hand_driver.step(state.latest_landmarks, loop_start)
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


def run_motion(args, *, output_factory=None, input_factory=None, clock=None,
               hand_driver_factory=None) -> int:
    """Full teleoperation. The arm moves."""
    sys.path.insert(1, str(ROOT / "third_party" / "AnyDexRetarget"))
    import mujoco
    import numpy as np

    from utils.adapters.input.pico4 import transform_pico4_to_robot_pose
    from utils.core.arm_controller import slerp_step
    from utils.core.body.ik import solve_pose_ik
    from utils.core.body.wrist_tracker import WristTracker
    from utils.core.quaternion import matrix_to_quaternion

    clock = time if clock is None else clock
    if output_factory is None:
        from utils.adapters.output.realman_rm75b_single_real import RM75BSingleOutput

        output_factory = RM75BSingleOutput

    config = load_config(args.config)
    ctrl, inp = config["control"], config["input"]
    spec = _hand_spec(args)
    model, data, addresses = _load_model(spec)
    controller_mode = bool(getattr(args, "controller", False))
    hand = getattr(args, "hand", "none")

    output, input_stream = None, None
    try:
        output = output_factory(config["network"]["right_ip"])
        output.connect()
        hand_driver = None
        if hand != "none":
            if controller_mode:
                # Controller tracking carries no hand landmarks; leave the hand
                # and its port untouched.
                print(f"--controller: the {hand} hand is not driven.", flush=True)
            else:
                # Before enable_motion: an unpowered tool port must stop the
                # program while the arm is still idle.
                hand_driver = (hand_driver_factory or _setup_hand)(output, args, spec, config)
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

        input_stream = _open_input_stream(args, inp, input_factory)

        print("MOTION ENABLED. Ctrl+C stops.", flush=True)
        print("Homing complete. Get into your start pose.", flush=True)
        if controller_mode:
            # The grip is the readiness signal, so there is nothing to wait out.
            print(
                "Squeeze the grip to track, release to hold still and re-anchor. "
                "Menu button exits.",
                flush=True,
            )
        else:
            _wait_for_tracking(input_stream, arm, clock=clock)
            _countdown()
        def take_anchor(announce=True):
            """Align to the operator's heading and forget the previous anchor.

            The heading is read now, so "forward" means forward regardless of
            where the operator chose to stand. It also absorbs the arm's own
            mounting yaw: an operator naturally squares up to the arm, so the
            two offsets cancel without anyone measuring them.
            """
            yaw = _heading_yaw_rad(input_stream.get_pico4_head_pose())
            if announce:
                if yaw is None:
                    print(
                        "No headset pose; using the raw VR frame. "
                        "Stand square to the arm.",
                        flush=True,
                    )
                else:
                    print(
                        f"Operator heading: {np.rad2deg(yaw):+.1f} deg; aligning to it.",
                        flush=True,
                    )
            arm.set_heading(yaw)
            # Any sample consumed before this point would have anchored on a
            # stale pose, so drop it and let the next one become the anchor.
            arm.tracker.reset_input_reference()
            arm.cmd_pos = data.site_xpos[arm.site_id].copy()
            arm.cmd_quat = matrix_to_quaternion(data.site_xmat[arm.site_id].reshape(3, 3))

        clutch = _GripClutch()
        if not controller_mode:
            take_anchor()
            print("Tracking.", flush=True)
        while True:
            loop_start = clock.monotonic()
            tracking = True
            if controller_mode:
                state = _controller_state(input_stream)
                if state is not None:
                    if state.get("menuButton", False):
                        raise KeyboardInterrupt
                    held, just_engaged = clutch.update(state.get("grip"))
                    if just_engaged:
                        take_anchor()
                        print("Grip engaged; tracking.", flush=True)
                    tracking = held
                    if held:
                        pose = state["pose"]
                        arm.update_from_pose(
                            *transform_pico4_to_robot_pose(pose[:3], pose[3:7])
                        )
                        arm.last_valid = loop_start
                else:
                    tracking = False
            else:
                # The absent left side is passed as None; poll_arms skips it.
                result = input_stream.poll_arms(None, arm, None, allow_stop_gesture=True)
                if result.stop_requested:
                    raise KeyboardInterrupt
                if result.saw_valid_data:
                    arm.last_valid = loop_start
                if hand_driver is not None:
                    hand_driver.step(arm.latest_landmarks, loop_start)

            if (tracking and arm.tracker.initialized
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


class _GripClutch:
    """The grip is a clutch: squeezing tracks, letting go releases.

    This replaces the countdown in controller mode. The operator says when
    they are ready by squeezing, so there is nothing to wait out, and letting
    go re-anchors without restarting the program.
    """

    def __init__(self, *, engage=GRIP_ENGAGE):
        self.engage = engage
        self.engaged = False

    def update(self, grip):
        """Return (held, just_engaged) for this frame."""
        try:
            held = grip is not None and float(grip) > self.engage
        except (TypeError, ValueError):
            held = False
        just_engaged = held and not self.engaged
        self.engaged = held
        return held, just_engaged


def _controller_state(input_stream, side=SIDE):
    """Latest controller state, or None when controllers are unavailable.

    TeleopInputStream only forwards hand tracking, so the controller is read
    straight from the Pico adapter underneath it.
    """
    pico = getattr(input_stream, "pico4_input", None)
    if pico is None or not hasattr(pico, "get_controller"):
        return None
    return pico.get_controller(side)


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


class _HandDriver:
    """Retarget hand landmarks and send them to the tool port, sparingly.

    Both the retarget optimisation and the tool-port write run inside the arm
    loop, so both are held to ``rate_hz``; a target that has barely moved is
    not resent. The hand simply holds its last target in between.
    """

    def __init__(self, *, retarget, to_channels, send, full_scale,
                 rate_hz=HAND_RATE_HZ, min_change_fraction=HAND_MIN_CHANGE_FRACTION):
        self._retarget = retarget
        self._to_channels = to_channels
        self._send = send
        self._period = 1.0 / float(rate_hz)
        self._min_change = float(full_scale) * float(min_change_fraction)
        self._last_time = None
        self._last_sent = None

    def step(self, landmarks, now) -> bool:
        """Return True when a target was actually sent."""
        import numpy as np

        if landmarks is None:
            return False
        if self._last_time is not None and now - self._last_time < self._period:
            return False
        self._last_time = now
        raw = self._retarget(landmarks)
        if raw is None or not np.all(np.isfinite(raw)):
            return False
        channels = self._to_channels(raw)
        if self._last_sent is not None and max(
            abs(a - b) for a, b in zip(channels, self._last_sent)
        ) < self._min_change:
            return False
        self._send(channels)
        self._last_sent = list(channels)
        return True


def _setup_hand(output, args, spec, config):
    """Configure the tool port and build a driver for the selected hand.

    Runs before any arm motion is enabled, so an unpowered tool port or a
    missing retarget config stops the program while the arm is still idle.
    """
    sys.path.insert(1, str(ROOT / "third_party" / "AnyDexRetarget"))
    from utils.adapters.output.inspire_mapping import raw_qpos_to_channels
    from utils.adapters.output.realman_rm75b_single_real import INSPIRE_CHANNEL_MAX
    from utils.core.hand.retarget import HandRetargeter

    kind = args.hand
    settings = config["hands"].get(kind, {})
    retargeter = HandRetargeter(spec.hand_config_path(args.input_source, SIDE), SIDE)
    if not retargeter.available:
        raise RuntimeError(f"{kind} hand retargeter unavailable")
    output.configure_hand(kind, baudrate=settings.get("baudrate", 115200))
    output.enable_hand()
    print(f"Hand: {kind} on the tool port ({output.tool_voltage()}).", flush=True)

    def to_channels(raw):
        return raw_qpos_to_channels(raw, spec, scale=INSPIRE_CHANNEL_MAX)

    return _HandDriver(
        retarget=retargeter.retarget_mediapipe,
        to_channels=to_channels,
        send=lambda channels: output.send_hand(channels, guard=_transport_guard),
        full_scale=INSPIRE_CHANNEL_MAX,
        rate_hz=settings.get("rate_hz", HAND_RATE_HZ),
    )


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
        # Latest wrist-relative hand landmarks, for the hand driver. Stored
        # rather than retargeted here so the optimiser runs only at hand rate.
        self.latest_landmarks = None
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
        if self.tracker is None:
            return  # Hand-only mode: the arm is not tracked.
        if self._heading_rotation is not None:
            import numpy as np

            from utils.core.quaternion import quaternion_multiply

            position = tuple(
                self._heading_rotation @ np.asarray(position, dtype=np.float64)
            )
            quaternion = quaternion_multiply(self._heading_quat, tuple(quaternion))
        self.tracker.update(position, quaternion)

    def update_hand_from_mediapipe(self, landmarks):
        if landmarks is not None:
            self.latest_landmarks = landmarks
        # Not wrist data: hand landmarks alone must not keep a stale wrist
        # pose looking fresh to the arm.
        return False

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
    mode.add_argument(
        "--hand-only", action="store_true",
        help="Drive only the --hand; the arm is never enabled. Use for a hand's "
             "first power-up.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--hand", choices=("none", "inspire"), default="none",
        help="Hand on the flange's tool port: none (default) or inspire. "
             "Hand-tracking only; not driven with --controller.",
    )
    parser.add_argument(
        "--input-source", choices=("pico4", "avp"), default="pico4",
        help="Tracking input source for --enable-motion (default: pico4)",
    )
    parser.add_argument("--avp-ip", default=None, help="Vision Pro IP for --input-source avp")
    parser.add_argument("--port", type=int, default=9000, help="Quest 3 UDP port")
    parser.add_argument(
        "--controller", action="store_true",
        help="Track the Pico controller instead of the bare hand: squeeze the "
             "grip to track, release to hold still and re-anchor",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to run in --read-only mode; 0 runs until interrupted",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.input_source == "avp" and (args.enable_motion or args.hand_only) and not args.avp_ip:
        parser.error("--avp-ip is required with --input-source avp")
    if args.controller and args.input_source != "pico4":
        parser.error("--controller is only available with --input-source pico4")
    if args.hand_only and args.hand == "none":
        parser.error("--hand-only needs --hand inspire")
    if args.hand_only and args.controller:
        parser.error("--hand-only uses hand tracking; --controller carries no hand data")

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.read_only:
            return run_read_only(args)
        if args.hand_only:
            return run_hand_only(args)
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

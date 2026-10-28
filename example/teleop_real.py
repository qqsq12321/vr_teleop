#!/usr/bin/env python3
"""RealMan RM75-B dual-arm teleop, following master's RM65 control loop.

Pico -> wrist smoothing -> pose IK -> joint passthrough (follow=True).
Moves real arms to the configured RM75 home BEFORE waiting for Pico tracking,
then starts teleop. No confirmation prompt or viewer. Default: both arms, hands
off. --enable-hands enables the connected Inspire pair after startup motion.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# master:example/teleop_real.py, RM65 constants. Hardware/model mapping is RM75.
CONTROL_PERIOD_S = 0.02
PACKET_TIMEOUT_S = 0.25
POSITION_DEADBAND = 0.03
ROTATION_DEADBAND_DEG = 8.0
# AVP/Pico pinch distance (meters) mapped to the RealMan gripper range.
# A closer pinch means a more closed gripper.
GRIPPER_PINCH_CLOSED_M = 0.02
GRIPPER_PINCH_OPEN_M = 0.10


def _transport_guard(_now):
    """No additional host deadline policy on the master's direct-send path."""


def _pinch_to_gripper_position(distance):
    """Convert a pinch distance in meters to a RealMan position (1..1000)."""
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(distance):
        return None
    span = GRIPPER_PINCH_OPEN_M - GRIPPER_PINCH_CLOSED_M
    normalized = (distance - GRIPPER_PINCH_CLOSED_M) / span
    normalized = max(0.0, min(1.0, normalized))
    return int(round(1.0 + 999.0 * normalized))


def run(args, *, output_factory=None, input_factory=None, retargeter_factory=None,
        hand_output_factory=None, clock=None):
    # Lazy imports keep --help from loading the robot stack.
    sys.path.insert(0, str(ROOT / 'third_party' / 'AnyDexRetarget'))
    import mujoco
    import numpy as np
    import yaml

    from utils.adapters.input.pico4 import Pico4, transform_pico4_to_robot_pose
    from utils.adapters.input.avp import AVPInput
    from utils.adapters.output.inspire_mapping import raw_qpos_to_channels
    from utils.adapters.output.realman_real import RealManOutput
    from utils.adapters.output.linker_l20_real import LinkerL20RealOutput
    from utils.core.arm_controller import slerp_step
    from utils.core.body.ik import solve_pose_ik
    from utils.core.body.wrist_tracker import WristTracker
    from utils.core.hand.retarget import HandRetargeter
    from utils.core.quaternion import matrix_to_quaternion, quaternion_to_matrix
    from utils.robots.realman_rm75b import RealManRM75BInspireSpec
    from utils.robots.realman_rm75b_l20 import RealManRM75BL20Spec

    clock = time if clock is None else clock
    output_factory = RealManOutput if output_factory is None else output_factory
    input_source = getattr(args, 'input_source', 'pico4')
    hand_type = getattr(args, 'hand', 'inspire')
    output_input_factory = Pico4 if input_factory is None else input_factory
    retargeter_factory = HandRetargeter if retargeter_factory is None else retargeter_factory
    hand_output_factory = LinkerL20RealOutput if hand_output_factory is None else hand_output_factory
    controller_mode = bool(getattr(args, 'controller', False))
    legacy_gripper_mode = getattr(args, 'ee', None) == 'gripper'
    if hand_type not in {'inspire', 'linker_l20'}:
        raise ValueError(f"Unsupported hand backend: {hand_type}")
    if controller_mode and input_source != 'pico4':
        raise ValueError('--controller is only supported with --input-source pico4')
    if input_source == 'avp' and not getattr(args, 'avp_ip', None):
        raise ValueError('--avp-ip is required with --input-source avp')

    # ``--hand`` remains the backwards-compatible global selector.  The
    # side-specific selectors allow mixed hardware, e.g. left gripper + right
    # L20, without probing or opening the absent left L20.
    sides = ('left', 'right') if args.arm == 'both' else (args.arm,)
    side_hand = {
        side: (getattr(args, f'{side}_hand', None) or hand_type)
        for side in sides
    }
    for side, backend in side_hand.items():
        if backend not in {'none', 'gripper', 'inspire', 'linker_l20'}:
            raise ValueError(f"Unsupported {side} hand backend: {backend}")
    explicit_hand = any(getattr(args, f'{side}_hand', None) is not None for side in sides)
    if legacy_gripper_mode:
        if explicit_hand or any(backend == 'linker_l20' for backend in side_hand.values()):
            raise ValueError('--ee gripper cannot be combined with side-specific hands or L20')
        if not controller_mode:
            raise ValueError('--ee gripper requires --controller')
        side_hand = {side: 'gripper' for side in sides}
    l20_sides = tuple(side for side in sides if side_hand[side] == 'linker_l20')
    gripper_sides = tuple(side for side in sides if side_hand[side] == 'gripper')
    inspire_sides = tuple(side for side in sides if side_hand[side] == 'inspire')
    l20_mode = bool(l20_sides)
    if l20_mode and controller_mode:
        raise ValueError('--controller cannot be combined with linker_l20; L20 requires hand landmarks')
    # Explicit side selection activates that hand.  The old global Inspire
    # selector still requires --enable-hands, while global L20 remains active.
    enable_hands = bool(inspire_sides) and (
        explicit_hand or bool(getattr(args, 'enable_hands', False))
    )
    active_inspire_sides = inspire_sides if enable_hands else ()
    with (ROOT / 'config/realman_real.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    with (ROOT / 'config/realman_calibration.yaml').open(encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream)
    ctrl, inp = config['control'], config['input']
    spec = RealManRM75BL20Spec() if l20_mode else RealManRM75BInspireSpec()
    realman_kwargs = {**config['network'], **calibration}
    # The L20 scene unwraps right J1 by -2*pi to stay inside MuJoCo's joint
    # limits. The physical RealMan joint uses the equivalent +2*pi branch.
    if l20_mode:
        realman_kwargs['right_j1_model_wrap_rad'] = 2.0 * np.pi
    model = mujoco.MjModel.from_xml_path(str(spec.scene_path))
    data = mujoco.MjData(model)
    addresses = {s: np.array([model.jnt_qposadr[model.joint(n).id]
                             for n in spec.arm_joint_names(s)])
                 for s in ('left', 'right')}
    output, input_device, hand_output = None, None, None
    try:
        # Load only the requested hand configurations before allowing arm motion.
        retargeters = {}
        for side in sides:
            backend = side_hand[side]
            hand_active = backend == 'linker_l20' or side in active_inspire_sides
            retargeter = (retargeter_factory(spec.hand_config_path(input_source, side), side)
                          if hand_active and not controller_mode else None)
            if retargeter is not None and not retargeter.available:
                raise RuntimeError(f'{side}: hand retargeter unavailable')
            retargeters[side] = retargeter
        if l20_mode:
            hand_output = hand_output_factory(
                sides=l20_sides,
                port=getattr(args, 'l20_port', '/dev/ttyUSB0'),
                left_port=getattr(args, 'l20_left_port', None),
                right_port=getattr(args, 'l20_right_port', None),
                baudrate=int(getattr(args, 'l20_baudrate', 460800)),
                left_slave_id=int(getattr(args, 'l20_left_slave_id', 0x2A)),
                right_slave_id=int(getattr(args, 'l20_right_slave_id', 0x29)),
                speed=int(getattr(args, 'l20_speed', 200)),
                current_limit=int(getattr(args, 'l20_current_limit', 200)),
                clear_faults=bool(getattr(args, 'l20_clear_faults', False)),
                open_on_exit=bool(getattr(args, 'l20_open_on_exit', False)),
                print_registers=bool(getattr(args, 'l20_print_registers', False)),
                dry_run=bool(getattr(args, 'l20_dry_run', False)),
            )
        output = output_factory(sides=sides, **realman_kwargs)
        output.connect()  # Check the selected controllers and their current fault state.
        output.enable_motion()
        # Match master's planned, blocking startup move, using RM75's seven-joint
        # home (not RM65's six-axis values). SDK calls remain serial for both arms.
        for side in sides:
            q_home = spec.arm_home_qpos(side)
            print(f'Moving {side} to initial pose (deg): {output.model_to_sdk_degrees(side, q_home).round(2).tolist()}', flush=True)
            output.move_to_initial_pose(side, q_home, guard=_transport_guard)
        # Re-read AFTER all startup moves. Never use the pre-move feedback or
        # merely assume the commanded home was reached when establishing FK/IK.
        states = output.read_feedback()
        for side in sides:
            indices = addresses[side]
            data.qpos[indices] = states[side].q_rad
        mujoco.mj_forward(model, data)
        arms = {}
        for side in sides:
            site_id = model.site(f'{side}_tcp').id
            position = data.site_xpos[site_id].copy()
            quaternion = matrix_to_quaternion(data.site_xmat[site_id].reshape(3, 3))
            tracker = WristTracker(
                position, quaternion, position_scale=ctrl['position_scale'],
                ema_alpha=ctrl['ema_alpha'], negate_rot_xy=False,
                base_xmat=None, position_deadband=POSITION_DEADBAND,
                rotation_deadband_deg=ROTATION_DEADBAND_DEG)
            retargeter = retargeters[side]
            arms[side] = SimpleNamespace(
                tracker=tracker, site_id=site_id, ik_data=mujoco.MjData(model),
                cmd_pos=position.copy(), cmd_quat=np.array(quaternion),
                home=data.qpos[addresses[side]].copy(), retargeter=retargeter,
                hand_angles=None, hand_qpos=None, hand_joint_names=(), last_valid=0.)
        if input_source == 'avp':
            input_device = (AVPInput(ip=getattr(args, 'avp_ip', None))
                            if input_factory is None else
                            output_input_factory(ip=getattr(args, 'avp_ip', None)))
        else:
            input_device = output_input_factory(
                mode=inp['mode'], relay_host=inp['relay_host'],
                relay_port=inp['relay_port'], port=inp['direct_port'],
                broadcast_port=inp['broadcast_port'])
        if l20_mode:
            for side in l20_sides:
                hand_output.open(side)
        if active_inspire_sides:
            output.configure_hands(dict(config['hands'], sides=active_inspire_sides), guard=_transport_guard)
        if gripper_sides:
            for side in gripper_sides:
                output.configure_gripper_speed(side, 1000, guard=_transport_guard)
        gripper_position = {side: 1000 for side in gripper_sides}
        last_gripper_command = {side: None for side in gripper_sides}
        gripper_step = 1000
        print(f'MOTION ENABLED; arm={args.arm}; hands={side_hand}; input={input_source}; controller={controller_mode}', flush=True)
        print('Initial pose motion completed. First wrist frame aligns to the measured pose; Ctrl+C stops.', flush=True)
        while True:
            loop_start = clock.monotonic()
            # AVP requires an explicit frame refresh; Pico maintains a background cache.
            if input_source == 'avp':
                input_device.poll()
                if input_device.check_stop_gesture():
                    raise KeyboardInterrupt
            # Same getter/cache semantics as master. Valid wrist OR hand data
            # refreshes that arm's last-valid-packet time.
            for side, arm in arms.items():
                saw_valid_data = False
                wrist = input_device.get_wrist_pose(side)
                if controller_mode:
                    controller = input_device.get_controller(side)
                    if controller is not None:
                        if controller.get('menuButton', False):
                            raise KeyboardInterrupt
                        if side in gripper_sides:
                            trigger = float(controller.get('trigger', 0.0))
                            if trigger > 0.1:
                                gripper_position[side] = max(
                                    1, gripper_position[side] - max(1, int(round(gripper_step * trigger))))
                            else:
                                gripper_position[side] = min(
                                    1000, gripper_position[side] + gripper_step)
                        if float(controller.get('grip', 0.0)) <= 0.1:
                            arm.tracker.reset_input_reference()
                            continue
                        pose = controller['pose']
                        wrist = transform_pico4_to_robot_pose(pose[:3], pose[3:7])
                if wrist is not None:
                    arm.tracker.update(*wrist)
                    saw_valid_data = True
                if side in gripper_sides and not controller_mode:
                    pinch = input_device.get_pinch_distance(side)
                    position = _pinch_to_gripper_position(pinch)
                    if position is not None:
                        gripper_position[side] = position
                        saw_valid_data = True
                if arm.retargeter is not None:
                    points = input_device.get_landmarks_mediapipe(side)
                    if points is not None:
                        raw = arm.retargeter.retarget_mediapipe(points)
                        if side_hand[side] == 'linker_l20':
                            valid = raw is not None and np.shape(raw) == (21,) and np.all(np.isfinite(raw))
                            if valid:
                                arm.hand_qpos = np.asarray(raw, dtype=np.float64).copy()
                                arm.hand_joint_names = arm.retargeter.joint_names
                                saw_valid_data = True
                        elif raw is not None and np.shape(raw) == (12,) and np.all(np.isfinite(raw)):
                            # Master normalizes to 0..2000, then divides by two
                            # at its serial boundary. Native SDK needs 0..1000.
                            channels = raw_qpos_to_channels(raw, spec, scale=2000)
                            arm.hand_angles = [int(round(v / 2.0)) for v in channels]
                            saw_valid_data = True
                if saw_valid_data:
                    arm.last_valid = loop_start
            for side, arm in arms.items():
                if arm.tracker.initialized and clock.monotonic() - arm.last_valid <= PACKET_TIMEOUT_S:
                    arm.cmd_pos += (arm.tracker.target_position - arm.cmd_pos) * ctrl['pos_gain']
                    arm.cmd_quat = slerp_step(arm.cmd_quat, arm.tracker.target_quaternion, ctrl['rot_gain'])
                    q_sol = solve_pose_ik(
                        model, arm.ik_data, arm.site_id, arm.cmd_pos, arm.cmd_quat,
                        data.qpos.copy(), rot_weight=ctrl['rot_weight'],
                        damping=ctrl['ik_damping'], current_q_weight=ctrl['ik_current_weight'],
                        home_qpos=arm.home, skip_tail_joints=0, dof_indices=addresses[side])
                    output.send_joints(side, q_sol[addresses[side]],
                                       guard=_transport_guard, follow=ctrl['follow'])
                    data.qpos[addresses[side]] = q_sol[addresses[side]]
                # Master sends the last valid hand target outside the arm timeout
                # branch, with no added 10 Hz cap, sample dedup, or ±20 slew limit.
                if side in gripper_sides and last_gripper_command[side] != gripper_position[side]:
                    output.send_gripper_position(side, gripper_position[side], guard=_transport_guard)
                    last_gripper_command[side] = gripper_position[side]
                if side_hand[side] == 'linker_l20' and arm.hand_qpos is not None:
                    hand_output.send(side, arm.hand_qpos, arm.hand_joint_names)
                elif (side in active_inspire_sides and arm.hand_angles is not None
                      and not controller_mode):
                    output.send_hand(side, arm.hand_angles, guard=_transport_guard)
            remaining = CONTROL_PERIOD_S - (clock.monotonic() - loop_start)
            if remaining > 0:
                clock.sleep(remaining)
    finally:
        try:
            if hand_output is not None:
                hand_output.close()
        finally:
            try:
                if output is not None:
                    output.close()  # Best-effort stop both arms, then disconnect.
            finally:
                if input_device is not None and hasattr(input_device, 'stop'):
                    input_device.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='both',
                        help='Arm selection (default: both; single-arm option retained)')
    parser.add_argument('--hand', choices=('inspire', 'linker_l20'), default='inspire',
                        help='Hand backend (default: inspire; linker_l20 uses RS485)')
    parser.add_argument('--left-hand', choices=('none', 'gripper', 'inspire', 'linker_l20'), default=None,
                        help='Left-side hand backend; overrides --hand for the left side')
    parser.add_argument('--right-hand', choices=('none', 'gripper', 'inspire', 'linker_l20'), default=None,
                        help='Right-side hand backend; overrides --hand for the right side')
    parser.add_argument('--input-source', choices=('pico4', 'avp'), default='pico4',
                        help='Tracking input source (default: pico4)')
    parser.add_argument('--avp-ip', default=None,
                        help='Vision Pro IP address used with --input-source avp')
    parser.add_argument('--ee', choices=('gripper',),
                        help='Enable official RealMan gripper control; requires --controller')
    parser.add_argument('--enable-hands', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--controller', action='store_true',
                        help='Use Pico controller poses instead of hand/wrist tracking for arm teleoperation')
    parser.add_argument('--l20-port', default='/dev/ttyUSB0',
                        help='Shared/default L20 RS485 serial port')
    parser.add_argument('--l20-left-port', default=None,
                        help='Optional dedicated left-hand RS485 port')
    parser.add_argument('--l20-right-port', default=None,
                        help='Optional dedicated right-hand RS485 port')
    parser.add_argument('--l20-baudrate', type=int, default=460800)
    parser.add_argument('--l20-left-slave-id', type=lambda value: int(value, 0), default=0x2A)
    parser.add_argument('--l20-right-slave-id', type=lambda value: int(value, 0), default=0x29)
    parser.add_argument('--l20-speed', type=int, default=200)
    parser.add_argument('--l20-current-limit', type=int, default=200)
    parser.add_argument('--l20-clear-faults', action='store_true')
    parser.add_argument('--l20-open-on-exit', action='store_true')
    parser.add_argument('--l20-print-registers', action='store_true')
    parser.add_argument('--l20-dry-run', action='store_true',
                        help='Disable L20 serial writes while testing the arm/input path')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    # The vendor SDK logs every command at INFO; per-cycle gripper/CANFD calls
    # would flood the console. Keep its warnings and errors.
    logging.getLogger('third_party.realman_api.robotic_arm').setLevel(logging.WARNING)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Stopped by operator.')
        return 130
    except Exception as exc:
        logging.error('Session aborted: %s', exc)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

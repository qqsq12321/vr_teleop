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


def _transport_guard(_now):
    """No additional host deadline policy on the master's direct-send path."""


def run(args, *, output_factory=None, input_factory=None, retargeter_factory=None,
        clock=None):
    # Lazy imports keep --help from loading the robot stack.
    sys.path.insert(0, str(ROOT / 'third_party' / 'AnyDexRetarget'))
    import mujoco
    import numpy as np
    import yaml

    from utils.adapters.input.pico4 import Pico4, transform_pico4_to_robot_pose
    from utils.adapters.output.inspire_mapping import raw_qpos_to_channels
    from utils.adapters.output.realman_real import RealManOutput
    from utils.core.arm_controller import slerp_step
    from utils.core.body.ik import solve_pose_ik
    from utils.core.body.wrist_tracker import WristTracker
    from utils.core.hand.retarget import HandRetargeter
    from utils.core.quaternion import matrix_to_quaternion, quaternion_to_matrix
    from utils.robots.realman_rm75b import RealManRM75BInspireSpec

    clock = time if clock is None else clock
    output_factory = RealManOutput if output_factory is None else output_factory
    input_factory = Pico4 if input_factory is None else input_factory
    retargeter_factory = HandRetargeter if retargeter_factory is None else retargeter_factory
    controller_mode = bool(getattr(args, 'controller', False))
    gripper_mode = getattr(args, 'ee', None) == 'gripper'
    enable_hands = bool(getattr(args, 'enable_hands', False)) and not gripper_mode
    if gripper_mode and not controller_mode:
        raise ValueError('--ee gripper requires --controller')
    with (ROOT / 'config/realman_real.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    with (ROOT / 'config/realman_calibration.yaml').open(encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream)
    ctrl, inp = config['control'], config['input']
    spec = RealManRM75BInspireSpec()
    sides = ('left', 'right') if args.arm == 'both' else (args.arm,)
    model = mujoco.MjModel.from_xml_path(str(spec.scene_path))
    data = mujoco.MjData(model)
    addresses = {s: np.array([model.jnt_qposadr[model.joint(n).id]
                             for n in spec.arm_joint_names(s)])
                 for s in ('left', 'right')}
    output, pico = None, None
    try:
        # Load all requested hand configurations before allowing any arm motion.
        retargeters = {}
        for side in sides:
            retargeter = (retargeter_factory(spec.hand_config_path('pico4', side), side)
                          if enable_hands and not controller_mode else None)
            if retargeter is not None and not retargeter.available:
                raise RuntimeError(f'{side}: hand retargeter unavailable')
            retargeters[side] = retargeter
        output = output_factory(**config['network'], **calibration)
        output.connect()  # Check both controllers and their current fault state.
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
        for side, indices in addresses.items():
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
                hand_angles=None, last_valid=0.)
        pico = input_factory(mode=inp['mode'], relay_host=inp['relay_host'],
                             relay_port=inp['relay_port'], port=inp['direct_port'],
                             broadcast_port=inp['broadcast_port'])
        if enable_hands:
            output.configure_hands(dict(config['hands'], sides=sides), guard=_transport_guard)
        if gripper_mode:
            for side in sides:
                output.configure_gripper_speed(side, 1000, guard=_transport_guard)
        gripper_position = {side: 1000 for side in sides}
        last_gripper_command = {side: None for side in sides}
        gripper_step = 1000
        print(f'MOTION ENABLED; arm={args.arm}; hands={enable_hands}; controller={controller_mode}; master-style loop', flush=True)
        print('Initial pose motion completed. First wrist frame aligns to the measured pose; Ctrl+C stops.', flush=True)
        while True:
            loop_start = clock.monotonic()
            # Same Pico getter/cache semantics as master. Valid wrist OR hand
            # data refreshes that arm's last-valid-packet time, as in RM65.
            for side, arm in arms.items():
                saw_valid_data = False
                wrist = pico.get_wrist_pose(side)
                if controller_mode:
                    controller = pico.get_controller(side)
                    if controller is not None:
                        if controller.get('menuButton', False):
                            raise KeyboardInterrupt
                        if gripper_mode:
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
                if arm.retargeter is not None:
                    points = pico.get_landmarks_mediapipe(side)
                    if points is not None:
                        raw = arm.retargeter.retarget_mediapipe(points)
                        if raw is not None and np.shape(raw) == (12,) and np.all(np.isfinite(raw)):
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
                if gripper_mode and controller_mode and last_gripper_command[side] != gripper_position[side]:
                    output.send_gripper_position(side, gripper_position[side], guard=_transport_guard)
                    last_gripper_command[side] = gripper_position[side]
                if arm.hand_angles is not None and not (enable_hands and controller_mode):
                    output.send_hand(side, arm.hand_angles, guard=_transport_guard)
            remaining = CONTROL_PERIOD_S - (clock.monotonic() - loop_start)
            if remaining > 0:
                clock.sleep(remaining)
    finally:
        try:
            if output is not None:
                output.close()  # Best-effort stop both arms, then disconnect.
        finally:
            if pico is not None:
                pico.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='both',
                        help='Arm selection (default: both; single-arm option retained)')
    parser.add_argument('--ee', choices=('gripper',),
                        help='Enable official RealMan gripper control; requires --controller')
    parser.add_argument('--enable-hands', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--controller', action='store_true',
                        help='Use Pico controller poses instead of hand/wrist tracking for arm teleoperation')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
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

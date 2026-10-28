"""Daily master-style real entry. Every socket/actuator in these tests is fake."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    'realman_daily_entry', Path(__file__).resolve().parents[1] / 'example/teleop_real.py')
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)


@pytest.mark.parametrize('argv, expected', [
    ([], ('both', False)),
    (['--arm', 'right'], ('right', False)),
    (['--arm', 'both', '--enable-hands'], ('both', True)),
])
def test_entry_defaults_to_both_and_selects_hands(monkeypatch, argv, expected):
    calls = []
    monkeypatch.setattr(entry, 'run', lambda args: calls.append((args.arm, args.enable_hands)))
    assert entry.main(argv) == 0
    assert calls == [expected]


@pytest.mark.parametrize('flag', ['--dry-run', '--viewer', '--read-only', '--check',
                                  '--pause-on-tracking-loss', '--duration', '--dexforce-ip'])
def test_old_debug_flags_do_not_start_hardware(monkeypatch, flag):
    def forbidden(*args): raise AssertionError('Must not start robot stack')
    monkeypatch.setattr(entry, 'run', forbidden)
    with pytest.raises(SystemExit) as exc:
        entry.main([flag])
    assert exc.value.code == 2


def test_help_imports_neither_sdk_nor_legacy_runner():
    script = '''
import sys
from example import teleop_real
try:
    teleop_real.main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'mujoco' not in sys.modules
assert 'third_party.realman_api.robotic_arm' not in sys.modules
assert 'example.teleop_realman_real' not in sys.modules
assert 'utils.core.safety' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=entry.ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_control_parameters_match_local_master():
    source = subprocess.check_output(
        ['git', 'show', 'master:example/teleop_real.py'], cwd=entry.ROOT, text=True)
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    config = yaml.safe_load((entry.ROOT / 'config/realman_real.yaml').read_text())
    for key in ('position_scale', 'ema_alpha', 'rot_weight', 'ik_damping',
                'ik_current_weight', 'pos_gain', 'rot_gain'):
        assert config['control'][key] == constants['_RM65_' + key.upper()]
    assert config['control']['follow'] is True
    assert entry.CONTROL_PERIOD_S == constants['_RM65_CONTROL_PERIOD_S']
    assert entry.PACKET_TIMEOUT_S == constants['_RM65_PACKET_TIMEOUT_S']
    assert entry.POSITION_DEADBAND == constants['_RM65_WRIST_POS_DEADBAND']
    assert entry.ROTATION_DEADBAND_DEG == constants['_RM65_WRIST_ROT_DEADBAND_DEG']


@pytest.fixture
def fake_rig(monkeypatch):
    from utils.adapters.output.realman_real import RealManOutput
    from utils.core.body import ik
    from utils.robots.realman_rm75b import RealManRM75BInspireSpec
    spec = RealManRM75BInspireSpec()
    state = SimpleNamespace(cycle=0, sides=('left', 'right'), cycles=12,
                            work=.02, missing=set(), bad_hand=set(), fail_sdk=False,
                            read_error=False, post_home_read_error=False, fail_home=None,
                            interrupt_home=None, hand_unavailable=False,
                            events=[], solutions={}, initial={},
                            prior={}, retargeters=[], model=None)
    class Clock:
        now = 10.
        sleeps = []
        def monotonic(self): return self.now
        def sleep(self, duration):
            self.sleeps.append(duration)
            self.now += duration
    clock = Clock()
    state.clock = clock
    def sdk_angles(side, q):
        result = np.rad2deg(q)
        if side == "right":
            result[0] -= 180.0
        return result
    state.sdk_angles = sdk_angles
    class SDK:
        code = 7
        def __init__(self, ip):
            self.side = 'left' if ip.endswith('.18') else 'right'
            # Before homing, feedback deliberately differs from the home.
            # After homing it stays constant, even as CANFD targets change.
            self.q = spec.arm_home_qpos(self.side) + .05
            self.read_count = 0
            state.initial[self.side] = self.q.copy()
        def Arm_Socket_State(self): return 0
        def Get_Arm_Software_Info(self):
            return 0, SimpleNamespace(product_version=b'RM75-BI')
        def Get_Current_Arm_State(self, retry):
            self.read_count += 1
            state.events.append(('read', state.cycle, self.side))
            error = state.read_error or (state.post_home_read_error and self.read_count > 1)
            return 0, sdk_angles(self.side, self.q).tolist(), [], [1] if error else [0]
        def Movej_Cmd(self, q, v, r, trajectory_connect, block):
            assert (v, r, trajectory_connect, block) == (20, 0, 0, True)
            np.testing.assert_array_equal(q, sdk_angles(self.side, spec.arm_home_qpos(self.side)))
            state.events.append(('home', state.cycle, self.side, np.array(q)))
            if state.interrupt_home == self.side:
                raise KeyboardInterrupt
            if state.fail_home == self.side:
                return 9
            clock.now += 2.0  # Planning lasts much longer than a teleop cycle.
            # Model arrival feedback slightly different from the requested pose.
            # The first IK / CANFD must use this measurement, not the old q or target.
            self.q = np.deg2rad(q) + .001
            if self.side == "right":
                self.q[0] += np.pi
            state.initial[self.side] = self.q.copy()
            state.events.append(('home-done', state.cycle, self.side))
            return 0
        def Movej_CANFD(self, q, follow):
            assert follow is True
            np.testing.assert_array_equal(q, sdk_angles(self.side, state.solutions[self.side]))
            state.events.append(('move', state.cycle, self.side, np.array(q)))
            return 9 if state.fail_sdk else 0
        def Set_Hand_Angle(self, channels, block):
            assert block is False
            state.events.append(('hand', state.cycle, self.side, list(channels)))
            return 0
        def Move_Stop_Cmd(self, block):
            state.events.append(('stop', state.cycle, self.side))
            return 0
        def Arm_Socket_Close(self): state.events.append(('close', state.cycle, self.side))
        def RM_API_UnInit(self): state.events.append(('uninit', state.cycle))
    class Pico:
        def __init__(self, **kwargs):
            assert {e[2] for e in state.events if e[0] == 'home-done'} == set(state.sides)
            state.events.append(('pico-start', state.cycle))
        def get_wrist_pose(self, side):
            if side == state.sides[0]:
                state.cycle += 1
                if state.cycle > state.cycles: raise KeyboardInterrupt
                clock.now += state.work
            if state.cycle in state.missing: return None
            return ((.3 + .035 * (state.cycle - 1), .3, 1), (0, 0, 0, 1))
        def get_landmarks_mediapipe(self, side):
            return None if state.cycle in state.missing else np.full((21, 3), state.cycle)
        def stop(self): state.events.append(('pico-stop', state.cycle))
    class Retargeter:
        available = True
        def __init__(self, config, side):
            self.available = not state.hand_unavailable
            state.retargeters.append((config, side))
        def retarget_mediapipe(self, points):
            if state.cycle in state.bad_hand: return np.full(12, np.nan)
            return np.full(12, 0 if state.cycle % 2 else .7)
    original_solve = ik.solve_pose_ik
    def capture_solve(model, workspace, site_id, pos, quat, q_init, **kwargs):
        state.model = model
        side = 'left' if model.site(site_id).name == 'left_tcp' else 'right'
        ids = kwargs['dof_indices']
        assert len(ids) == 7 and kwargs['skip_tail_joints'] == 0
        assert kwargs['rot_weight'] == 1.0
        assert kwargs['damping'] == .001
        assert kwargs['current_q_weight'] == .3
        expected = state.prior.get(side, state.initial[side])
        np.testing.assert_allclose(q_init[ids], expected, atol=1e-14)
        np.testing.assert_allclose(kwargs['home_qpos'], state.initial[side], atol=1e-14)
        q = original_solve(model, workspace, site_id, pos, quat, q_init, **kwargs)
        state.solutions[side] = q[ids].copy()
        state.prior[side] = q[ids].copy()
        return q
    monkeypatch.setattr(ik, 'solve_pose_ik', capture_solve)
    def forbidden_prompt(*args): raise AssertionError('No confirmation prompt')
    monkeypatch.setattr('builtins.input', forbidden_prompt)
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    state.run = lambda arm='both', hands=False: entry.run(
        SimpleNamespace(arm=arm, enable_hands=hands), clock=clock,
        output_factory=lambda **kw: RealManOutput(**kw, factory=SDK, clock=clock.monotonic),
        input_factory=Pico, retargeter_factory=Retargeter)
    return state


@pytest.mark.parametrize('side', ['left', 'right', 'both'])
def test_real_ik_reaches_fake_sdk_and_reads_feedback_only_at_start(fake_rig, side):
    r = fake_rig
    r.sides = ('left', 'right') if side == 'both' else (side,)
    with pytest.raises(KeyboardInterrupt): r.run(side)
    moves = [e for e in r.events if e[0] == 'move']
    assert {e[2] for e in moves} == set(r.sides)
    for s in r.sides:
        sent = [e[3] for e in moves if e[2] == s]
        assert len(sent) == r.cycles
        np.testing.assert_allclose(sent[0], r.sdk_angles(s, r.initial[s]), atol=1e-12)
        assert np.max(np.abs(sent[-1] - sent[0])) > .01
    assert [(e[1], e[2]) for e in r.events if e[0] == 'read'] == [
        (0, selected_side) for _ in range(2) for selected_side in r.sides
    ]
    assert [e[2] for e in r.events if e[0] == 'home'] == list(r.sides)
    startup = [e[0] for e in r.events[:next(i for i, e in enumerate(r.events) if e[0] == 'move')]]
    assert startup == ['read'] * len(r.sides) + ['home', 'home-done'] * len(r.sides) + ['read'] * len(r.sides) + ['pico-start']
    assert {e[2] for e in r.events if e[0] == 'stop'} == set(r.sides)
    assert not any(e[0] == 'hand' for e in r.events)
    assert r.events[-1][0] == 'pico-stop'


def test_timeout_skips_only_arm_and_resumes_without_realigning(fake_rig):
    r = fake_rig
    r.work = .06
    r.missing = set(range(4, 10))
    with pytest.raises(KeyboardInterrupt): r.run(hands=True)
    assert not any(e[0] == 'move' and e[1] in (8, 9) for e in r.events)
    assert {e[2] for e in r.events if e[0] == 'move' and e[1] == 10} == {'left', 'right'}
    assert {e[2] for e in r.events if e[0] == 'hand' and e[1] == 9} == {'left', 'right'}
    assert all(e[1] == 13 for e in r.events if e[0] == 'stop')


def test_hand_has_master_channel_mapping_and_no_extra_rate_or_slew_limit(fake_rig):
    from utils.adapters.output.inspire_mapping import raw_qpos_to_channels
    from utils.robots.realman_rm75b import RealManRM75BInspireSpec
    r = fake_rig
    r.bad_hand = {3}
    with pytest.raises(KeyboardInterrupt): r.run(hands=True)
    for side in r.sides:
        hands = [e[3] for e in r.events if e[0] == 'hand' and e[2] == side]
        assert len(hands) == r.cycles
        expected = [round(v / 2) for v in raw_qpos_to_channels(
            np.full(12, .7), RealManRM75BInspireSpec(), scale=2000)]
        assert hands[0] == [1000] * 6
        assert hands[1] == expected
        assert hands[2] == expected  # Invalid retarget result retains previous target.
        assert max(abs(a-b) for a, b in zip(hands[0], hands[1])) > 20


@pytest.mark.parametrize('work', [.005, .03, .13])
def test_master_loop_pacing_without_custom_deadlines(fake_rig, work):
    r = fake_rig
    r.work = work
    with pytest.raises(KeyboardInterrupt): r.run()
    assert len([e for e in r.events if e[0] == 'move']) == 2 * r.cycles
    if work < .02:
        np.testing.assert_allclose(r.clock.sleeps, .02-work, atol=1e-12)
    else:
        assert r.clock.sleeps == []


def test_sdk_error_stops_both_without_retry(fake_rig):
    from utils.adapters.output.realman_real import RealManError
    r = fake_rig
    r.fail_sdk = True
    with pytest.raises(RealManError, match='SDK returned 9'): r.run()
    assert len([e for e in r.events if e[0] == 'move']) == 1
    assert {e[2] for e in r.events if e[0] == 'stop'} == set(r.sides)


def test_failed_startup_feedback_sends_no_actuator_commands(fake_rig):
    from utils.adapters.output.realman_real import RealManError
    r = fake_rig
    r.read_error = True
    with pytest.raises(RealManError, match='controller fault'): r.run()
    assert not any(e[0] in ('home', 'move', 'hand', 'stop') for e in r.events)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_failed_initial_pose_does_not_start_teleop_or_hands(fake_rig, side):
    from utils.adapters.output.realman_real import RealManError
    r = fake_rig
    r.fail_home = side
    with pytest.raises(RealManError, match=f'{side} initial Movej_Cmd: SDK returned 9'):
        r.run(hands=True)
    assert [e[2] for e in r.events if e[0] == 'home'] == (['left'] if side == 'left' else ['left', 'right'])
    assert not any(e[0] in ('move', 'hand', 'pico-start') for e in r.events)
    assert {e[2] for e in r.events if e[0] == 'stop'} == set(r.sides)
    assert r.events[-1][0] == 'uninit'


def test_bad_feedback_after_initial_pose_does_not_start_teleop(fake_rig):
    from utils.adapters.output.realman_real import RealManError
    r = fake_rig
    r.post_home_read_error = True
    with pytest.raises(RealManError, match='controller fault'):
        r.run(hands=True)
    assert [e[2] for e in r.events if e[0] == 'home-done'] == ['left', 'right']
    assert not any(e[0] in ('move', 'hand', 'pico-start') for e in r.events)
    assert {e[2] for e in r.events if e[0] == 'stop'} == set(r.sides)


def test_interrupted_initial_pose_stops_both_without_starting_teleop(fake_rig):
    r = fake_rig
    r.interrupt_home = 'left'
    with pytest.raises(KeyboardInterrupt):
        r.run(hands=True)
    assert [e[2] for e in r.events if e[0] == 'home'] == ['left']
    assert not any(e[0] in ('move', 'hand', 'pico-start') for e in r.events)
    assert {e[2] for e in r.events if e[0] == 'stop'} == set(r.sides)


def test_unavailable_hand_retargeter_prevents_initial_arm_motion(fake_rig):
    r = fake_rig
    r.hand_unavailable = True
    with pytest.raises(RuntimeError, match='hand retargeter unavailable'):
        r.run(hands=True)
    assert r.events == []


def test_side_specific_mixed_hand_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(entry, 'run', lambda args: calls.append(args))
    assert entry.main([
        '--arm', 'both', '--left-hand', 'gripper', '--right-hand', 'linker_l20',
        '--input-source', 'avp', '--avp-ip', '10.10.10.126',
    ]) == 0
    args = calls[0]
    assert args.arm == 'both'
    assert args.left_hand == 'gripper'
    assert args.right_hand == 'linker_l20'
    assert args.input_source == 'avp'
    assert args.avp_ip == '10.10.10.126'


def test_pinch_distance_maps_to_realman_gripper_position():
    assert entry._pinch_to_gripper_position(None) is None
    assert entry._pinch_to_gripper_position(0.02) == 1
    assert entry._pinch_to_gripper_position(0.06) == 500
    assert entry._pinch_to_gripper_position(0.10) == 1000
    assert entry._pinch_to_gripper_position(0.0) == 1
    assert entry._pinch_to_gripper_position(1.0) == 1000

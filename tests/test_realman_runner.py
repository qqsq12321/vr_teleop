import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from example import teleop_realman_real as runner
from utils.core.realman_input_stream import InputPollResult
from utils.core.realman_collision import RealManCollisionGuard
from utils.core.safety import SafetyFault


@pytest.fixture(scope='module')
def geometry():
    model, data, addresses, ranges = runner.load_model()
    guard = RealManCollisionGuard(runner.SPEC.scene_path)
    return model, data.qpos.copy(), addresses, guard


def test_collision_does_not_depend_on_disabled_scene_contacts(geometry):
    model, home, addresses, guard = geometry
    assert not np.any(model.geom_contype[model.geom_type == mujoco.mjtGeom.mjGEOM_MESH])
    guard.check(home)
    # Deterministic bounded random poses should contain self-collisions even
    # though the source scene's collision masks would report no contacts.
    rng = np.random.default_rng(42)
    found = False
    for _ in range(30):
        q = home.copy()
        for side, ids in addresses.items():
            jids = [model.joint(n).id for n in runner.SPEC.arm_joint_names(side)]
            ranges = model.jnt_range[jids]
            q[ids] = rng.uniform(ranges[:,0] + .1, ranges[:,1] - .1)
        try:
            guard.check(q)
        except SafetyFault as exc:
            assert 'Collision clearance' in str(exc)
            found = True
            break
    assert found


def test_finger_envelopes_enclose_all_sampled_articulations(geometry):
    _, home, _, guard = geometry
    model, data = guard.model, guard.data
    rng = np.random.default_rng(91)
    for _ in range(5):
        data.qpos[:] = home
        for side in ('left','right'):
            for name in runner.SPEC.hand_joint_names(side):
                j = model.joint(name).id
                data.qpos[model.jnt_qposadr[j]] = rng.uniform(*model.jnt_range[j])
        mujoco.mj_forward(model, data)
        for name, radius in guard.hand_radii.items():
            root = model.body(name).id
            for g in range(model.ngeom):
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                b = int(model.geom_bodyid[g])
                while b and b != root:
                    b = int(model.body_parentid[b])
                if b == root:
                    bound = np.linalg.norm(data.geom_xpos[g] - data.xpos[root]) + model.geom_rbound[g]
                    assert bound <= radius + 1e-9


def test_check_never_loads_vendor_sdk_or_opens_input(capsys):
    def forbidden(**kwargs):
        raise AssertionError('No hardware/input connections allowed')
    runner.run(runner.parse_args(['--check']), output_factory=forbidden, input_factory=forbidden)
    assert 'CHECK OK' in capsys.readouterr().out
    assert 'third_party.realman_api.robotic_arm' not in sys.modules


def test_dry_run_complete_ik_and_safety_loop_without_hardware(capsys):
    class FakeInput:
        closed = False
        def __init__(self, args):
            pass
        def poll_arms(self, left, right, converter):
            import time
            for arm in (left,right):
                arm.update_from_pose((0.3,0.3,1), (0,0,0,1))
            now = time.monotonic()
            return InputPollResult(True, wrist_received_at={'left':now, 'right':now})
        def close(self):
            FakeInput.closed = True
    def forbidden(**kwargs):
        raise AssertionError('No hardware connections allowed')
    runner.run(runner.parse_args(['--dry-run','--arm','both','--duration','0.2']),
               output_factory=forbidden, input_factory=FakeInput)
    text = capsys.readouterr().out
    assert 'Tracking aligned' in text
    assert 'command cycles=0.' not in text
    assert FakeInput.closed


def test_lost_tracking_exits_dry_run_and_closes_input():
    class FakeInput:
        closed = False
        def __init__(self, args):
            self.count = 0
        def poll_arms(self, left, right, converter):
            import time
            self.count += 1
            if self.count > 1:
                return InputPollResult(invalid_wrist_sides={'left'})
            for arm in (left,right):
                arm.update_from_pose((0.3,0.3,1), (0,0,0,1))
            now = time.monotonic()
            return InputPollResult(True, wrist_received_at={'left':now, 'right':now})
        def close(self):
            FakeInput.closed = True
    with pytest.raises(SafetyFault, match='left: wrist'):
        runner.run(runner.parse_args(['--duration','1']), input_factory=FakeInput)
    assert FakeInput.closed


@pytest.mark.parametrize('args', [['--duration','nan'], ['--duration','-1'],
    ['--read-only','--enable-motion'], ['--read-only','--enable-hands']])
def test_cli_rejects_unsafe_combinations(args):
    with pytest.raises(SystemExit):
        runner.parse_args(args)


def test_motion_runner_syncs_measured_pose_and_stops_both_on_input_loss(monkeypatch):
    import time
    from utils.adapters.output.realman_real import ArmFeedback

    class FakeOutput:
        instance = None
        def __init__(self, **network):
            FakeOutput.instance = self
            self.log = []
            self.q = {s: runner.SPEC.arm_home_qpos(s) + .01 for s in ('left','right')}
            self.initial = {s:q.copy() for s,q in self.q.items()}
        def connect(self):
            self.log.append('connect')
            return self.read_feedback()
        def read_feedback(self):
            return {s: ArmFeedback(q.copy(), time.monotonic()) for s,q in self.q.items()}
        def enable_motion(self):
            self.log.append('enable')
        def send_joints(self, side, q, *, guard, follow):
            guard(time.monotonic())
            self.log.append(('move', side, q.copy()))
            self.q[side] = q.copy()
        def close(self):
            self.log.extend(['stop-left','stop-right','close'])

    class FakeInput:
        closed = False
        def __init__(self, args):
            self.count = 0
        def poll_arms(self, left, right, _):
            self.count += 1
            if self.count > 5:
                return InputPollResult(stop_requested=True)
            if self.count > 3:
                return InputPollResult(invalid_wrist_sides={'left'})
            for arm in (left,right):
                # First fresh frame must start from measured joints, NOT home.
                np.testing.assert_allclose(arm.arm_home_qpos, FakeOutput.instance.initial[arm.side])
                arm.update_from_pose((.3,.3,1), (0,0,0,1))
            now = time.monotonic()
            return InputPollResult(True, wrist_received_at={'left':now,'right':now})
        def close(self):
            FakeInput.closed = True

    with pytest.raises(SafetyFault, match='Input requested stop'):
        runner.run(runner.parse_args(['--enable-motion','--arm','both','--duration','1']),
                   output_factory=FakeOutput, input_factory=FakeInput)
    out = FakeOutput.instance
    moves = [event for event in out.log if isinstance(event,tuple)]
    assert {event[1] for event in moves} == {'left','right'}
    for _,side,q in moves:
        np.testing.assert_allclose(q, out.initial[side], atol=1e-8)
    assert out.log[-3:] == ['stop-left','stop-right','close']
    assert FakeInput.closed


def test_read_only_runner_never_enables_or_uses_input(capsys):
    import time
    from utils.adapters.output.realman_real import ArmFeedback
    class ReadOnlyOutput:
        closed = False
        def __init__(self, **kwargs):
            pass
        def connect(self):
            return self.read_feedback()
        def read_feedback(self):
            return {s: ArmFeedback(np.zeros(7),time.monotonic()) for s in ('left','right')}
        def close(self):
            ReadOnlyOutput.closed = True
    def no_input(_):
        raise AssertionError('read-only should not start Pico')
    runner.run(runner.parse_args(['--read-only','--duration','.01']),
               output_factory=ReadOnlyOutput, input_factory=no_input)
    text = capsys.readouterr().out
    assert 'left: deg=' in text and 'right: deg=' in text
    assert ReadOnlyOutput.closed


def test_viewer_cleanup_even_when_input_close_raises():
    events = []
    class Input:
        def __init__(self, args): pass
        def close(self):
            events.append('input-close')
            raise RuntimeError('input cleanup failed')
    class Viewer:
        def __init__(self, path, qpos):
            assert path == runner.SPEC.scene_path
            assert np.all(np.isfinite(qpos))
        def is_running(self): return False
        def close(self): events.append('viewer-close')
    with pytest.raises(RuntimeError, match='input cleanup failed'):
        runner.run(runner.parse_args(['--dry-run', '--viewer']),
                   input_factory=Input, viewer_factory=Viewer)
    assert events == ['input-close', 'viewer-close']


def test_viewer_receives_only_pose_snapshots_without_cycle_deadline(monkeypatch):
    # A 130 ms interval with fresh tracking/feedback must no longer abort.
    events = []
    class Clock:
        now = 10.
        def monotonic(self): return self.now
        def sleep(self, value): self.now += value
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    class Input:
        def __init__(self, args): self.count = 0
        def poll_arms(self, left, right, _):
            self.count += 1
            if self.count == 2:
                clock.now += .11
            for arm in (left, right):
                arm.update_from_pose((.3,.3,1), (0,0,0,1))
            return InputPollResult(True, wrist_received_at={'left':clock.now, 'right':clock.now})
        def close(self): events.append('input-close')
    class Viewer:
        def __init__(self, path, qpos): pass
        def is_running(self): return True
        def publish(self, qpos):
            assert np.all(np.isfinite(qpos))
            events.append('publish')
            return False  # Dropped frames are harmless to control.
        def close(self): events.append('viewer-close')
    runner.run(runner.parse_args(['--dry-run','--viewer','--duration','.2']),
               input_factory=Input, viewer_factory=Viewer)
    assert events.count('publish') > 1
    assert events[-2:] == ['input-close', 'viewer-close']

@pytest.mark.parametrize('mode', ['--enable-motion', '--read-only', '--check'])
def test_tracking_pause_rejected_outside_dry_run(mode):
    with pytest.raises(SystemExit):
        runner.parse_args([mode, '--pause-on-tracking-loss'])
    args = runner.parse_args([mode])
    args.pause_on_tracking_loss = True
    with pytest.raises(ValueError, match='only allowed in offline dry-run'):
        runner.run(args)

@pytest.mark.parametrize('loss', ['invalid', 'timeout'])
def test_offline_pause_holds_pose_and_realigns_without_old_target(monkeypatch, capsys, caplog, loss):
    class Clock:
        now = 10.
        def monotonic(self): return self.now
        def sleep(self, value): self.now += value
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    snapshots = []
    class Viewer:
        def __init__(self, path, qpos): pass
        def is_running(self): return True
        def publish(self, qpos): snapshots.append(qpos.copy())
        def close(self): pass
    class Input:
        closed = False
        def __init__(self, args): self.count = 0
        def poll_arms(self, left, right, _):
            self.count += 1
            if self.count == 4:
                if loss == 'timeout':
                    clock.now += .26
                    return InputPollResult()
                return InputPollResult(invalid_wrist_sides={'left'})
            if self.count in (5,6):
                return InputPollResult()  # No cached alignment while waiting.
            # A large operator displacement while occluded must be rebaselined,
            # NOT interpreted as motion from the pre-loss target.
            pos = (.3,.3,1) if self.count < 4 else (1.3,1.3,2)
            for arm in (left,right): arm.update_from_pose(pos, (0,0,0,1))
            return InputPollResult(True, wrist_received_at={'left':clock.now,'right':clock.now})
        def close(self): Input.closed = True
    def forbidden(**kwargs): raise AssertionError('No robot connection allowed')
    runner.run(runner.parse_args(['--dry-run','--viewer','--duration','.5','--pause-on-tracking-loss']),
               input_factory=Input, output_factory=forbidden, viewer_factory=Viewer)
    assert Input.closed
    assert capsys.readouterr().out.count('Tracking aligned') == 2
    assert 'DRY RUN paused' in caplog.text
    assert ('input marked invalid' if loss == 'invalid' else 'no fresh sample for') in caplog.text
    assert len(snapshots) >= 9
    for pose in snapshots:
        np.testing.assert_allclose(pose, snapshots[0], atol=1e-8)


def test_offline_pause_does_not_suppress_nontracking_fault(monkeypatch):
    class Clock:
        now = 10.
        def monotonic(self): return self.now
        def sleep(self, value): self.now += value
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    class Input:
        def __init__(self, args): self.count = 0
        def poll_arms(self, left, right, _):
            self.count += 1
            if self.count == 2: clock.now += .21
            for arm in (left,right): arm.update_from_pose((.3,.3,1),(0,0,0,1))
            return InputPollResult(True, wrist_received_at={'left':clock.now,'right':clock.now})
        def close(self): pass
    with pytest.raises(SafetyFault, match='feedback timeout'):
        runner.run(runner.parse_args(['--dry-run','--pause-on-tracking-loss','--duration','1']),
                   input_factory=Input)


@pytest.mark.parametrize('work_s', [.005, .025])
def test_loop_matches_master_remaining_period_sleep(monkeypatch, work_s):
    class Clock:
        now = 10.
        sleeps = []
        def monotonic(self): return self.now
        def sleep(self, duration):
            self.sleeps.append(duration)
            self.now += duration
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    class Input:
        def __init__(self, args): pass
        def poll_arms(self, left, right, _):
            clock.now += work_s
            for arm in (left, right):
                arm.update_from_pose((.3, .3, 1), (0, 0, 0, 1))
            return InputPollResult(True, wrist_received_at={
                'left': clock.now, 'right': clock.now})
        def close(self): pass
    def forbidden(**kwargs):
        raise AssertionError('No hardware connections allowed')
    runner.run(runner.parse_args(['--dry-run', '--arm', 'both', '--duration', '.1']),
               input_factory=Input, output_factory=forbidden)
    if work_s < .02:
        assert clock.sleeps
        np.testing.assert_allclose(clock.sleeps, .02 - work_s, atol=1e-12)
    else:
        assert clock.sleeps == []

@pytest.mark.parametrize('loss', ['timeout', 'invalid_wrist', 'invalid_hand'])
def test_hardware_tracking_pause_skips_both_arms_and_hands_then_resumes(monkeypatch, loss):
    from utils.adapters.output.realman_real import ArmFeedback
    class Clock:
        now = 10.
        def monotonic(self): return self.now
        def sleep(self, duration): self.now += duration
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    events = []
    class Input:
        count = 0
        def __init__(self, args): pass
        def poll_arms(self, left, right, _):
            Input.count += 1
            if Input.count > 32:
                return InputPollResult(stop_requested=True)
            if 4 <= Input.count <= 28:
                if loss == 'invalid_wrist':
                    return InputPollResult(invalid_wrist_sides={'left'})
                if loss == 'invalid_hand':
                    return InputPollResult(invalid_hand_sides={'right'})
                return InputPollResult()
            for arm in (left, right):
                arm.update_from_pose((.3, .3, 1), (0, 0, 0, 1))
                arm.latest_hand_qpos = np.zeros(12)
            stamps = {'left': clock.now, 'right': clock.now}
            return InputPollResult(True, wrist_received_at=stamps, hand_received_at=stamps)
        def close(self): events.append(('input-close', Input.count))
    class Output:
        def __init__(self, **kwargs):
            self.q = {s: runner.SPEC.arm_home_qpos(s).copy() for s in ('left', 'right')}
        def connect(self): return self.read_feedback()
        def read_feedback(self):
            events.append(('read', Input.count))
            return {s: ArmFeedback(q.copy(), clock.now) for s, q in self.q.items()}
        def enable_motion(self): pass
        def configure_hands(self, config, *, guard): guard(clock.now)
        def send_joints(self, side, q, *, guard, follow):
            guard(clock.now)
            events.append(('arm', Input.count, side))
            self.q[side] = q.copy()
        def send_hand(self, side, channels, *, guard):
            guard(clock.now)
            events.append(('hand', Input.count, side))
        def close(self): events.append(('close', Input.count))
    with pytest.raises(SafetyFault, match='Input requested stop'):
        runner.run(runner.parse_args(['--enable-motion', '--arm', 'both', '--enable-hands']),
                   output_factory=Output, input_factory=Input)
    writes = [e for e in events if e[0] in ('arm', 'hand')]
    # Even timeout samples have expired by cycle 18; feedback keeps refreshing.
    assert not any(18 <= e[1] <= 28 for e in writes)
    assert any(e[0] == 'read' and 18 <= e[1] <= 28 for e in events)
    for kind in ('arm', 'hand'):
        assert {e[2] for e in writes if e[0] == kind and e[1] >= 29} == {'left', 'right'}
    assert [e for e in events if e[0] == 'close'] == [('close', 33)]

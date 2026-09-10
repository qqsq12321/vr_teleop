"""All transport tests use a fake SDK. Never connect a physical robot."""
from types import SimpleNamespace

import numpy as np
import pytest

from utils.adapters.output.inspire_mapping import raw_qpos_to_channels
from utils.adapters.output.realman_real import ArmFeedback, RealManError, RealManOutput
from utils.core.realman_input_stream import InputPollResult
from utils.core.safety import SafetyFault, SafetyLimits, SafetySupervisor
from utils.robots.realman_rm75b import RealManRM75BInspireSpec


class FakeArm:
    def __init__(self, ip, log):
        self.ip, self.log = ip, log
        self.code = 7
        self.error = []
        self.tag = 0
        self.q_deg = np.arange(7, dtype=float)

    def Arm_Socket_State(self):
        return self.tag

    def Get_Arm_Software_Info(self):
        return 0, SimpleNamespace(product_version=b"RM75-B")

    def Get_Current_Arm_State(self, retry=0):
        self.log.append((self.ip, "read"))
        return self.tag, self.q_deg, [0] * 6, self.error

    def Movej_Cmd(self, q, v, r, trajectory_connect, block):
        self.log.append((self.ip, "home", q, v, r, trajectory_connect, block))
        return self.tag

    def Movej_CANFD(self, q, follow):
        self.log.append((self.ip, "move", q, follow))
        return self.tag

    def Set_Hand_Angle(self, q, block):
        self.log.append((self.ip, "hand", q, block))
        return self.tag

    def Set_Modbus_Mode(self, *args):
        self.log.append((self.ip, "modbus_mode", args))
        return self.tag

    def Write_Registers(self, *args):
        self.log.append((self.ip, "registers", args))
        return self.tag

    def Move_Stop_Cmd(self, block):
        self.log.append((self.ip, "stop"))
        return self.tag

    def Arm_Socket_Close(self):
        self.log.append((self.ip, "close"))

    def RM_API_UnInit(self):
        self.log.append((self.ip, "uninit"))


@pytest.fixture
def sdk():
    log, arms = [], {}
    def factory(ip):
        arm = FakeArm(ip, log)
        arms[ip] = arm
        return arm
    output = RealManOutput("left", "right", factory=factory)
    try:
        yield output, arms, log
    finally:
        output.close()


def allow(_now):
    pass


def test_read_only_never_sends_motion_or_stop(sdk):
    out, _, log = sdk
    states = out.connect()
    np.testing.assert_allclose(states['left'].q_rad, np.deg2rad(np.arange(7)))
    for method in (out.send_joints, out.move_to_initial_pose):
        with pytest.raises(RealManError, match="Motion disabled"):
            method("left", np.zeros(7), guard=allow)
    out.close()
    assert [event[1] for event in log] == ['read', 'read', 'close', 'close', 'uninit']


def test_sdk_units_and_stop_both_before_global_uninit(sdk):
    out, _, log = sdk
    out.connect(); out.enable_motion()
    out.send_joints("left", np.full(7, np.pi / 2), guard=allow)
    move = next(event for event in log if event[1] == "move")
    assert move[3] is False
    np.testing.assert_allclose(move[2], np.full(7, 90))
    out.close()
    assert [event[1] for event in log][-5:] == ['stop', 'stop', 'close', 'close', 'uninit']


@pytest.mark.parametrize('method', ['send_joints', 'move_to_initial_pose'])
def test_sdk_stale_guard_prevents_write(sdk, method):
    out, _, log = sdk
    out.connect(); out.enable_motion()
    def expired(_):
        raise SafetyFault("expired")
    with pytest.raises(SafetyFault):
        getattr(out, method)("right", np.zeros(7), guard=expired)
    assert not any(event[1] in ("move", "home") for event in log)


@pytest.mark.parametrize("q", [[1]*6, [1]*8, [[1]*7], [float('nan')]*7, [float('inf')]*7])
@pytest.mark.parametrize('method', ['send_joints', 'move_to_initial_pose'])
def test_sdk_rejects_invalid_joints(sdk, q, method):
    out, _, log = sdk
    out.connect(); out.enable_motion()
    with pytest.raises(RealManError):
        getattr(out, method)("left", q, guard=allow)
    assert not any(event[1] in ("move", "home") for event in log)


@pytest.mark.parametrize('method', ['send_joints', 'move_to_initial_pose'])
def test_sdk_failure_latches_and_other_arm_still_stops(sdk, method):
    out, arms, log = sdk
    out.connect(); out.enable_motion()
    arms['left'].tag = "link failed"
    with pytest.raises(RealManError):
        getattr(out, method)("left", np.zeros(7), guard=allow)
    with pytest.raises(RealManError):
        out.send_joints("right", np.zeros(7), guard=allow)
    out.close()
    assert ('right', 'stop') in log
    assert ('right', 'close') in log


def test_partial_connection_cleanup():
    log = []
    def factory(ip):
        arm = FakeArm(ip, log)
        if ip == 'right':
            arm.code = 6
        return arm
    out = RealManOutput('left', 'right', factory=factory)
    with pytest.raises(RealManError, match='7-DOF'):
        out.connect()
    assert [e[1] for e in log] == ['close', 'close', 'uninit']
    assert not out._owns_api


def test_feedback_errors_fail_closed(sdk):
    out, arms, _ = sdk
    out.connect()
    arms['left'].error = ['0x0000', '0x0001']
    with pytest.raises(RealManError, match='controller fault'):
        out.read_feedback()
    with pytest.raises(RealManError, match='Latched fault'):
        out.enable_motion()


def test_exclusive_sdk_owner(sdk):
    out, _, _ = sdk
    out.connect()
    other = RealManOutput('a', 'b', factory=lambda ip: FakeArm(ip, []))
    with pytest.raises(RealManError, match='Only one'):
        other.connect()
    other.close()
    out.read_feedback()


@pytest.mark.parametrize('backend', ['native', 'modbus'])
def test_hand_native_and_modbus_wire_format(sdk, backend):
    out, _, log = sdk
    out.connect(); out.enable_motion()
    out.configure_hands(dict(backend=backend, sides=['left'], baudrate=115200,
                             timeout_100ms=1, address=1486, device_id=1), guard=allow)
    values = [0, 1, 255, 256, 999, 1000]
    out.send_hand('left', values, guard=allow)
    if backend == 'native':
        assert ('left', 'hand', values, False) in log
        assert not any(e[1] == 'modbus_mode' for e in log)
    else:
        assert ('left', 'registers', (1, 1486, 6, [0,0, 0,1, 0,255, 1,0, 3,231, 3,232], 1, False)) in log
    with pytest.raises(RealManError):
        out.send_hand('right', values, guard=allow)
    with pytest.raises(ValueError):
        out.send_hand('left', [1001]*6, guard=allow)


def test_inspire_raw_order_not_mujoco_order():
    spec = RealManRM75BInspireSpec()
    assert raw_qpos_to_channels(np.zeros(12), spec) == [1000]*6
    q = np.zeros(12)
    for index, maximum in zip(spec.inspire_channel_indices, spec.inspire_channel_max_rad):
        q[index] = maximum
    assert raw_qpos_to_channels(q, spec) == [0]*6
    q = np.zeros(12); q[4] = 1.47  # raw pinky -> hardware little finger
    assert raw_qpos_to_channels(q, spec) == [0,1000,1000,1000,1000,1000]
    with pytest.raises(ValueError):
        raw_qpos_to_channels(q[spec.hand_qpos_mapping, None], spec)
    with pytest.raises(ValueError):
        raw_qpos_to_channels(np.full(12, np.nan), spec)


def make_supervisor(*, hands=False, **kwargs):
    limits = SafetyLimits(**kwargs)
    ranges = {s: np.tile([-3.0, 3.0], (7, 1)) for s in ('left', 'right')}
    safety = SafetySupervisor(limits, ranges, ('left', 'right'), hands=hands)
    refresh(safety, 10)
    safety.start(10)
    return safety


def refresh(safety, now, q=None):
    safety.observe_input(InputPollResult(wrist_received_at={'left': now, 'right': now},
                                        hand_received_at={'left': now, 'right': now}))
    safety.observe_feedback({s: ArmFeedback(np.zeros(7) if q is None else q, now) for s in ('left', 'right')}, now)


@pytest.mark.parametrize('elapsed', [.005, .02, .1257, .5])
def test_validated_targets_are_exact_ik_angles_regardless_of_cycle_time(elapsed):
    safety = make_supervisor()
    now = 10 + elapsed
    refresh(safety, now)
    targets = {'left': np.linspace(-1, 1, 7), 'right': np.linspace(1, -1, 7)}
    commands = safety.validate_targets(targets, now)
    for side in targets:
        np.testing.assert_array_equal(commands[side], targets[side])
        assert not np.shares_memory(commands[side], targets[side])
        assert not np.shares_memory(commands[side], safety.q_command[side])

def test_one_fresh_side_cannot_refresh_stale_other():
    safety = make_supervisor()
    safety.observe_input(InputPollResult(wrist_received_at={'right': 10.26}))
    with pytest.raises(SafetyFault, match='left: wrist'):
        safety.check_live(10.26)
    refresh(safety, 10.27)
    with pytest.raises(SafetyFault):
        safety.check_live(10.27)  # no automatic recovery


def test_hand_landmarks_do_not_refresh_wrist():
    safety = make_supervisor(hands=True)
    safety.observe_input(InputPollResult(hand_received_at={'left':10.26, 'right':10.26}))
    with pytest.raises(SafetyFault, match='wrist'):
        safety.check_live(10.26)


def test_explicit_inactive_side_stops_immediately():
    safety = make_supervisor()
    safety.observe_input(InputPollResult(invalid_wrist_sides={'left'}))
    with pytest.raises(SafetyFault, match='left: wrist'):
        safety.check_live(10.001)


def test_feedback_latency_and_command_expiration():
    safety = make_supervisor()
    with pytest.raises(SafetyFault, match='Command expired'):
        safety.check_live(10.09, generated_at=10)
    safety = make_supervisor()
    with pytest.raises(SafetyFault, match='feedback timeout'):
        safety.observe_feedback({s: ArmFeedback(np.zeros(7), 9.7) for s in ('left','right')}, 10)


def test_tracking_error_grace_then_latch():
    safety = make_supervisor()
    refresh(safety, 10.01, q=np.ones(7))
    with pytest.raises(SafetyFault, match='tracking error'):
        refresh(safety, 10.22, q=np.ones(7))


def test_no_cycle_deadline_and_soft_limits():
    safety = make_supervisor()
    safety.check_live(10.1257)  # Fresh input/feedback; no 100 ms cycle limit.
    refresh(safety, 10.5)
    q = safety.validate_targets({s: np.ones(7) for s in ('left', 'right')}, 10.5)
    np.testing.assert_array_equal(q['left'], np.ones(7))
    safety = make_supervisor()
    with pytest.raises(SafetyFault, match='soft limit'):
        safety.validate_targets({s: np.full(7, 3) for s in ('left','right')}, 10.02)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '50'])
def test_invalid_safety_config_rejected(value):
    with pytest.raises(ValueError):
        SafetyLimits(rate_hz=value)

@pytest.mark.parametrize('hand', [False, True])
def test_tracking_guard_skips_sdk_write_without_latching_transport(sdk, hand):
    from utils.core.safety import InputTrackingFault
    out, _, log = sdk
    out.connect(); out.enable_motion()
    out.configure_hands(dict(backend='native', sides=('left','right')), guard=allow)
    def stale(_): raise InputTrackingFault('tracking paused')
    send = (lambda guard: out.send_hand('right', [500]*6, guard=guard)) if hand else (
        lambda guard: out.send_joints('right', np.zeros(7), guard=guard))
    before = len(log)
    with pytest.raises(InputTrackingFault): send(stale)
    assert len(log) == before
    send(allow)
    assert len(log) == before + 1


@pytest.mark.parametrize('side', ['left', 'right'])
def test_planned_initial_pose_uses_seven_degrees_and_master_movej_arguments(sdk, side):
    out, _, log = sdk
    out.connect()
    out.enable_motion()
    q = RealManRM75BInspireSpec().arm_home_qpos(side)
    out.move_to_initial_pose(side, q, guard=allow)
    moves = [e for e in log if e[1] == 'home']
    assert len(moves) == 1 and moves[0][0] == side
    np.testing.assert_array_equal(moves[0][2], np.rad2deg(q))
    assert moves[0][3:] == (20, 0, 0, True)
    assert not any(e[1] in ('move', 'hand') for e in log)

@pytest.mark.parametrize('offset', [-180.0, 0.0, 180.0])
def test_right_j1_calibration_startup_stream_feedback(offset):
    log, arms = [], {}
    def factory(ip):
        arms[ip] = FakeArm(ip, log)
        return arms[ip]
    out = RealManOutput('left', 'right', factory=factory, right_j1_offset_deg=offset)
    q = np.array([.47, 0, .5, 2.04, -.62, -1.09, .59])
    original = q.copy()
    try:
        out.connect()
        out.enable_motion()
        for side in ('left', 'right'):
            expected = np.rad2deg(q)
            if side == 'right':
                expected[0] += offset
            out.move_to_initial_pose(side, q, guard=allow)
            out.send_joints(side, q, guard=allow, follow=True)
            for event in log:
                if event[0] == side and event[1] in ('home', 'move'):
                    np.testing.assert_allclose(event[2], expected)
            arms[side].q_deg = expected.copy()
            np.testing.assert_allclose(out.read_feedback()[side].q_rad, q, atol=1e-14)
            np.testing.assert_array_equal(arms[side].q_deg, expected)
        np.testing.assert_array_equal(q, original)
    finally:
        out.close()

@pytest.mark.parametrize('offset', [float('nan'), float('inf')])
def test_reject_nonfinite_right_j1_offset(offset):
    with pytest.raises(ValueError, match='finite'):
        RealManOutput('left', 'right', right_j1_offset_deg=offset)

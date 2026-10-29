"""Offline tests for the single-arm RM75-B real-robot path.

A fake RM_API2-shaped handle is injected through ``factory=``; the vendor SDK
is never imported and no socket is ever opened.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from utils.adapters.output.realman_rm75b_single_real import (
    ARM_DOF,
    RM75BSingleOutput,
    RealManError,
)
from utils.core.input_stream import InputPollResult
from utils.core.safety import InputTrackingFault

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "teleop_rm75b_single_real", ROOT / "example" / "teleop_rm75b_single_real.py"
)
entry = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(entry)

MEASURED_DEG = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


class FakeHandle:
    def __init__(self, handle_id=1):
        self.id = handle_id


class FakeArm:
    """Mimics Robotic_Arm.rm_robot_interface.RoboticArm, logging every call."""

    def __init__(self, log, *, arm_dof=ARM_DOF, handle_id=1, fail=None):
        self.log = log
        self.arm_dof = arm_dof
        self._handle_id = handle_id
        self._fail = fail or {}

    def _code(self, verb):
        return self._fail.get(verb, 0)

    def rm_create_robot_arm(self, ip, port, level=3, log_func=None):
        self.log.append((ip, "create", port))
        return FakeHandle(self._handle_id)

    def rm_get_robot_info(self):
        self.log.append((None, "info", None))
        return self._code("info"), {"arm_model": "RM75-B", "robot_controller_version": 4}

    def rm_get_joint_degree(self):
        self.log.append((None, "read", None))
        return self._code("read"), list(MEASURED_DEG)

    def rm_movej(self, joint, v, r, connect, block):
        self.log.append((None, "movej", tuple(joint)))
        return self._code("movej")

    def rm_movej_canfd(self, joint, follow, expand=0, trajectory_mode=0, radio=0):
        self.log.append((None, "canfd", tuple(joint)))
        return self._code("canfd")

    def rm_set_arm_stop(self):
        self.log.append((None, "stop", None))
        return self._code("stop")

    def rm_delete_robot_arm(self):
        self.log.append((None, "delete", None))
        return 0

    def rm_destroy(self):
        self.log.append((None, "destroy", None))
        return 0


def allow(_now):
    pass


@pytest.fixture
def rig():
    log = []
    arm_box = {}

    def factory(ip, **kwargs):
        arm = FakeArm(log, **kwargs)
        arm_box["arm"] = arm
        return arm

    def build(**kwargs):
        fail = kwargs.pop("fail", None)
        arm_dof = kwargs.pop("arm_dof", ARM_DOF)
        handle_id = kwargs.pop("handle_id", 1)
        return RM75BSingleOutput(
            "192.168.1.18",
            factory=lambda ip: factory(ip, fail=fail, arm_dof=arm_dof, handle_id=handle_id),
            **kwargs,
        )

    yield build, log


def _verbs(log):
    return [entry_[1] for entry_ in log]


# -- connection ------------------------------------------------------------


def test_connect_uses_only_the_configured_address(rig):
    build, log = rig
    out = build()
    try:
        feedback = out.connect()
    finally:
        out.close()
    assert [e[0] for e in log if e[0] is not None] == ["192.168.1.18"]
    np.testing.assert_allclose(feedback.q_rad, np.deg2rad(MEASURED_DEG))


def test_non_seven_dof_arm_is_rejected(rig):
    build, log = rig
    out = build(arm_dof=6)
    with pytest.raises(RealManError, match="7-DoF"):
        out.connect()
    # The failed connect must still tear the handle down.
    assert "delete" in _verbs(log) and "destroy" in _verbs(log)


def test_failed_handle_is_rejected(rig):
    build, _log = rig
    out = build(handle_id=-1)
    with pytest.raises(RealManError, match="Could not connect"):
        out.connect()


def test_second_adapter_cannot_own_the_sdk(rig):
    build, _log = rig
    first = build()
    try:
        first.connect()
        second = build()
        with pytest.raises(RealManError, match="owns the SDK"):
            second.connect()
    finally:
        first.close()


# -- read-only safety ------------------------------------------------------


def test_read_only_never_sends_motion_or_stop(rig):
    build, log = rig
    out = build()
    try:
        out.connect()
        for _ in range(3):
            out.read_feedback()
    finally:
        out.close()
    verbs = _verbs(log)
    assert "movej" not in verbs
    assert "canfd" not in verbs
    # stop() is a no-op while motion was never enabled.
    assert "stop" not in verbs
    assert verbs[-2:] == ["delete", "destroy"]


def test_motion_requires_explicit_enable(rig):
    build, _log = rig
    out = build()
    try:
        out.connect()
        with pytest.raises(RealManError, match="Motion disabled"):
            out.send_joints(np.zeros(7), guard=allow)
    finally:
        out.close()


# -- units -----------------------------------------------------------------


def test_degree_conversion_round_trips(rig):
    build, _log = rig
    out = build()
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    degrees = out.model_to_sdk_degrees(q)
    np.testing.assert_allclose(out.sdk_degrees_to_model(degrees), q, atol=1e-12)
    # A floor-mounted arm needs no per-joint offset: model degrees go to the
    # controller unchanged.
    np.testing.assert_allclose(degrees, np.rad2deg(q))


def test_feedback_is_reported_in_model_radians(rig):
    build, _log = rig
    out = build()
    try:
        feedback = out.connect()
    finally:
        out.close()
    np.testing.assert_allclose(feedback.q_rad, np.deg2rad(MEASURED_DEG))


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(7, np.nan), np.zeros((2, 7))])
def test_invalid_joint_vectors_are_rejected(rig, bad):
    build, _log = rig
    out = build()
    try:
        out.connect()
        out.enable_motion()
        with pytest.raises(RealManError):
            out.send_joints(bad, guard=allow)
    finally:
        out.close()


# -- guard protocol --------------------------------------------------------


def test_tracking_guard_skips_the_write_without_latching(rig):
    build, log = rig
    out = build()

    def stale(_now):
        raise InputTrackingFault("stale input")

    try:
        out.connect()
        out.enable_motion()
        with pytest.raises(InputTrackingFault):
            out.send_joints(np.zeros(7), guard=stale)
        assert "canfd" not in _verbs(log)
        # Not latched: a fresh frame resumes the same session.
        out.send_joints(np.zeros(7), guard=allow)
        assert "canfd" in _verbs(log)
    finally:
        out.close()


def test_other_guard_errors_latch_the_fault(rig):
    build, log = rig
    out = build()

    def boom(_now):
        raise RuntimeError("host deadline missed")

    try:
        out.connect()
        out.enable_motion()
        with pytest.raises(RuntimeError, match="host deadline"):
            out.send_joints(np.zeros(7), guard=boom)
        assert "canfd" not in _verbs(log)
        with pytest.raises(RealManError, match="Motion disabled or faulted"):
            out.send_joints(np.zeros(7), guard=allow)
    finally:
        out.close()


# -- failure handling ------------------------------------------------------


@pytest.mark.parametrize("verb", ["canfd", "movej"])
def test_nonzero_return_code_latches_and_blocks_later_writes(rig, verb):
    build, _log = rig
    out = build(fail={verb: 1})
    try:
        out.connect()
        out.enable_motion()
        call = out.send_joints if verb == "canfd" else out.move_to_initial_pose
        with pytest.raises(RealManError, match="RM_API2 returned 1"):
            call(np.zeros(7), guard=allow)
        with pytest.raises(RealManError, match="Motion disabled or faulted"):
            out.send_joints(np.zeros(7), guard=allow)
    finally:
        out.close()


def test_feedback_failure_fails_closed(rig):
    build, _log = rig
    out = build(fail={"read": -2})
    with pytest.raises(RealManError, match="joint feedback"):
        out.connect()


def test_close_order_is_stop_then_delete_then_destroy(rig):
    build, log = rig
    out = build()
    out.connect()
    out.enable_motion()
    out.send_joints(np.zeros(7), guard=allow)
    out.close()
    verbs = _verbs(log)
    assert verbs[-3:] == ["stop", "delete", "destroy"]


def test_close_is_idempotent_and_releases_the_sdk(rig):
    build, _log = rig
    out = build()
    out.connect()
    out.close()
    out.close()
    # The lock is free, so a fresh adapter can take over.
    other = build()
    try:
        other.connect()
    finally:
        other.close()


# -- entrypoint ------------------------------------------------------------


def test_check_is_the_default_mode_and_opens_no_sockets(capsys):
    assert entry.main([]) == 0
    out = capsys.readouterr().out
    assert "No sockets were opened" in out
    assert "Check passed" in out


def test_check_does_not_import_the_vendor_sdk():
    """--check must stay usable on a machine without the RM_API2 SDK."""
    import subprocess
    import sys as _sys

    probe = (
        "import runpy, sys\n"
        "sys.argv = ['teleop_rm75b_single_real.py', '--check']\n"
        "try:\n"
        "    runpy.run_path('example/teleop_rm75b_single_real.py', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('SDK_LOADED' if 'Robotic_Arm' in sys.modules else 'SDK_ABSENT')\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True
    )
    assert "SDK_ABSENT" in result.stdout, result.stdout + result.stderr


def test_modes_are_mutually_exclusive():
    for pair in (
        ["--check", "--read-only"],
        ["--check", "--enable-motion"],
        ["--read-only", "--enable-motion"],
    ):
        with pytest.raises(SystemExit):
            entry.main(pair)


def test_dual_arm_config_is_rejected():
    with pytest.raises(ValueError, match="exactly right_ip"):
        entry.load_config(ROOT / "config" / "realman_real.yaml")


def test_single_arm_config_loads():
    config = entry.load_config(ROOT / "config" / "realman_rm75b_single_real.yaml")
    assert set(config["network"]) == {"right_ip"}


def test_read_only_entrypoint_sends_no_motion():
    log = []

    def output_factory(ip, **kwargs):
        return RM75BSingleOutput(ip, factory=lambda _ip: FakeArm(log), **kwargs)

    args = entry.argparse.Namespace(
        config=ROOT / "config" / "realman_rm75b_single_real.yaml",
        duration=0.01,
    )
    assert entry.run_read_only(args, output_factory=output_factory) == 0
    verbs = _verbs(log)
    assert "movej" not in verbs and "canfd" not in verbs and "stop" not in verbs


# -- startup sequencing ----------------------------------------------------


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _FakeStream:
    """Yields no tracking data until `ready_after` polls have gone by."""

    def __init__(self, ready_after=0, stop_at=None):
        self.ready_after = ready_after
        self.stop_at = stop_at
        self.polls = 0

    def poll_arms(self, _left, _right, _converter, allow_stop_gesture=False):
        self.polls += 1
        stop = self.stop_at is not None and self.polls >= self.stop_at
        return InputPollResult(
            saw_valid_data=self.polls > self.ready_after, stop_requested=stop
        )


def test_wait_blocks_until_tracking_data_arrives():
    """Everything else is ready; the countdown must not run against dead air."""
    clock = _FakeClock()
    stream = _FakeStream(ready_after=5)
    entry._wait_for_tracking(stream, object(), clock=clock, emit=lambda _m: None)
    assert stream.polls == 6  # Polled until the first valid frame.


def test_wait_honours_the_stop_gesture():
    clock = _FakeClock()
    stream = _FakeStream(ready_after=100, stop_at=3)
    with pytest.raises(KeyboardInterrupt):
        entry._wait_for_tracking(stream, object(), clock=clock, emit=lambda _m: None)


def test_wait_times_out_when_the_headset_never_reports():
    clock = _FakeClock()
    stream = _FakeStream(ready_after=10**9)
    with pytest.raises(TimeoutError, match="No tracking data"):
        entry._wait_for_tracking(
            stream, object(), clock=clock, emit=lambda _m: None, timeout_s=1.0
        )


def test_countdown_counts_three_two_one_then_anchors():
    spoken, slept = [], []
    entry._countdown(3, sleep=slept.append, emit=spoken.append)
    assert spoken[:3] == ["3...", "2...", "1..."]
    assert "Anchoring now" in spoken[3]
    assert slept == [1.0, 1.0, 1.0]


def test_arm_state_hand_hooks_are_inert():
    """No end effector: hand data must never look like valid tracking."""
    state = entry._ArmState(
        tracker=None, site_id=0, ik_data=None,
        cmd_pos=None, cmd_quat=None, home=None,
    )
    assert state.update_hand_from_pinch(0.02) is False
    assert state.update_hand_from_mediapipe(object()) is False
    assert state.update_hand_from_raw_landmarks(object()) is False


# -- heading alignment -----------------------------------------------------


def _quat_z(degrees):
    half = np.deg2rad(degrees) / 2.0
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


class _RecordingTracker:
    def __init__(self):
        self.samples = []

    def update(self, position, quaternion):
        self.samples.append((np.asarray(position, dtype=float), tuple(quaternion)))


def _state_with_heading(yaw_deg):
    state = entry._ArmState(
        tracker=_RecordingTracker(), site_id=0, ik_data=None,
        cmd_pos=None, cmd_quat=None, home=None,
    )
    state.set_heading(None if yaw_deg is None else np.deg2rad(yaw_deg))
    return state


@pytest.mark.parametrize("heading_deg", [0.0, 45.0, 90.0, -30.0, 180.0])
def test_forward_push_maps_forward_from_any_heading(heading_deg):
    """The operator may stand anywhere; "forward" must stay forward."""
    state = _state_with_heading(heading_deg)
    # The same subjective forward push, expressed in the VR world frame for
    # an operator facing heading_deg.
    rotation = entry._yaw_matrix(np.deg2rad(heading_deg))
    push = rotation @ np.array([0.1, 0.0, 0.0])
    state.update_from_pose(tuple(push), (0.0, 0.0, 0.0, 1.0))
    corrected, _quat = state.tracker.samples[0]
    np.testing.assert_allclose(corrected, [0.1, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("heading_deg", [0.0, 45.0, 90.0, -30.0])
def test_wrist_rotation_maps_identically_from_any_heading(heading_deg):
    """Rotation is left-multiplied, so it needs the same correction."""
    from utils.core.quaternion import quaternion_multiply

    state = _state_with_heading(heading_deg)
    # A wrist held at the same orientation relative to the operator's body.
    body = _quat_z(heading_deg)
    state.update_from_pose((0.0, 0.0, 0.0), quaternion_multiply(body, _quat_z(20.0)))
    _position, corrected = state.tracker.samples[0]
    expected = _quat_z(20.0)
    # Quaternions are double-covered, so compare up to sign.
    dot = abs(float(np.dot(np.asarray(corrected), np.asarray(expected))))
    assert dot == pytest.approx(1.0, abs=1e-9)


def test_without_a_heading_samples_pass_through_unchanged():
    state = _state_with_heading(None)
    position = (0.1, -0.2, 0.3)
    quaternion = _quat_z(33.0)
    state.update_from_pose(position, quaternion)
    got_position, got_quaternion = state.tracker.samples[0]
    np.testing.assert_allclose(got_position, position)
    np.testing.assert_allclose(got_quaternion, quaternion)


def test_heading_ignores_pitch_so_looking_down_does_not_roll_the_frame():
    """Only yaw matters; glancing at the bench must not tilt the mapping."""
    half = np.deg2rad(40.0) / 2.0
    pitched = [0.0, 0.0, 0.0, float(np.sin(half)), 0.0, 0.0, float(np.cos(half))]
    # A pure pitch in the Pico frame must not register as a heading change.
    level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert entry._heading_yaw_rad(pitched) == pytest.approx(
        entry._heading_yaw_rad(level), abs=1e-9
    )


def test_missing_head_pose_yields_no_heading():
    assert entry._heading_yaw_rad(None) is None
    assert entry._heading_yaw_rad([0.0, 0.0]) is None

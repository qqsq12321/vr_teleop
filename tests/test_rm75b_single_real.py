"""Offline tests for the single-arm RM75-B real-robot path.

A fake RM_API2-shaped handle is injected through ``factory=``; the vendor SDK
is never imported and no socket is ever opened.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    """Mimics Robotic_Arm.rm_robot_interface.RoboticArm, logging every call.

    Deliberately has no rm_set_tool_voltage: calling it would raise, which is
    how the tests prove the adapter never changes the tool supply.
    """

    def __init__(self, log, *, arm_dof=ARM_DOF, handle_id=1, fail=None,
                 version=4, voltage=3, hand_read=None):
        self.log = log
        self.arm_dof = arm_dof
        self._handle_id = handle_id
        self._fail = fail or {}
        self._version = version
        self._voltage = voltage
        self._hand_read = hand_read

    def _code(self, verb):
        return self._fail.get(verb, 0)

    def rm_create_robot_arm(self, ip, port, level=3, log_func=None):
        self.log.append((ip, "create", port))
        return FakeHandle(self._handle_id)

    def rm_get_robot_info(self):
        self.log.append((None, "info", None))
        return self._code("info"), {"arm_model": "RM75-B", "robot_controller_version": self._version}

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

    # -- tool port --
    def rm_get_tool_voltage(self):
        self.log.append((None, "voltage", None))
        return self._code("voltage"), self._voltage

    def rm_set_tool_rs485_mode(self, mode, baudrate):
        self.log.append((None, "rs485_mode", (mode, baudrate)))
        return self._code("rs485_mode")

    def rm_set_modbus_mode(self, port, baudrate, timeout):
        self.log.append((None, "modbus_mode", (port, baudrate, timeout)))
        return self._code("modbus_mode")

    def rm_close_modbus_mode(self, port):
        self.log.append((None, "modbus_close", port))
        return 0

    def rm_set_hand_angle(self, angles, block, timeout):
        self.log.append((None, "hand_angle", (tuple(angles), block)))
        return self._code("hand_angle")

    def rm_write_modbus_rtu_registers(self, params):
        self.log.append((None, "rtu_write", params))
        return self._code("rtu_write")

    def rm_write_registers(self, params, data):
        self.log.append((None, "v3_write", (params, tuple(data))))
        return self._code("v3_write")

    def rm_read_modbus_rtu_holding_registers(self, params):
        self.log.append((None, "rtu_read", params))
        return self._code("rtu_read"), list(self._hand_read or [0] * params.num)

    def rm_read_multiple_holding_registers(self, params):
        self.log.append((None, "v3_read", params))
        return self._code("v3_read"), list(self._hand_read or [0] * (params.num * 2))


class FakeTypes:
    """Stand-ins for the RM_API2 ctypes parameter structs, in their arg order."""

    @staticmethod
    def rm_modbus_rtu_write_params_t(address, device, type, num, data):
        return SimpleNamespace(address=address, device=device, type=type, num=num, data=list(data))

    @staticmethod
    def rm_modbus_rtu_read_params_t(address, device, type, num):
        return SimpleNamespace(address=address, device=device, type=type, num=num)

    @staticmethod
    def rm_peripheral_read_write_params_t(port, address, device, num):
        return SimpleNamespace(port=port, address=address, device=device, num=num)


def allow(_now):
    pass


@pytest.fixture(autouse=True)
def _no_real_hardware(monkeypatch):
    """Fail any test that would build a real RM_API2 handle.

    A test once called main() with arguments that passed validation, which ran
    the real entrypoint against the real arm. Every adapter here must be given
    a fake factory; one without it now fails instead of connecting.
    """
    original = RM75BSingleOutput._build_arm

    def guarded(self):
        if self._factory is None:
            raise RuntimeError("Test tried to reach real hardware; inject a fake factory")
        return original(self)

    monkeypatch.setattr(RM75BSingleOutput, "_build_arm", guarded)


@pytest.fixture
def rig():
    log = []
    arm_box = {}

    def factory(ip, **kwargs):
        arm = FakeArm(log, **kwargs)
        arm_box["arm"] = arm
        return arm

    def build(**kwargs):
        arm_kwargs = {
            key: kwargs.pop(key)
            for key in ("fail", "arm_dof", "handle_id", "version", "voltage", "hand_read")
            if key in kwargs
        }
        return RM75BSingleOutput(
            "192.168.1.18",
            factory=lambda ip: factory(ip, **arm_kwargs),
            sdk_types=FakeTypes,
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


# -- controller clutch -----------------------------------------------------


def test_clutch_starts_released():
    """The arm must hold still until the operator squeezes."""
    clutch = entry._GripClutch()
    assert clutch.engaged is False
    assert clutch.update(0.0) == (False, False)
    assert clutch.update(None) == (False, False)


def test_squeeze_engages_once_and_release_disengages():
    clutch = entry._GripClutch()
    # Rising edge re-anchors; holding does not re-anchor every frame.
    assert clutch.update(0.9) == (True, True)
    assert clutch.update(0.9) == (True, False)
    assert clutch.update(0.95) == (True, False)
    # Release stops tracking without re-anchoring.
    assert clutch.update(0.0) == (False, False)
    # Squeezing again re-anchors, so a drifted mapping can be fixed in place.
    assert clutch.update(0.8) == (True, True)


def test_clutch_ignores_travel_below_the_engage_threshold():
    clutch = entry._GripClutch(engage=0.1)
    assert clutch.update(0.05) == (False, False)
    assert clutch.update(0.1) == (False, False)  # Exactly at the threshold.
    assert clutch.update(0.11) == (True, True)


def test_clutch_treats_unusable_grip_values_as_released():
    clutch = entry._GripClutch()
    assert clutch.update("nonsense") == (False, False)
    assert clutch.update(0.9) == (True, True)
    # A bad reading must release rather than latch tracking on.
    assert clutch.update(None) == (False, False)


def test_controller_state_is_none_without_a_pico():
    class _NoPico:
        pico4_input = None

    assert entry._controller_state(_NoPico()) is None
    assert entry._controller_state(object()) is None


def test_controller_state_reads_the_requested_side():
    seen = []

    class _Pico:
        def get_controller(self, side):
            seen.append(side)
            return {"grip": 0.5, "pose": [0.0] * 7}

    class _Stream:
        pico4_input = _Pico()

    state = entry._controller_state(_Stream())
    assert state["grip"] == 0.5
    # The lone arm is addressed as the right side throughout the stack.
    assert seen == ["right"]


def test_controller_requires_pico():
    with pytest.raises(SystemExit):
        entry.main(["--enable-motion", "--controller", "--input-source", "avp",
                    "--avp-ip", "1.2.3.4"])


# -- hand on the tool port --------------------------------------------------

from utils.adapters.output.rohand_mapping import raw_qpos_to_positions  # noqa: E402
from utils.robots.realman_rm75b_single import RealManRM75BSingleROHandSpec  # noqa: E402

SIX = [0, 1, 255, 256, 999, 1000]
ROH_SIX = [0, 1, 255, 256, 40000, 65535]


def _connected(build, **kwargs):
    out = build(**kwargs)
    out.connect()
    return out


def _calls(log, verb):
    return [entry_[2] for entry_ in log if entry_[1] == verb]


def test_unpowered_tool_port_is_refused_and_left_alone(rig):
    build, log = rig
    out = _connected(build, voltage=0)
    try:
        with pytest.raises(RealManError, match="0V"):
            out.configure_hand("rohand")
    finally:
        out.close()
    # Refused before touching the port, and the supply is never switched on.
    for verb in ("rs485_mode", "modbus_mode", "rtu_write", "v3_write", "hand_angle"):
        assert _calls(log, verb) == []


def test_unknown_controller_generation_is_refused(rig):
    build, _log = rig
    out = _connected(build, version=None)
    try:
        with pytest.raises(RealManError, match="generation"):
            out.configure_hand("inspire")
    finally:
        out.close()


@pytest.mark.parametrize("version", [3, 4])
def test_inspire_wire_format(rig, version):
    build, log = rig
    out = _connected(build, version=version)
    try:
        out.configure_hand("inspire", baudrate=115200)
        out.enable_hand()
        out.send_hand(SIX, guard=allow)
    finally:
        out.close()
    # Fourth generation must switch the port to its built-in hand protocol;
    # the third generation speaks it by default.
    assert _calls(log, "rs485_mode") == ([(1, 115200)] if version == 4 else [])
    assert _calls(log, "hand_angle") == [(tuple(SIX), False)]
    assert _calls(log, "modbus_mode") == []


def test_rohand_fourth_generation_wire_format(rig):
    build, log = rig
    out = _connected(build, version=4)
    try:
        out.configure_hand("rohand", baudrate=115200, device_id=2)
        out.enable_hand()
        out.send_hand(ROH_SIX, guard=allow)
    finally:
        out.close()
    assert _calls(log, "rs485_mode") == [(0, 115200)]  # Tool port as RTU master.
    (params,) = _calls(log, "rtu_write")
    assert (params.address, params.device, params.type, params.num) == (1135, 2, 1, 6)
    assert params.data == ROH_SIX


def test_rohand_third_generation_wire_format_and_port_restore(rig):
    build, log = rig
    out = _connected(build, version=3)
    try:
        out.configure_hand("rohand", baudrate=115200, device_id=2)
        out.enable_hand()
        out.send_hand(ROH_SIX, guard=allow)
    finally:
        out.close()
    assert _calls(log, "modbus_mode") == [(1, 115200, 1)]
    ((params, data),) = _calls(log, "v3_write")
    assert (params.port, params.address, params.device, params.num) == (1, 1135, 2, 6)
    # Third-generation writes take bytes, most significant first.
    assert data == (0, 0, 0, 1, 0, 255, 1, 0, 156, 64, 255, 255)
    # The port is handed back on close.
    assert _calls(log, "modbus_close") == [1]


@pytest.mark.parametrize(
    "version, raw, expected",
    [
        (4, [10, 20, 30, 40, 50, 65535], [10, 20, 30, 40, 50, 65535]),
        # Third generation returns signed int8 bytes; 0xFF arrives as -1.
        (3, [0, 10, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0], [10, 256, 65535, 0, 0, 0]),
    ],
)
def test_rohand_position_read(rig, version, raw, expected):
    build, _log = rig
    out = _connected(build, version=version, hand_read=raw)
    try:
        out.configure_hand("rohand")
        assert out.read_hand() == expected
    finally:
        out.close()


@pytest.mark.parametrize(
    "hand, bad",
    [
        ("inspire", [1001, 0, 0, 0, 0, 0]),
        ("inspire", [0] * 5),
        ("inspire", [0.5, 0, 0, 0, 0, 0]),
        ("rohand", [65536, 0, 0, 0, 0, 0]),
        ("rohand", [-1, 0, 0, 0, 0, 0]),
    ],
)
def test_out_of_range_hand_targets_are_rejected(rig, hand, bad):
    build, log = rig
    out = _connected(build)
    try:
        out.configure_hand(hand)
        out.enable_hand()
        with pytest.raises(ValueError):
            out.send_hand(bad, guard=allow)
    finally:
        out.close()
    assert _calls(log, "hand_angle") == [] and _calls(log, "rtu_write") == []


def test_hand_is_gated_separately_from_the_arm(rig):
    """--hand-only drives the hand while arm motion stays refused."""
    build, log = rig
    out = _connected(build)
    try:
        out.configure_hand("inspire")
        with pytest.raises(RealManError, match="Hand disabled"):
            out.send_hand(SIX, guard=allow)  # Configured is not enabled.
        out.enable_hand()
        out.send_hand(SIX, guard=allow)
        with pytest.raises(RealManError, match="Motion disabled"):
            out.send_joints(np.zeros(7), guard=allow)
    finally:
        out.close()
    verbs = _verbs(log)
    assert "hand_angle" in verbs
    assert "movej" not in verbs and "canfd" not in verbs and "stop" not in verbs


def test_hand_write_failure_latches_and_blocks_the_arm(rig):
    build, _log = rig
    out = _connected(build, fail={"rtu_write": -2})
    try:
        out.configure_hand("rohand")
        out.enable_hand()
        out.enable_motion()
        with pytest.raises(RealManError, match="ROHand register write"):
            out.send_hand(ROH_SIX, guard=allow)
        # A hand that stopped answering fails closed for everything.
        with pytest.raises(RealManError, match="faulted"):
            out.send_joints(np.zeros(7), guard=allow)
    finally:
        out.close()


def test_hand_tracking_guard_skips_without_latching(rig):
    build, log = rig
    out = _connected(build)

    def stale(_now):
        raise InputTrackingFault("stale")

    try:
        out.configure_hand("inspire")
        out.enable_hand()
        with pytest.raises(InputTrackingFault):
            out.send_hand(SIX, guard=stale)
        out.send_hand(SIX, guard=allow)
    finally:
        out.close()
    assert len(_calls(log, "hand_angle")) == 1


# -- hand driver: rate limit and de-duplication -------------------------------


def _driver(sent, *, rate_hz=25.0, retarget=lambda lm: lm):
    return entry._HandDriver(
        retarget=retarget,
        to_channels=lambda raw: [int(v) for v in raw],
        send=sent.append,
        full_scale=1000,
        rate_hz=rate_hz,
    )


def test_hand_driver_rate_limits_both_retarget_and_send():
    sent, retargeted = [], []

    def retarget(landmarks):
        retargeted.append(landmarks)
        return landmarks

    driver = _driver(sent, rate_hz=25.0, retarget=retarget)  # 40 ms period.
    assert driver.step([100] * 6, 0.000) is True
    assert driver.step([500] * 6, 0.020) is False  # Within the period.
    assert driver.step([500] * 6, 0.041) is True
    assert sent == [[100] * 6, [500] * 6]
    # The optimiser did not run on the throttled frame either.
    assert len(retargeted) == 2


def test_hand_driver_skips_targets_that_barely_moved():
    sent = []
    driver = _driver(sent)
    driver.step([100] * 6, 0.0)
    # 0.5% of a 1000 full scale is 5 counts.
    assert driver.step([103] * 6, 1.0) is False
    assert driver.step([106] * 6, 2.0) is True
    assert sent == [[100] * 6, [106] * 6]


def test_hand_driver_ignores_missing_or_invalid_retargets():
    sent = []
    assert _driver(sent).step(None, 0.0) is False
    assert _driver(sent, retarget=lambda lm: None).step([1] * 6, 0.0) is False
    assert _driver(sent, retarget=lambda lm: [float("nan")] * 6).step([1] * 6, 0.0) is False
    assert sent == []


def test_arm_state_stores_landmarks_without_faking_wrist_freshness():
    state = entry._ArmState(tracker=None, site_id=None, ik_data=None,
                            cmd_pos=None, cmd_quat=None, home=None)
    assert state.update_hand_from_mediapipe("points") is False
    assert state.latest_landmarks == "points"
    state.update_hand_from_mediapipe(None)  # A dropped frame keeps the last.
    assert state.latest_landmarks == "points"
    state.update_from_pose((0, 0, 0), (0, 0, 0, 1))  # No tracker: ignored.


# -- ROHand channel mapping ------------------------------------------------


def _rohand_raw(**angles):
    spec = RealManRM75BSingleROHandSpec()
    names = [joint for joint, _o, _c in spec.rohand_channels] + ["if_slider_link"]
    values = [angles.get(name, 0.0) for name in names]
    return spec, values, names


def test_rohand_mapping_reads_joints_by_name_in_register_order():
    spec = RealManRM75BSingleROHandSpec()
    opened = {joint: open_ for joint, open_, _c in spec.rohand_channels}
    closed = {joint: closed_ for joint, _o, closed_ in spec.rohand_channels}
    _spec, values, names = _rohand_raw(**opened)
    assert raw_qpos_to_positions(values, names, spec) == [0] * 6
    _spec, values, names = _rohand_raw(**closed)
    assert raw_qpos_to_positions(values, names, spec) == [65535] * 6
    # Shuffling the retarget output order must not change the result.
    order = list(reversed(range(len(names))))
    assert raw_qpos_to_positions(
        [values[i] for i in order], [names[i] for i in order], spec
    ) == [65535] * 6


def test_rohand_mapping_clips_and_can_invert():
    spec = RealManRM75BSingleROHandSpec()
    beyond = {joint: closed_ * 3 if closed_ else 99.0 for joint, _o, closed_ in spec.rohand_channels}
    _spec, values, names = _rohand_raw(**beyond)
    positions = raw_qpos_to_positions(values, names, spec)
    assert all(0 <= p <= 65535 for p in positions)
    flipped = raw_qpos_to_positions(values, names, spec, closed_is_max=False)
    assert [65535 - p for p in positions] == flipped


def test_rohand_mapping_rejects_bad_input():
    spec, values, names = _rohand_raw()
    with pytest.raises(ValueError, match="no joint"):
        raw_qpos_to_positions(values[1:], names[1:], spec)
    with pytest.raises(ValueError, match="finite"):
        raw_qpos_to_positions([float("nan")] * len(names), names, spec)
    with pytest.raises(ValueError, match="matching"):
        raw_qpos_to_positions(values[:-1], names, spec)


# -- entrypoint modes ------------------------------------------------------


class _StopAfter:
    """Input stream yielding landmarks, then a stop gesture."""

    def __init__(self, _args, frames=3):
        self.frames = frames
        self.polls = 0

    def poll_arms(self, _left, arm, _converter, allow_stop_gesture=False):
        self.polls += 1
        if self.polls > self.frames:
            return InputPollResult(stop_requested=True)
        arm.update_hand_from_mediapipe([self.polls * 100] * 6)
        arm.update_from_pose((0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        return InputPollResult(saw_valid_data=True)

    def get_pico4_head_pose(self):
        return None

    def close(self):
        pass


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _hand_args(hand="rohand", controller=False):
    return entry.argparse.Namespace(
        config=ROOT / "config" / "realman_rm75b_single_real.yaml",
        hand=hand, input_source="pico4", avp_ip=None, port=9000,
        controller=controller, duration=0.01,
    )


def test_hand_only_never_moves_the_arm():
    log = []

    def output_factory(ip):
        return RM75BSingleOutput(ip, factory=lambda _ip: FakeArm(log), sdk_types=FakeTypes)

    def hand_driver_factory(output, args, spec, config):
        output.configure_hand(args.hand)
        output.enable_hand()
        return entry._HandDriver(
            retarget=lambda lm: lm, to_channels=lambda raw: list(raw),
            send=lambda ch: output.send_hand(ch, guard=allow), full_scale=65535,
            rate_hz=1000.0, min_change_fraction=0.0,
        )

    with pytest.raises(KeyboardInterrupt):
        entry.run_hand_only(
            _hand_args(), output_factory=output_factory, input_factory=_StopAfter,
            clock=_FakeClock(), hand_driver_factory=hand_driver_factory,
        )
    verbs = _verbs(log)
    assert verbs.count("rtu_write") == 3  # Every landmark frame drove the hand.
    for forbidden in ("movej", "canfd", "stop"):
        assert forbidden not in verbs
    assert verbs[-2:] == ["delete", "destroy"]


def test_read_only_with_a_hand_writes_no_target():
    log = []

    def output_factory(ip):
        return RM75BSingleOutput(ip, factory=lambda _ip: FakeArm(log), sdk_types=FakeTypes)

    assert entry.run_read_only(_hand_args(), output_factory=output_factory) == 0
    verbs = _verbs(log)
    assert "voltage" in verbs and "rtu_read" in verbs
    for forbidden in ("movej", "canfd", "stop", "hand_angle", "rtu_write", "v3_write"):
        assert forbidden not in verbs


def test_controller_mode_leaves_the_hand_untouched():
    log, built = [], []

    def output_factory(ip):
        return RM75BSingleOutput(ip, factory=lambda _ip: FakeArm(log), sdk_types=FakeTypes)

    class _Clock(_FakeClock):
        def sleep(self, seconds):
            super().sleep(seconds)
            if self.now > 0.1:
                raise KeyboardInterrupt

    class _NoController(_StopAfter):
        pico4_input = None

    with pytest.raises(KeyboardInterrupt):
        entry.run_motion(
            _hand_args(controller=True),
            output_factory=output_factory,
            input_factory=_NoController,
            clock=_Clock(),
            hand_driver_factory=lambda *a: built.append(a),
        )
    assert built == []  # The hand driver was never set up.
    for verb in ("voltage", "rs485_mode", "modbus_mode", "hand_angle", "rtu_write", "v3_write"):
        assert verb not in _verbs(log)


def test_unguarded_main_cannot_reach_real_hardware():
    """Arguments that pass validation stop before connecting under test."""
    assert entry.main(["--hand-only", "--hand", "rohand"]) == 1


@pytest.mark.parametrize(
    "hands, match",
    [
        ({"rohand": {"baudrate": 12345}}, "baudrate"),
        ({"rohand": {"device_id": 0}}, "device_id"),
        ({"rohand": {"closed_is_max": "yes"}}, "closed_is_max"),
        ({"gripper": {}}, "only contain"),
    ],
)
def test_hand_config_is_validated(tmp_path, hands, match):
    import yaml

    config = yaml.safe_load((ROOT / "config" / "realman_rm75b_single_real.yaml").read_text())
    config["hands"] = hands
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        entry.load_config(path)


def test_hand_cli_validation():
    # Only argument sets the parser rejects: anything that passes validation
    # would run the real entrypoint.
    for argv in (
        ["--hand-only"],
        ["--hand-only", "--hand", "rohand", "--controller"],
        ["--hand-only", "--enable-motion", "--hand", "rohand"],
    ):
        with pytest.raises(SystemExit):
            entry.main(argv)

"""DexForce real-robot tests. No ROS, no hardware: every transport is faked."""

import json
import time

import numpy as np
import pytest

from example import teleop_dexforce_real as runner
from utils.adapters.output.dexforce_ros import (
    DexForceFeedbackError,
    DexForceRealOutput,
    joint_positions_by_name,
    parse_server_state,
)
from utils.adapters.output.linker_l20_mapping import (
    DEFAULT_SIGNS,
    DEXFORCE_HAND_JOINT_NAMES,
    EXPECTED_RETARGET_ORDER,
    load_sim_hand_limits,
    open_hand_percent,
    retarget_indices,
    retarget_to_dexforce_percent,
)
from utils.robots.dexforce import DexForceSpec

_spec = DexForceSpec()


def make_state(**overrides):
    names = list(_spec.full_body_joint_names())
    state = {
        "timestamp": 1,
        "status": "Running",
        "run_mode": "Position",
        "joint_name": names,
        "joint_position": [0.1] * len(names),
        "joint_velocity": [0.0] * len(names),
        "joint_torque": [0.0] * len(names),
        "motor_error_code": [0] * len(names),
    }
    state.update(overrides)
    return json.dumps(state)


# --- server state parsing ---------------------------------------------------


def test_parse_server_state_accepts_a_well_formed_document():
    state = parse_server_state(make_state())
    assert state["run_mode"] == "Position"
    assert len(state["joint_name"]) == 20


def test_parse_server_state_reports_invalid_json():
    with pytest.raises(DexForceFeedbackError, match="not valid JSON"):
        parse_server_state("{not json")


@pytest.mark.parametrize("overrides", [
    {"joint_name": None},
    {"joint_position": None},
    {"joint_name": ["A", "B"], "joint_position": [1.0]},
])
def test_parse_server_state_rejects_inconsistent_documents(overrides):
    with pytest.raises(DexForceFeedbackError):
        parse_server_state(make_state(**overrides))


def test_parse_server_state_rejects_non_finite_positions():
    names = list(_spec.full_body_joint_names())
    payload = make_state(joint_position=[float("nan")] + [0.0] * (len(names) - 1))
    with pytest.raises(DexForceFeedbackError, match="non-finite"):
        parse_server_state(payload)


def test_joint_positions_by_name_maps_names_to_values():
    state = parse_server_state(make_state())
    mapping = joint_positions_by_name(state)
    assert mapping["WAIST"] == pytest.approx(0.1)
    assert set(mapping) == set(_spec.full_body_joint_names())


# --- startup gating ---------------------------------------------------------


def _output_with_state(**overrides):
    output = DexForceRealOutput(_spec, read_only=True)
    output._latest_state = parse_server_state(make_state(**overrides))
    return output


def test_startup_state_is_clean_for_a_healthy_robot():
    assert _output_with_state().check_startup_state() == []


def test_startup_state_flags_missing_feedback():
    problems = DexForceRealOutput(_spec, read_only=True).check_startup_state()
    assert any("no /feedback/robot_server_state" in p for p in problems)


@pytest.mark.parametrize("run_mode", ["Drag", "Impedance", "Sleep"])
def test_startup_state_flags_blocked_run_modes(run_mode):
    problems = _output_with_state(run_mode=run_mode).check_startup_state()
    assert any("run_mode" in p for p in problems)


def test_startup_state_flags_an_error_status():
    problems = _output_with_state(status="Error").check_startup_state()
    assert any("status" in p for p in problems)


def test_startup_state_flags_motor_errors():
    codes = [0] * 20
    codes[3] = 20
    problems = _output_with_state(motor_error_code=codes).check_startup_state()
    assert any("motor_error_code" in p for p in problems)


def test_startup_state_flags_joints_the_robot_does_not_report():
    names = list(_spec.full_body_joint_names())[:-1]
    problems = _output_with_state(joint_name=names, joint_position=[0.0] * len(names)).check_startup_state()
    assert any("NECK2" in p for p in problems)


# --- command assembly, clipping and slew limiting ---------------------------


def test_build_command_uses_measured_body_and_keeps_joint_order():
    output = _output_with_state()
    command = output.build_command({"left": np.arange(7, dtype=float),
                                    "right": np.arange(7, dtype=float) + 10.0},
                                   None, body_from_feedback=True)
    assert command.shape == (20,)
    index = output.joint_index
    assert command[index["LEFT_J1"]] == pytest.approx(0.0)
    assert command[index["RIGHT_J1"]] == pytest.approx(10.0)
    # Body joints come from feedback, not from a stale local value.
    assert command[index["WAIST"]] == pytest.approx(0.1)


def test_build_command_rejects_a_wrong_length_arm():
    output = _output_with_state()
    with pytest.raises(ValueError, match="expected 7"):
        output.build_command({"left": [0.0, 0.0]}, None)


def test_clamp_command_clips_into_the_joint_limits():
    output = _output_with_state(joint_position=[0.0] * 20)
    output._limits = np.tile([-1.0, 1.0], (20, 1))
    clipped = output.clamp_command(np.full(20, 5.0))
    assert np.all(clipped <= 1.0 - 0.05 + 1e-9)


def test_clamp_command_respects_the_joint_margin():
    output = _output_with_state(joint_position=[0.0] * 20)
    output._limits = np.tile([-1.0, 1.0], (20, 1))
    clipped = output.clamp_command(np.full(20, -5.0))
    assert np.all(clipped >= -1.0 + 0.05 - 1e-9)


def test_clamp_command_passes_values_inside_the_limits_through():
    output = _output_with_state(joint_position=[0.0] * 20)
    output._limits = np.tile([-2.0, 2.0], (20, 1))
    target = np.full(20, 1.0)
    assert np.allclose(output.clamp_command(target), target)


def test_clamp_command_rejects_nan():
    output = _output_with_state()
    with pytest.raises(ValueError, match="NaN/Inf"):
        output.clamp_command(np.full(20, np.nan))


def test_read_only_never_creates_publishers():
    output = DexForceRealOutput(_spec, read_only=True)
    assert output.send_command(np.zeros(20)) is False
    assert output.send_hand("left", open_hand_percent()) is False


def test_send_command_returns_false_without_a_publisher():
    # read_only installs no publisher, so nothing can reach the robot even if
    # the caller forgets the flag further up.
    output = DexForceRealOutput(_spec, read_only=True)
    assert output._pub is None
    assert output.send_command(np.zeros(20)) is False
    assert output.commands_sent == 0


def test_send_command_rejects_non_finite_commands():
    output = DexForceRealOutput(_spec, read_only=True)
    with pytest.raises(ValueError, match="NaN/Inf"):
        output.send_command(np.full(20, np.inf))


# --- hand mapping -----------------------------------------------------------


def _fake_limits(lower=0.0, upper=1.0):
    limits = np.tile([lower, upper], (len(DEXFORCE_HAND_JOINT_NAMES), 1))
    return limits


def test_retarget_indices_maps_thumb_yaw_from_the_verified_order():
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    by_name = dict(zip(DEXFORCE_HAND_JOINT_NAMES, indices))
    # Thumb yaw is index 17 in retarget output but comes first for DexForce.
    assert by_name["T_CMC_YAW"] == EXPECTED_RETARGET_ORDER.index("thumb_cmc_yaw")
    assert by_name["T_CMC_ROLL"] == EXPECTED_RETARGET_ORDER.index("thumb_cmc_roll")
    assert by_name["IF_PIP"] == EXPECTED_RETARGET_ORDER.index("index_pip")


def test_retarget_indices_skips_the_passive_joints():
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    dropped = {EXPECTED_RETARGET_ORDER.index(n) for n in
               ("index_dip", "middle_dip", "pinky_dip", "ring_dip", "thumb_ip")}
    assert dropped.isdisjoint(indices)


def test_retarget_indices_rejects_an_unexpected_order():
    with pytest.raises(ValueError, match="Unexpected linker_l20 retarget joint order"):
        retarget_indices(["thumb_ip"] * 21)


def test_retarget_to_dexforce_percent_matches_the_indices_mapping():
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    qpos = np.zeros(21)
    qpos[EXPECTED_RETARGET_ORDER.index("index_pip")] = 1.0   # at the upper limit
    qpos[EXPECTED_RETARGET_ORDER.index("ring_mcp_roll")] = 0.5
    values = retarget_to_dexforce_percent(qpos, _fake_limits(), indices=indices)

    by_name = dict(zip(DEXFORCE_HAND_JOINT_NAMES, values))
    assert by_name["IF_PIP"] == pytest.approx(100.0)
    # Bidirectional ab/adduction joints map 0.5 onto the range midpoint.
    assert by_name["RF_MCP_ROLL"] == pytest.approx(0.0)
    assert by_name["T_CMC_YAW"] == pytest.approx(-100.0)


def test_finger_mcp_roll_directions_are_flipped_for_the_hardware():
    # The hand abducted the wrong way on the robot, so these four joints carry
    # a -1 in DEFAULT_SIGNS. Pin it: reverting the table would silently send
    # fingers the wrong way again, and the midpoint assertion above cannot see
    # the difference because it lands on +/-0.0 either way.
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    rolls = [n for n in DEXFORCE_HAND_JOINT_NAMES if n.endswith("_MCP_ROLL")]
    assert len(rolls) == 4
    assert all(DEFAULT_SIGNS[n] == -1.0 for n in rolls)
    assert all(DEFAULT_SIGNS[n] == 1.0 for n in DEXFORCE_HAND_JOINT_NAMES if n not in rolls)

    source_index = dict(zip(DEXFORCE_HAND_JOINT_NAMES, indices))
    qpos = np.zeros(21)
    for name in rolls:
        qpos[source_index[name]] = 1.0
    by_name = dict(zip(DEXFORCE_HAND_JOINT_NAMES,
                       retarget_to_dexforce_percent(qpos, _fake_limits(), indices=indices)))
    # Retarget output at its upper limit must come out at the negative end.
    for name in rolls:
        assert by_name[name] == pytest.approx(-100.0), name


def test_sign_flip_leaves_the_other_finger_joints_untouched():
    # A flip must not bias the joints it does not apply to, and the neutral
    # open-hand command must stay neutral.
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    qpos = np.full(21, 0.6)
    unflipped = {n: 1.0 for n in DEXFORCE_HAND_JOINT_NAMES}
    before = retarget_to_dexforce_percent(qpos, _fake_limits(), indices=indices, signs=unflipped)
    after = retarget_to_dexforce_percent(qpos, _fake_limits(), indices=indices)

    for name, old, new in zip(DEXFORCE_HAND_JOINT_NAMES, before, after):
        if name.endswith("_MCP_ROLL"):
            assert new == pytest.approx(-old), name
        else:
            assert new == pytest.approx(old), name


def test_retarget_to_dexforce_percent_clips_out_of_range_input():
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    qpos = np.full(21, 5.0)
    values = retarget_to_dexforce_percent(qpos, _fake_limits(), indices=indices)
    assert all(-100.0 <= v <= 100.0 for v in values)


def test_retarget_to_dexforce_percent_rejects_wrong_shapes():
    indices = retarget_indices(list(EXPECTED_RETARGET_ORDER))
    with pytest.raises(ValueError, match="hand limits"):
        retarget_to_dexforce_percent(np.zeros(21), np.zeros((3, 2)), indices=indices)
    with pytest.raises(ValueError, match="indices"):
        retarget_to_dexforce_percent(np.zeros(21), _fake_limits(), indices=[0, 1])
    with pytest.raises(ValueError, match="NaN/Inf"):
        retarget_to_dexforce_percent(np.full(21, np.nan), _fake_limits(), indices=indices)


def test_open_hand_percent_is_neutral_and_full_length():
    values = open_hand_percent()
    assert len(values) == len(DEXFORCE_HAND_JOINT_NAMES)
    assert all(v == 0.0 for v in values)


def test_load_sim_hand_limits_matches_the_scene_model():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_spec.scene_path))
    limits = load_sim_hand_limits(model, _spec, "left")
    assert limits.shape == (len(DEXFORCE_HAND_JOINT_NAMES), 2)
    assert np.all(limits[:, 1] > limits[:, 0])

    by_name = dict(zip(DEXFORCE_HAND_JOINT_NAMES, limits.tolist()))
    # The four finger ab/adduction joints are the bidirectional ones, which is
    # why the robot gives them a -100..100 range.
    assert by_name["IF_MCP_ROLL"][0] < 0.0
    assert by_name["IF_MCP_PITCH"][0] == pytest.approx(0.0)
    assert by_name["LF_PIP"][0] == pytest.approx(0.0)


# --- spec ------------------------------------------------------------------


def test_real_home_qpos_is_not_the_simulation_home():
    assert not np.allclose(_spec.real_home_qpos("left"), _spec.arm_home_qpos("left"))
    assert _spec.real_home_qpos("right").shape == (7,)


def test_full_body_joint_names_are_the_twenty_the_driver_expects():
    names = _spec.full_body_joint_names()
    assert names == (
        "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
        "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
        "ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2",
    )
    assert len(set(names)) == 20


def test_body_home_qpos_is_torso_then_head():
    spec = DexForceSpec()
    assert spec.real_body_home_qpos().tolist() == pytest.approx(
        [-0.6, 1.2, -0.6, 0.0, 0.0, -0.5590]
    )


# --- startup move -----------------------------------------------------------


def _state_with_arms(left, right):
    names = list(_spec.full_body_joint_names())
    by_name = dict.fromkeys(names, 0.0)
    for side, values in (("left", left), ("right", right)):
        by_name.update(zip(_spec.arm_joint_names(side), values))
    return make_state(joint_position=[by_name[n] for n in names])


def _output_fed_by(states):
    """An output whose spin_once delivers the queued states one at a time."""
    output = DexForceRealOutput(_spec, read_only=True)
    output.node = object()
    output._latest_state = parse_server_state(states[0])
    pending = list(states[1:])

    def spin_once(timeout_sec=0.0):
        if pending:
            output._latest_state = parse_server_state(pending.pop(0))
        else:
            time.sleep(timeout_sec)

    output.spin_once = spin_once
    return output


def _home_command(output):
    return output.build_command(
        {side: _spec.real_home_qpos(side) for side in ("left", "right")},
        np.zeros(len(_spec.real_body_joint_names())),
        body_from_feedback=False,
    )


def test_ramp_to_home_keeps_reading_feedback():
    home = {side: _spec.real_home_qpos(side) for side in ("left", "right")}
    output = _output_fed_by([
        _state_with_arms(np.zeros(7), np.zeros(7)),
        _state_with_arms(home["left"], home["right"]),
    ])
    target = _home_command(output)
    runner._ramp_to_home(output, target, target, duration_s=0.02, publish_period_s=0.01)
    assert output.read_arm_positions("left") == pytest.approx(home["left"])


def test_wait_until_reached_uses_fresh_feedback():
    home = {side: _spec.real_home_qpos(side) for side in ("left", "right")}
    output = _output_fed_by([
        _state_with_arms(np.zeros(7), np.zeros(7)),
        _state_with_arms(home["left"] * 0.5, home["right"] * 0.5),
        _state_with_arms(home["left"], home["right"]),
    ])
    error = runner._wait_until_reached(
        output, _home_command(output), ("left", "right"),
        timeout_s=1.0, tolerance_rad=0.05, publish_period_s=0.01,
    )
    assert error == pytest.approx(0.0)


def test_wait_until_reached_reports_the_error_when_the_arms_never_arrive():
    output = _output_fed_by([_state_with_arms(np.zeros(7), np.zeros(7))])
    start = time.monotonic()
    error = runner._wait_until_reached(
        output, _home_command(output), ("left", "right"),
        timeout_s=0.05, tolerance_rad=0.05, publish_period_s=0.01,
    )
    assert error == pytest.approx(np.max(np.abs(_spec.real_home_qpos("left"))))
    assert time.monotonic() - start >= 0.05

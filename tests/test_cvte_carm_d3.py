"""CVTE CARM D3 simulation scene, spec, and pinch-to-gripper control."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from example import teleop_sim
from utils.core.arm_controller import TeleopArmController
from utils.robots.cvte_carm_d3 import CVTECArmD3Spec
from utils.robots.realman_rm75b import RealManRM75BInspireSpec

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "example" / "scene_config" / "scene_cvte_carm_d3.xml"


@pytest.fixture(scope="module")
def spec() -> CVTECArmD3Spec:
    return CVTECArmD3Spec()


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _object_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id != -1, f"Missing {name}"
    return object_id


def _home_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)
    return data


def _arm_controller(model: mujoco.MjModel, data: mujoco.MjData, spec, side: str) -> TeleopArmController:
    return TeleopArmController(
        model,
        data,
        mujoco.MjData(model),
        side,
        SimpleNamespace(
            position_scale=1.0,
            ema_alpha=1.0,
            rot_weight=1.0,
            ik_damping=1e-3,
            ik_current_weight=0.1,
        ),
        hand_config=None,
        hand_type=spec.hand_type,
        home_qpos=spec.arm_home_qpos(side),
        robot_label="cvte+gripper",
        arm_joint_names=spec.arm_joint_names(side),
        hand_joint_names=spec.hand_joint_names(side),
        ee_body_name=spec.ee_body_name(side),
        base_body_name=spec.base_body_name,
        hand_qpos_mapping=spec.hand_qpos_mapping,
    )


def test_scene_loads_cvte_carm_d3_combination(model: mujoco.MjModel, spec: CVTECArmD3Spec) -> None:
    # 7 arm joints + 1 driven gripper joint per side, plus the mimicked
    # gripper_left joint (coupled via equality, not separately actuated).
    assert (model.nq, model.nv) == (18, 18)
    assert model.nu == 16
    assert model.neq == 2
    assert spec.hand_type == "cvte_gripper"
    assert spec.torso_joint_names == ()
    assert spec.head_joint_names == ()

    for side in ("left", "right"):
        assert len(spec.hand_joint_names(side)) == 1
        for joint_name in spec.arm_joint_names(side) + spec.hand_joint_names(side):
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name(side))
    _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_body_name)


def test_home_keyframe_is_within_limits(model: mujoco.MjModel, spec: CVTECArmD3Spec) -> None:
    data = _home_data(model)
    for side in ("left", "right"):
        for joint_name in spec.arm_joint_names(side) + spec.hand_joint_names(side):
            joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            value = data.qpos[model.jnt_qposadr[joint_id]]
            lower, upper = model.jnt_range[joint_id]
            assert lower <= value <= upper, f"{joint_name}={value} outside [{lower}, {upper}]"
        np.testing.assert_allclose(
            [data.qpos[model.jnt_qposadr[_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in spec.arm_joint_names(side)],
            spec.arm_home_qpos(side),
            atol=1e-12,
        )


def test_home_pose_points_both_grippers_forward(model: mujoco.MjModel, spec: CVTECArmD3Spec) -> None:
    data = _home_data(model)
    positions = {}
    for side in ("left", "right"):
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name(side))
        # gripper_base +Z is the finger (approach) direction.
        finger_direction = data.xmat[body_id].reshape(3, 3)[:, 2]
        assert finger_direction[0] > 0.98
        positions[side] = data.xpos[body_id].copy()

    left, right = positions["left"], positions["right"]
    assert left[0] > 0.25
    assert left[0] == pytest.approx(right[0], abs=1e-6)
    assert left[1] == pytest.approx(-right[1], abs=1e-6)
    assert left[2] == pytest.approx(right[2], abs=1e-6)


def test_gripper_mimic_equality_holds_under_forward_dynamics(model: mujoco.MjModel) -> None:
    data = _home_data(model)
    for side in ("LEFT", "RIGHT"):
        right_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_right_joint-{side}")
        left_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_left_joint-{side}")
        data.qpos[model.jnt_qposadr[right_id]] = 0.02
        mujoco.mj_forward(model, data)
        for _ in range(200):
            mujoco.mj_step(model, data)
        assert data.qpos[model.jnt_qposadr[left_id]] == pytest.approx(
            data.qpos[model.jnt_qposadr[right_id]], abs=1e-3
        )


def test_update_hand_from_pinch_maps_distance_into_gripper_range(
    model: mujoco.MjModel, spec: CVTECArmD3Spec
) -> None:
    data = _home_data(model)
    controller = _arm_controller(model, data, spec, "left")
    joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.hand_joint_names("left")[0])
    lower, upper = model.jnt_range[joint_id]

    assert controller.update_hand_from_pinch(None) is False

    assert controller.update_hand_from_pinch(0.02, closed_m=0.02, open_m=0.10) is True
    assert controller.latest_hand_qpos[0] == pytest.approx(lower, abs=1e-9)

    assert controller.update_hand_from_pinch(0.10, closed_m=0.02, open_m=0.10) is True
    assert controller.latest_hand_qpos[0] == pytest.approx(upper, abs=1e-9)

    # Out-of-range distances clamp instead of extrapolating past the joint limits.
    assert controller.update_hand_from_pinch(0.30, closed_m=0.02, open_m=0.10) is True
    assert controller.latest_hand_qpos[0] == pytest.approx(upper, abs=1e-9)


def test_update_hand_from_pinch_noops_for_dexterous_hand() -> None:
    """Regression guard: dex-hand robots (12+ hand DOF) must be unaffected."""
    inspire_spec = RealManRM75BInspireSpec()
    inspire_model = mujoco.MjModel.from_xml_path(str(inspire_spec.scene_path))
    data = mujoco.MjData(inspire_model)
    mujoco.mj_resetDataKeyframe(
        inspire_model, data, mujoco.mj_name2id(inspire_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    mujoco.mj_forward(inspire_model, data)
    controller = _arm_controller(inspire_model, data, inspire_spec, "left")

    assert len(controller.hand_dof_indices) > 1
    assert controller.update_hand_from_pinch(0.05) is False
    assert controller.latest_hand_qpos is None


def test_cvte_registered_in_teleop_sim_cli(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(sys, "argv", ["teleop_sim.py", "--robot", "cvte"])
    monkeypatch.setattr(
        teleop_sim,
        "_run_bimanual",
        lambda args: captured.update(vars(args)),
    )

    teleop_sim.main()

    assert captured["robot"] == "cvte"
    assert captured["hand"] == "gripper"


def test_cvte_gripper_never_loads_a_retarget_config(spec: CVTECArmD3Spec) -> None:
    args = SimpleNamespace(hand_config=None, input_source="pico4")
    for side in ("left", "right"):
        assert spec.hand_config_path("pico4", side) is None
        assert teleop_sim._resolve_hand_config(args, spec, side) is None

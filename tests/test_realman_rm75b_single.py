import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from example import teleop_sim
from utils.core.body.ik import solve_body_pose_ik
from utils.core.quaternion import matrix_to_quaternion
from utils.robots.realman_rm75b_single import (
    ARM_JOINT_COUNT,
    FLANGE_BODY,
    PEDESTAL_BODY,
    SINGLE_ARM_HOME_QPOS,
    TCP_SITE,
    RealManRM75BSingleSpec,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "example" / "scene_config" / "scene_realman_rm75b_single.xml"

# Manual section 6.1, "运动角度范围": J1 +-178, J2 +-130, J3 +-178, J4 +-135,
# J5 +-178, J6 +-128, J7 +-360 degrees.
MANUAL_JOINT_LIMITS_DEG = (178.0, 130.0, 178.0, 135.0, 178.0, 128.0, 360.0)
# Manual section 6.2.2: MDH link offsets d1, d3, d5 and the RM75-B d7.
MANUAL_SHOULDER_HEIGHT_M = 0.2405
# Manual 6.2.2 MDH: d1 + d3 + d5 + d7 for the RM75-B flange face.
MANUAL_BASE_TO_FLANGE_M = 0.2405 + 0.256 + 0.210 + 0.144


@pytest.fixture(scope="module")
def spec() -> RealManRM75BSingleSpec:
    return RealManRM75BSingleSpec()


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _object_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id != -1, f"Missing {name}"
    return object_id


def _home_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def test_scene_loads_one_arm_with_no_end_effector(
    model: mujoco.MjModel,
    spec: RealManRM75BSingleSpec,
) -> None:
    # Seven arm joints and nothing else: no torso, head, wheels, or hand.
    assert (model.nq, model.nv, model.nu) == (ARM_JOINT_COUNT,) * 3
    assert spec.hand_type == "none"
    assert spec.sides == ("right",)
    assert spec.torso_joint_names == ()
    assert spec.head_joint_names == ()
    assert spec.hand_joint_names("right") == ()
    assert spec.hand_config_path("pico4", "right") is None

    for joint_name in spec.arm_joint_names("right"):
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

    _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name("right"))
    _object_id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    _object_id(model, mujoco.mjtObj.mjOBJ_BODY, PEDESTAL_BODY)


def test_absent_left_arm_is_rejected_rather_than_aliased(
    spec: RealManRM75BSingleSpec,
) -> None:
    # The dual-arm specs map any non-"left" side onto the right arm, which would
    # silently accept a side this robot does not have.
    for accessor in (
        spec.arm_joint_names,
        spec.arm_home_qpos,
        spec.ee_body_name,
        spec.hand_joint_names,
    ):
        with pytest.raises(ValueError, match="single-arm supports"):
            accessor("left")
    with pytest.raises(ValueError, match="single-arm supports"):
        spec.hand_config_path("pico4", "left")


def test_joint_limits_match_the_vendor_manual(
    model: mujoco.MjModel,
    spec: RealManRM75BSingleSpec,
) -> None:
    for joint_name, limit_deg in zip(
        spec.arm_joint_names("right"), MANUAL_JOINT_LIMITS_DEG
    ):
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        lower, upper = np.rad2deg(model.jnt_range[joint_id])
        assert lower == pytest.approx(-limit_deg, abs=0.5), joint_name
        assert upper == pytest.approx(limit_deg, abs=0.5), joint_name


def test_home_keyframe_maps_qpos_to_each_named_actuator(
    model: mujoco.MjModel,
    spec: RealManRM75BSingleSpec,
) -> None:
    data = _home_data(model)
    key_id = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    actual = []
    for joint_name in spec.arm_joint_names("right"):
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        value = float(data.qpos[model.jnt_qposadr[joint_id]])
        lower, upper = model.jnt_range[joint_id]
        # The dual-arm scene's checked-in home puts r_joint1 at +227 deg,
        # outside its +-178 deg range. A single-arm home must stay reachable.
        assert lower <= value <= upper, f"{joint_name}={value} outside [{lower}, {upper}]"
        actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        assert model.key_ctrl[key_id, actuator_id] == pytest.approx(value)
        actual.append(value)

    np.testing.assert_allclose(actual, spec.arm_home_qpos("right"), atol=1e-12)
    np.testing.assert_allclose(actual, SINGLE_ARM_HOME_QPOS, atol=1e-12)


def test_arm_stands_on_the_floor_and_reaches_forward_at_home(
    model: mujoco.MjModel,
    spec: RealManRM75BSingleSpec,
) -> None:
    data = _home_data(model)
    base_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    flange_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name("right"))

    # Bolted to the floor: the base sits at the origin with no free joint, so
    # the model has exactly the arm's DoFs and nothing underneath it.
    np.testing.assert_allclose(data.xpos[base_id], np.zeros(3), atol=1e-9)
    assert model.nq == ARM_JOINT_COUNT

    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    shoulder_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "link_1")
    assert zero.xpos[shoulder_id][2] == pytest.approx(MANUAL_SHOULDER_HEIGHT_M, abs=1e-3)
    assert zero.xpos[flange_id][2] > zero.xpos[base_id][2]

    # The flange must sit at the manual's full d1+d3+d5+d7. Carving the arm out
    # of the dual-arm xacro instead lands 9 mm short, because that model's
    # link7 is part-way along the wrist rather than the flange face.
    assert zero.xpos[flange_id][2] == pytest.approx(MANUAL_BASE_TO_FLANGE_M, abs=1e-3)

    # Home reaches forward (+X) rather than standing straight up, well inside
    # the manual's 0.61 m work radius, with the tool axis pointing forward so
    # VR wrist tracking starts roughly aligned.
    flange = data.xpos[flange_id]
    assert flange[0] > 0.3
    assert np.linalg.norm(flange[:2]) < 0.61
    assert flange[2] > 0.15
    tool_forward = data.xmat[flange_id].reshape(3, 3)[0, 2]
    assert tool_forward > 0.9

    # Straight ahead, not off to one side: the arm stays in the X-Z plane.
    assert flange[1] == pytest.approx(0.0, abs=1e-3)

    # Elbow above the wrist, the way a raised arm reaches forward. With the
    # elbow below the flange the arm still reaches the same point but looks
    # slumped, which is what this home pose was retuned to avoid.
    elbow_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "link_4")
    assert data.xpos[elbow_id][2] > flange[2]


def test_home_avoids_the_documented_wrist_singularity(
    spec: RealManRM75BSingleSpec,
) -> None:
    # Manual section 6.2.5 lists q2 == 0 together with q6 == 0 as singular.
    home = spec.arm_home_qpos("right")
    assert not (abs(home[1]) < 1e-6 and abs(home[5]) < 1e-6)


def test_small_forward_ik_step_converges(
    model: mujoco.MjModel,
    spec: RealManRM75BSingleSpec,
) -> None:
    data = _home_data(model)
    workspace = mujoco.MjData(model)

    body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name("right"))
    initial_position = data.xpos[body_id].copy()
    target_position = initial_position + np.array([0.01, 0.0, 0.0])
    target_quaternion = matrix_to_quaternion(data.xmat[body_id].reshape(3, 3))
    dof_indices = np.array(
        [
            model.jnt_dofadr[_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
            for joint_name in spec.arm_joint_names("right")
        ],
        dtype=int,
    )

    solution = solve_body_pose_ik(
        model,
        workspace,
        body_id,
        target_position,
        target_quaternion,
        data.qpos.copy(),
        max_iters=50,
        home_weight=0.0,
        current_q_weight=0.0,
        dof_indices=dof_indices,
    )
    workspace.qpos[:] = solution
    mujoco.mj_forward(model, workspace)
    assert np.linalg.norm(workspace.xpos[body_id] - target_position) < 1e-4


def test_scene_is_portable_and_carries_no_torso_or_hand_assets() -> None:
    scene_text = SCENE.read_text()
    assert "/home/" not in scene_text
    assert "package://dual_75B_arm_robot" not in scene_text
    # Only the right arm's links; no AGV, torso, head, or left-arm meshes.
    for absent in (
        "body_base_link.STL",
        "head_link1.STL",
        "wheel_link",
        "l_link1.STL",
        "l_base_link.STL",
        "hand_base_link.STL",
    ):
        assert absent not in scene_text, absent
    # Sourced from the standalone single-arm package, not the dual-arm xacro.
    assert "realman_rm75b_single/meshes" in scene_text
    assert "r_joint" not in scene_text

    root = ET.fromstring(scene_text)
    for mesh in root.findall("./asset/mesh"):
        mesh_path = (SCENE.parent / mesh.get("file")).resolve()
        assert mesh_path.is_file(), mesh_path


def test_teleop_registration_defaults_single_arm_to_no_hand(
    spec: RealManRM75BSingleSpec,
) -> None:
    assert teleop_sim._normalize_hand("realman_rm75b_single", None) == "none"
    assert teleop_sim._normalize_hand("realman_rm75b_single", "none") == "none"
    registered = teleop_sim._ROBOT_HAND_SPECS[("realman_rm75b_single", "none")]
    assert registered.scene_path == spec.scene_path
    with pytest.raises(ValueError, match="does not support hand"):
        teleop_sim._normalize_hand("realman_rm75b_single", "inspire")


def test_arm_selection_narrows_to_the_available_side(
    spec: RealManRM75BSingleSpec,
) -> None:
    # "both" stays usable as the default flag value on a one-armed robot.
    assert teleop_sim._resolve_sides(spec, "both") == ("right",)
    assert teleop_sim._resolve_sides(spec, "right") == ("right",)
    with pytest.raises(ValueError, match="does not have a left arm"):
        teleop_sim._resolve_sides(spec, "left")

    # Dual-arm specs keep their existing behaviour.
    dual = teleop_sim._REALMAN_RM75B_SPEC
    assert teleop_sim._resolve_sides(dual, "both") == ("left", "right")
    assert teleop_sim._resolve_sides(dual, "left") == ("left",)

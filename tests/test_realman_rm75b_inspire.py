import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from example import teleop_sim
from utils.adapters.input.pico4 import transform_pico4_to_robot_pose
from utils.core.arm_controller import TeleopArmController
from utils.core.body.ik import solve_body_pose_ik
from utils.core.quaternion import matrix_to_quaternion
from utils.robots.realman_rm75b import RealManRM75BInspireSpec

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "example" / "scene_config" / "scene_realman_rm75b_inspire.xml"
BODY_MESH = (
    ROOT
    / "assets"
    / "arm_body"
    / "realman_rm75b"
    / "meshes"
    / "visual"
    / "body_base_link.STL"
)


@pytest.fixture(scope="module")
def spec() -> RealManRM75BInspireSpec:
    return RealManRM75BInspireSpec()


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


def test_scene_loads_fixed_realman_inspire_combination(
    model: mujoco.MjModel,
    spec: RealManRM75BInspireSpec,
) -> None:
    assert (model.nq, model.nv, model.nu) == (40, 40, 40)
    assert spec.hand_type == "inspire_rh56dfx"
    assert spec.retarget_hand_type == "inspire_hand"
    assert spec.torso_joint_names == ()

    for side in ("left", "right"):
        for joint_name in spec.arm_joint_names(side) + spec.hand_joint_names(side):
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name(side))
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link")
        _object_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")

    for joint_name in spec.head_joint_names:
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)


def test_home_keyframe_maps_qpos_to_each_named_actuator(
    model: mujoco.MjModel,
    spec: RealManRM75BInspireSpec,
) -> None:
    data = _home_data(model)
    key_id = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    for side in ("left", "right"):
        actual = []
        for joint_name in spec.arm_joint_names(side):
            joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = model.jnt_qposadr[joint_id]
            value = data.qpos[qpos_address]
            lower, upper = model.jnt_range[joint_id]
            assert lower <= value <= upper
            actual.append(value)

            actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            assert model.key_ctrl[key_id, actuator_id] == pytest.approx(value)
        np.testing.assert_allclose(actual, spec.arm_home_qpos(side), atol=1e-12)


def test_home_pose_is_folded_forward_and_bilaterally_symmetric(
    model: mujoco.MjModel,
    spec: RealManRM75BInspireSpec,
) -> None:
    data = _home_data(model)
    positions = {}
    for side in ("left", "right"):
        hand_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link")
        positions[side] = data.xpos[hand_id].copy()
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        finger_direction = -hand_rotation[:, 1]
        assert finger_direction[0] > 0.98

        elbow_id = _object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{'l' if side == 'left' else 'r'}_link3",
        )
        assert data.xpos[elbow_id, 1] > 0.4 if side == "left" else data.xpos[elbow_id, 1] < -0.4

    left = positions["left"]
    right = positions["right"]
    assert left[0] == pytest.approx(right[0], abs=0.015)
    assert left[1] == pytest.approx(-right[1], abs=0.002)
    assert left[2] == pytest.approx(right[2], abs=0.002)
    assert 0.35 < left[0] < 0.42
    assert 0.30 < left[1] < 0.40
    assert 0.95 < left[2] < 1.05


def test_pico_translation_uses_project_world_axes_without_double_base_rotation(
    model: mujoco.MjModel,
    spec: RealManRM75BInspireSpec,
) -> None:
    data = _home_data(model)
    controller = TeleopArmController(
        model,
        data,
        mujoco.MjData(model),
        "left",
        SimpleNamespace(
            position_scale=1.0,
            ema_alpha=1.0,
            rot_weight=1.0,
            ik_damping=1e-3,
            ik_current_weight=0.1,
        ),
        hand_config=None,
        hand_type=spec.hand_type,
        home_qpos=spec.arm_home_qpos("left"),
        robot_label="realman_rm75b+inspire",
        arm_joint_names=spec.arm_joint_names("left"),
        hand_joint_names=spec.hand_joint_names("left"),
        ee_body_name=spec.ee_body_name("left"),
        base_body_name=spec.base_body_name,
        hand_qpos_mapping=spec.hand_qpos_mapping,
    )

    # The imported vendor base is rotated +90 degrees to align its geometry with
    # project world axes. Pico translation is already converted to those world
    # axes, so applying base_link.xmat again would swap/rotate X and Y while Z
    # appeared correct. RealMan therefore tracks translation directly in world.
    base_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert not np.allclose(data.xmat[base_id].reshape(3, 3), np.eye(3))
    assert spec.base_body_name is None

    pico_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    controller.update_from_pose(
        *transform_pico4_to_robot_pose(np.zeros(3), pico_quaternion)
    )
    initial_target_position = controller.target_position.copy()
    initial_target_quaternion = controller.target_quaternion.copy()

    cases = (
        (np.array([0.0, 0.0, -0.01]), np.array([0.01, 0.0, 0.0])),  # forward
        (np.array([0.01, 0.0, 0.0]), np.array([0.0, -0.01, 0.0])),  # right
        (np.array([0.0, 0.01, 0.0]), np.array([0.0, 0.0, 0.01])),  # up
    )
    for pico_delta, expected_robot_delta in cases:
        controller.update_from_pose(
            *transform_pico4_to_robot_pose(pico_delta, pico_quaternion)
        )
        np.testing.assert_allclose(
            controller.target_position - initial_target_position,
            expected_robot_delta,
            atol=1e-12,
        )
        # Translation-frame correction must not alter the already-correct wrist rotation.
        np.testing.assert_allclose(
            controller.target_quaternion, initial_target_quaternion, atol=1e-12
        )


def test_small_forward_ik_step_converges_for_both_arms(
    model: mujoco.MjModel,
    spec: RealManRM75BInspireSpec,
) -> None:
    data = _home_data(model)
    workspace = mujoco.MjData(model)

    for side in ("left", "right"):
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name(side))
        initial_position = data.xpos[body_id].copy()
        target_position = initial_position + np.array([0.01, 0.0, 0.0])
        target_quaternion = matrix_to_quaternion(data.xmat[body_id].reshape(3, 3))
        dof_indices = np.array(
            [
                model.jnt_dofadr[
                    _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                ]
                for joint_name in spec.arm_joint_names(side)
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

        inactive = np.ones(model.nq, dtype=bool)
        inactive[dof_indices] = False
        np.testing.assert_allclose(solution[inactive], data.qpos[inactive], atol=1e-12)


def test_scene_is_portable_and_does_not_use_vendor_fixed_hands() -> None:
    scene_text = SCENE.read_text()
    assert "/home/" not in scene_text
    assert "package://dual_75B_arm_robot" not in scene_text
    assert "l_hand_base_link.STL" not in scene_text
    assert "r_hand_base_link.STL" not in scene_text

    root = ET.fromstring(scene_text)
    for mesh in root.findall("./asset/mesh"):
        mesh_path = (SCENE.parent / mesh.get("file")).resolve()
        assert mesh_path.is_file(), mesh_path

    assert BODY_MESH.is_file()
    with BODY_MESH.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    assert triangle_count == 123_967
    assert triangle_count < 200_000


def test_teleop_registration_only_allows_realman_with_inspire(
    spec: RealManRM75BInspireSpec,
) -> None:
    assert teleop_sim._normalize_hand("realman_rm75b", None) == "inspire"
    assert teleop_sim._ROBOT_HAND_SPECS[("realman_rm75b", "inspire")] == spec
    with pytest.raises(ValueError, match="does not support hand"):
        teleop_sim._normalize_hand("realman_rm75b", "linker_l20")

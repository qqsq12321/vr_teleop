"""RealMan RM75-B + Linker L20 simulation scene and spec."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from example import teleop_sim
from utils.core.arm_controller import TeleopArmController
from utils.robots.dexforce import DexForceSpec
from utils.robots.realman_rm75b import (
    LEFT_ARM_HOME_QPOS as INSPIRE_LEFT_HOME,
    RIGHT_ARM_HOME_QPOS as INSPIRE_RIGHT_HOME,
)
from utils.robots.realman_rm75b_l20 import RealManRM75BL20Spec

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "example" / "scene_config" / "scene_realman_rm75b_l20.xml"
sys.path.insert(0, str(ROOT / "tools"))
import build_realman_rm75b_l20_scene as build_l20  # noqa: E402


@pytest.fixture(scope="module")
def spec() -> RealManRM75BL20Spec:
    return RealManRM75BL20Spec()


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


def _triad(finger: np.ndarray, across: np.ndarray) -> np.ndarray:
    finger = finger / np.linalg.norm(finger)
    across = across - finger * float(across @ finger)
    across /= np.linalg.norm(across)
    return np.column_stack([finger, across, np.cross(finger, across)])


def test_scene_loads_realman_l20_combination(
    model: mujoco.MjModel,
    spec: RealManRM75BL20Spec,
) -> None:
    # 21 arm/head joints plus 21 finger joints per hand, each with an actuator.
    assert (model.nq, model.nv, model.nu) == (58, 58, 58)
    assert spec.hand_type == "linker_l20"
    assert spec.retarget_hand_type == "linker_l20"
    assert spec.torso_joint_names == ()

    for side in ("left", "right"):
        assert len(spec.hand_joint_names(side)) == 21
        for joint_name in spec.arm_joint_names(side) + spec.hand_joint_names(side):
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body_name(side))
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link")
        _object_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")

    for joint_name in spec.head_joint_names:
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

    # The Inspire pair must be gone, not merely hidden inside the tree.
    for name in ("left_thumb_proximal_base", "right_thumb_proximal_base"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1


def test_home_keyframe_is_within_limits_and_matches_spec(
    model: mujoco.MjModel,
    spec: RealManRM75BL20Spec,
) -> None:
    data = _home_data(model)
    key_id = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    for side in ("left", "right"):
        actual = []
        for joint_name in spec.arm_joint_names(side):
            joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            value = data.qpos[model.jnt_qposadr[joint_id]]
            lower, upper = model.jnt_range[joint_id]
            assert lower <= value <= upper, f"{joint_name}={value} outside [{lower}, {upper}]"
            actual.append(value)

            actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            assert model.key_ctrl[key_id, actuator_id] == pytest.approx(value)
        np.testing.assert_allclose(actual, spec.arm_home_qpos(side), atol=1e-12)


def test_home_pose_is_the_inspire_pose_with_r_joint1_unwrapped(
    spec: RealManRM75BL20Spec,
) -> None:
    """Same arm pose as the Inspire spec, kept inside the model's limits.

    Unwrapping by 2*pi leaves forward kinematics unchanged but stops
    --control-mode ctrl from clamping the right arm to a different startup pose
    than --control-mode qpos produces. Only the left wrist roll differs, which
    test_home_pose_points_both_palms_down covers.
    """
    left = spec.arm_home_qpos("left")
    np.testing.assert_allclose(left[:6], INSPIRE_LEFT_HOME[:6], atol=1e-12)

    right = spec.arm_home_qpos("right")
    np.testing.assert_allclose(right[1:], INSPIRE_RIGHT_HOME[1:], atol=1e-12)
    assert right[0] == pytest.approx(INSPIRE_RIGHT_HOME[0] - 2.0 * np.pi, abs=1e-9)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    addresses = {
        side: [model.jnt_qposadr[model.joint(name).id] for name in spec.arm_joint_names(side)]
        for side in ("left", "right")
    }

    def right_tcp(right_qpos) -> np.ndarray:
        data.qpos[:] = 0.0
        data.qpos[addresses["left"]] = spec.arm_home_qpos("left")
        data.qpos[addresses["right"]] = right_qpos
        mujoco.mj_forward(model, data)
        return data.site_xpos[model.site("right_tcp").id].copy()

    np.testing.assert_allclose(right_tcp(right), right_tcp(INSPIRE_RIGHT_HOME), atol=1e-12)


def test_hands_adopt_the_mount_frame_convention_used_for_inspire(
    model: mujoco.MjModel,
) -> None:
    """Fingers point -Y and index->pinky runs +Z left / -Z right in the mount.

    The teleop IK and retarget mapping are written against this convention, so
    the grafted L20 hands have to reproduce it rather than their own canonical
    frame.
    """
    data = _home_data(model)
    expected = {
        "left": _triad(np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "right": _triad(np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    }
    for side in ("left", "right"):
        mount_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_end_effector_mount_link"
        )
        origin = data.xpos[mount_id]
        rotation = data.xmat[mount_id].reshape(3, 3)

        def local(name: str) -> np.ndarray:
            return rotation.T @ (data.xpos[_object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)] - origin)

        base = local(f"{side}_hand_base_link")
        measured = _triad(
            local(f"{side}_mf_dip_link") - base,
            local(f"{side}_if_dip_link") - local(f"{side}_lf_dip_link"),
        )
        relative = measured.T @ expected[side]
        degrees = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        assert degrees < 5.0, f"{side} hand frame is {degrees:.1f} deg off"


def test_home_pose_points_both_palms_down(model: mujoco.MjModel) -> None:
    """Both palms face down and mirror each other at the home pose.

    The mount-frame check above cannot see an inverted palm: it aligns the
    index->pinky axis, whose sign differs between the two source hands. Palm
    direction is read from where the fingers curl, which is unambiguous.
    """
    data = _home_data(model)
    key_id = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    palms = {}
    for side in ("left", "right"):
        palms[side] = build_l20.palm_direction(model, data, side)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

    for side, palm in palms.items():
        assert palm[2] < -0.2, f"{side} palm is not facing down: {palm.round(3)}"

    mirrored = palms["left"] * np.array([1.0, -1.0, 1.0])
    spread = np.degrees(np.arccos(np.clip(float(mirrored @ palms["right"]), -1.0, 1.0)))
    assert spread < 10.0, f"palms are not mirrored: {spread:.1f} deg apart"


def test_left_wrist_roll_differs_from_inspire_without_moving_the_tcp(
    model: mujoco.MjModel,
    spec: RealManRM75BL20Spec,
) -> None:
    """l_joint7 rotates the palm only; its axis runs through the TCP site."""
    left = spec.arm_home_qpos("left")
    assert left[6] != pytest.approx(INSPIRE_LEFT_HOME[6])

    joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_joint7")
    lower, upper = model.jnt_range[joint_id]
    assert lower <= left[6] <= upper

    data = mujoco.MjData(model)
    addresses = {
        side: [model.jnt_qposadr[model.joint(n).id] for n in spec.arm_joint_names(side)]
        for side in ("left", "right")
    }

    def left_tcp(left_qpos) -> np.ndarray:
        data.qpos[:] = 0.0
        data.qpos[addresses["left"]] = left_qpos
        data.qpos[addresses["right"]] = spec.arm_home_qpos("right")
        mujoco.mj_forward(model, data)
        return data.site_xpos[model.site("left_tcp").id].copy()

    np.testing.assert_allclose(left_tcp(left), left_tcp(INSPIRE_LEFT_HOME), atol=1e-12)


def test_home_pose_has_no_self_contact(model: mujoco.MjModel) -> None:
    data = _home_data(model)
    assert data.ncon == 0

    tips = [
        data.xpos[_object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_mf_dip_link")]
        for side in ("left", "right")
    ]
    # The L20 reaches ~48 mm further than Inspire, so the pair sits closer than
    # the Inspire scene's 0.249 m. Guard the gap rather than assume it.
    assert np.linalg.norm(tips[0] - tips[1]) > 0.15


def test_retarget_mapping_matches_the_shared_canonical_hand(
    spec: RealManRM75BL20Spec,
) -> None:
    """Both scenes embed the same canonical L20, so the tables must agree."""
    dexforce = DexForceSpec()
    assert spec.hand_qpos_mapping == dexforce.hand_qpos_mapping
    assert spec.hand_joint_suffixes == dexforce.hand_joint_suffixes
    assert sorted(spec.hand_qpos_mapping) == list(range(21))


def test_hand_config_resolves_per_input_source(spec: RealManRM75BL20Spec) -> None:
    for source in ("quest3", "avp", "pico4"):
        path = spec.hand_config_path(source, side="left")
        assert path.name == f"{source}_linker_l20.yaml"
        assert path.is_file(), f"Missing AnyDex config: {path}"
    # Unknown sources fall back to quest3, as the other specs do.
    assert spec.hand_config_path("mediapipe", side="left").name == "quest3_linker_l20.yaml"


def test_teleop_builds_both_arm_controllers(
    model: mujoco.MjModel,
    spec: RealManRM75BL20Spec,
) -> None:
    data = mujoco.MjData(model)
    teleop_sim._apply_initial_pose(model, data, spec)
    args = SimpleNamespace(
        position_scale=3.0,
        ema_alpha=0.8,
        rot_weight=1.0,
        ik_damping=0.001,
        ik_current_weight=0.1,
        control_mode="qpos",
    )
    for side in ("left", "right"):
        controller = TeleopArmController(
            model,
            data,
            mujoco.MjData(model),
            side,
            args,
            hand_config=None,
            hand_type=spec.hand_type,
            **teleop_sim._controller_kwargs(spec, side, "realman_rm75b+linker_l20"),
        )
        assert len(controller.arm_dof_indices) == 7
        assert len(controller.hand_dof_indices) == 21
        assert all(actuator != -1 for actuator in controller.arm_actuator_ids)
        assert all(actuator != -1 for actuator in controller.hand_actuator_ids)

        # Retarget output is reordered, not truncated, before it reaches qpos.
        controller.latest_hand_qpos = np.arange(21, dtype=float)
        mapped = controller.mapped_hand_qpos()
        assert mapped is not None and len(mapped) == 21


def test_scene_is_registered_for_the_sim_entry_point(spec: RealManRM75BL20Spec) -> None:
    assert teleop_sim._ROBOT_HAND_SPECS[("realman_rm75b", "linker_l20")].scene_path == SCENE
    assert teleop_sim._normalize_hand("realman_rm75b", "linker") == "linker_l20"


def test_visualizer_alias_resolves_to_the_scene() -> None:
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "_viz_under_test", ROOT / "example" / "viz" / "visualize.py"
    )
    assert module_spec is not None and module_spec.loader is not None
    visualize = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(visualize)

    alias = visualize._SCENE_ALIASES["realman_l20"]
    assert (ROOT / alias).resolve() == SCENE.resolve()

#!/usr/bin/env python3
"""Build the RealMan RM75-B + Linker L20 MuJoCo scene.

Grafts the checked-in ``linker_l20_{side}_mujoco.xml`` hands onto the arm tree of
``scene_realman_rm75b_inspire.xml``, replacing the Inspire pair.  Both source
scenes are checked in, so this has no vendor/xacro dependency -- unlike
``build_realman_rm75b_assets.py``, which rebuilds the arm body itself.

The L20 canonical frame (fingers +Z, palm normal +Y, index->pinky +X) is rotated
onto the mount frame convention the arm scene already uses for Inspire
(fingers -Y, palm normal -X left / +X right).  The two agree within 3 deg, so a
half-integer quaternion is exact enough; ``--check`` asserts it.
"""

from __future__ import annotations

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.robots.realman_rm75b_l20 import (  # noqa: E402
    LEFT_ARM_HOME_QPOS,
    RIGHT_ARM_HOME_QPOS,
)

SCENE_CONFIG_DIR = ROOT / "example" / "scene_config"
ARM_SCENE_PATH = SCENE_CONFIG_DIR / "scene_realman_rm75b_inspire.xml"
SCENE_PATH = SCENE_CONFIG_DIR / "scene_realman_rm75b_l20.xml"
L20_ASSET_DIR = ROOT / "assets" / "dex_hand" / "linker_l20"

# L20 canonical -> mount frame. Left: X->Z, Y->-X, Z->-Y; right mirrors across Y.
HAND_MOUNT_QUATERNION = {"left": "0.5 0.5 -0.5 0.5", "right": "0.5 0.5 0.5 -0.5"}
# Palm mass properties from assets/dex_hand/linker_l20/{side}_hand.urdf.xacro,
# rotated into the canonical MJCF frame (its palm geom sits at 0 0 0.036 with
# quat 0.707107 0 0 0.707107). fullinertia order is ixx iyy izz ixy ixz iyz.
HAND_BASE_INERTIAL = {
    "left": {
        "pos": "-0.000268049 -0.000986417 0.113379847",
        "mass": "0.110373496",
        "fullinertia": (
            "0.000159680178 0.000264402779 0.000118075546 "
            "-4.59263483e-07 4.78660407e-06 1.32798259e-06"
        ),
    },
    "right": {
        "pos": "0.000476951 -0.002355333 0.120027468",
        "mass": "0.130043891",
        "fullinertia": (
            "0.000164906211 0.000269869494 0.000118834524 "
            "5.66836729e-07 -4.94586056e-06 1.07919384e-06"
        ),
    },
}
# Visual STLs live beside the collision OBJs the canonical hand XML references.
L20_VISUAL_SUBDIR = "../../assets/dex_hand/linker_l20/visual/{side}_hand/{stem}.stl"
MODEL_NAME = "realman_rm75b_l20"
# 21 arm/head + 2 * 21 finger joints, each with a position actuator.
EXPECTED_DIMENSIONS = (58, 58, 58)


def _fmt(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    raise ValueError(f"Missing <{tag} name='{name}'>")


def _remove_inspire_hands(root: ET.Element) -> None:
    """Drop the Inspire bodies, their actuators, and their now-unused meshes."""
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("Arm scene is missing its actuator section")
    for element in list(actuator):
        if element.get("class") == "realman_hand":
            actuator.remove(element)

    for side in ("left", "right"):
        mount = _find_named(root, "body", f"{side}_arm_end_effector_mount_link")
        mount.remove(_find_named(mount, "body", f"{side}_hand_base_link"))

    asset = root.find("asset")
    if asset is None:
        raise ValueError("Arm scene is missing its asset section")
    referenced = {geom.get("mesh") for geom in root.iter("geom")}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in referenced:
            asset.remove(mesh)


def _add_l20_hand(root: ET.Element, side: str) -> None:
    hand_xml = ET.parse(L20_ASSET_DIR / f"linker_l20_{side}_mujoco.xml").getroot()
    source_worldbody = hand_xml.find("worldbody")
    source_asset = hand_xml.find("asset")
    if source_worldbody is None or source_asset is None:
        raise ValueError(f"Canonical L20 {side} hand is missing worldbody/asset")

    # Both canonical hands use identical mesh names, so prefix by side.
    asset = root.find("asset")
    assert asset is not None
    for mesh in source_asset.findall("mesh"):
        stem = Path(mesh.get("file", "")).stem
        asset.append(
            ET.Element(
                "mesh",
                {
                    "name": f"{side}_{mesh.get('name')}",
                    "content_type": "model/stl",
                    "file": L20_VISUAL_SUBDIR.format(side=side, stem=stem),
                },
            )
        )

    hand = ET.Element(
        "body",
        {"name": f"{side}_hand_base_link", "quat": HAND_MOUNT_QUATERNION[side]},
    )
    # The canonical hand carries its palm as bare worldbody geoms, so the
    # wrapper needs an explicit inertial: the geoms below are visual-only
    # (density 0) and MuJoCo rejects a zero-mass body with child joints.
    hand.append(ET.Element("inertial", dict(HAND_BASE_INERTIAL[side])))
    for child in source_worldbody:
        hand.append(copy.deepcopy(child))
    # Match the Inspire pair: visual-only geoms, mass carried by <inertial>.
    for geom in hand.iter("geom"):
        geom.set("mesh", f"{side}_{geom.get('mesh')}")
        geom.set("type", "mesh")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")
        geom.set("density", "0")

    actuator = root.find("actuator")
    assert actuator is not None
    for joint in hand.iter("joint"):
        joint.set("damping", "0.005")
        joint.set("armature", "0.00001")
        name = joint.get("name")
        joint_range = joint.get("range")
        if not name or not joint_range:
            raise ValueError(f"L20 {side} joint is missing a name/range: {name}")
        actuator.append(
            ET.Element(
                "position",
                {
                    "class": "realman_hand",
                    "name": name,
                    "joint": name,
                    "ctrlrange": joint_range,
                },
            )
        )

    _find_named(root, "body", f"{side}_arm_end_effector_mount_link").append(hand)


def _add_home_keyframe(root: ET.Element, scene_path: Path) -> None:
    """Keyframe the arm home pose; fingers and head stay at zero."""
    old = root.find("keyframe")
    if old is not None:
        root.remove(old)
    _write_xml(root, scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected RealMan+L20 dimensions {EXPECTED_DIMENSIONS}, got "
            f"({model.nq}, {model.nv}, {model.nu})"
        )
    data = mujoco.MjData(model)
    for prefix, values in (("l", LEFT_ARM_HOME_QPOS), ("r", RIGHT_ARM_HOME_QPOS)):
        for index, value in enumerate(values, 1):
            name = f"{prefix}_joint{index}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Missing joint while building home keyframe: {name}")
            lower, upper = model.jnt_range[joint_id]
            if not lower <= value <= upper:
                raise ValueError(
                    f"Home value {value} for {name} is outside [{lower}, {upper}]"
                )
            data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)

    ctrl = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": _fmt(data.qpos), "ctrl": _fmt(ctrl)},
    )
    _write_xml(root, scene_path)


def _triad(finger: np.ndarray, across: np.ndarray) -> np.ndarray:
    """Orthonormal frame from a finger axis and an index->pinky axis.

    The measured axes are not exactly perpendicular, so ``across`` is projected
    off ``finger``. Comparing the resulting frames as one rotation avoids
    counting the same tilt once per axis.
    """
    finger = finger / np.linalg.norm(finger)
    across = across - finger * float(across @ finger)
    across /= np.linalg.norm(across)
    return np.column_stack([finger, across, np.cross(finger, across)])


def _hand_frame(model, data, side: str) -> np.ndarray:
    """The hand's measured frame, expressed in its arm mount frame."""
    mount = model.body(f"{side}_arm_end_effector_mount_link").id
    origin = data.xpos[mount]
    rotation = data.xmat[mount].reshape(3, 3)

    def local(name: str) -> np.ndarray:
        return rotation.T @ (data.xpos[model.body(name).id] - origin)

    base = local(f"{side}_hand_base_link")
    return _triad(
        local(f"{side}_mf_dip_link") - base,
        local(f"{side}_if_dip_link") - local(f"{side}_lf_dip_link"),
    )


FINGERTIPS = ("if_dip_link", "mf_dip_link", "rf_dip_link", "lf_dip_link")


def palm_direction(model, data, side: str) -> np.ndarray:
    """World-frame direction the palm faces, from where the fingers curl.

    Fingers always flex toward the palm, so this has no sign ambiguity -- unlike
    the index->pinky axis, whose sign differs between the two source hands and
    which therefore cannot tell a palm-down pose from a palm-up one.

    Mutates ``data``; callers should re-apply the keyframe afterwards.
    """
    names = [f"{side}_{tip}" for tip in FINGERTIPS]
    before = np.array([data.xpos[model.body(name).id].copy() for name in names])
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if name.lower().startswith(side) and ("_MCP_PITCH" in name or "_PIP" in name):
            lower, upper = model.jnt_range[joint_id]
            data.qpos[model.jnt_qposadr[joint_id]] = lower + 0.6 * (upper - lower)
    mujoco.mj_forward(model, data)
    after = np.array([data.xpos[model.body(name).id].copy() for name in names])
    travel = (after - before).mean(axis=0)
    return travel / np.linalg.norm(travel)


def _check(scene_path: Path) -> None:
    """Verify the grafted hands sit in the frame the arm scene expects."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id == -1:
        raise ValueError("Generated scene is missing the home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Inspire's mount-frame convention, which the teleop IK and retarget
    # mapping are both written against: fingers -Y, index->pinky +Z left / -Z
    # right (palm normal follows as the cross product).
    expected = {
        "left": _triad(np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "right": _triad(np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    }
    for side in ("left", "right"):
        relative = _hand_frame(model, data, side).T @ expected[side]
        degrees = np.degrees(
            np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        )
        if degrees > 5.0:
            raise ValueError(
                f"{side} hand frame is {degrees:.1f} deg off the Inspire convention"
            )
        print(f"  {side:5s} hand frame {degrees:4.2f} deg off the Inspire convention")

    # The frame check above cannot catch an inverted palm, so check it directly.
    palms = {}
    for side in ("left", "right"):
        palms[side] = palm_direction(model, data, side)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
    for side, palm in palms.items():
        if palm[2] > -0.2:
            raise ValueError(
                f"{side} palm is not facing down at home: world direction "
                f"{palm.round(3).tolist()}"
            )
        print(f"  {side:5s} palm faces down (world Z {palm[2]:+.3f})")
    # A mirrored pair shares X and Z and flips Y.
    mirrored = palms["left"] * np.array([1.0, -1.0, 1.0])
    spread = np.degrees(
        np.arccos(np.clip(float(mirrored @ palms["right"]), -1.0, 1.0))
    )
    if spread > 10.0:
        raise ValueError(f"Palms are not mirrored: {spread:.1f} deg apart")
    print(f"  palms mirror each other within {spread:.2f} deg")

    if data.ncon:
        contacts = {
            tuple(
                sorted(
                    model.body(int(model.geom_bodyid[geom])).name
                    for geom in (data.contact[index].geom1, data.contact[index].geom2)
                )
            )
            for index in range(data.ncon)
        }
        raise ValueError(f"Home pose has {data.ncon} contact(s): {sorted(contacts)}")

    tips = [data.xpos[model.body(f"{side}_mf_dip_link").id] for side in ("left", "right")]
    print(f"  home fingertip gap {np.linalg.norm(tips[0] - tips[1]):.3f} m, no contacts")


def build(check: bool) -> None:
    root = ET.parse(ARM_SCENE_PATH).getroot()
    root.set("model", MODEL_NAME)
    _remove_inspire_hands(root)
    for side in ("left", "right"):
        _add_l20_hand(root, side)
    _add_home_keyframe(root, SCENE_PATH)
    print(f"Wrote {SCENE_PATH.relative_to(ROOT)}")
    if check:
        _check(SCENE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify hand frame alignment and home-pose clearance after building.",
    )
    args = parser.parse_args()
    build(args.check)


if __name__ == "__main__":
    main()

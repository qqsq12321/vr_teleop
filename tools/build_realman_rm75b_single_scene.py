#!/usr/bin/env python3
"""Build the single-arm RealMan RM75-B MuJoCo scene.

One RM75-B arm bolted to the floor, with no end effector beyond the flange.

The source is the vendor's standalone single-arm package (``RM75-B.urdf``),
not the dual-arm ``dual_75B_arm_robot`` xacro.  That matters for the flange:
the dual-arm model's ``*_link7`` sits part-way along the wrist, 0.105 m from
link6, and reaches the mount frame only via an extra hand-mount offset.  The
standalone URDF puts ``joint_7`` at the manual's d7 = 144 mm, so ``link_7`` is
the real flange face.  Being plain URDF, it also needs no ROS xacro binary.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_realman_rm75b_assets import (  # noqa: E402
    _find_named,
    _fmt,
    _remove_children,
    _set_joint_qpos,
    _write_xml,
)
from utils.robots.realman_rm75b_single import (  # noqa: E402
    ARM_JOINT_COUNT,
    FLANGE_BODY,
    PEDESTAL_BODY,
    SINGLE_ARM_HOME_QPOS,
    TCP_SITE,
    joint_name,
)

ASSET_ROOT = ROOT / "assets" / "arm_body" / "realman_rm75b_single"
SOURCE_URDF = ASSET_ROOT / "source" / "urdf" / "RM75-B.urdf"
VISUAL_MESH_DIR = ASSET_ROOT / "meshes" / "visual"
MESH_REL_DIR = "../../assets/arm_body/realman_rm75b_single/meshes/visual"
SCENE_PATH = ROOT / "example" / "scene_config" / "scene_realman_rm75b_single.xml"

BASE_LINK = "base_link"
TIP_LINK = f"link_{ARM_JOINT_COUNT}"
# Manual 6.2.4: the reachable envelope is a 610 mm sphere about the shoulder.
WORK_RADIUS_M = 0.61
# Manual 6.2.2 MDH: d1 (base to shoulder) and the RM75-B flange distance d7.
MANUAL_D1_M = 0.2405
MANUAL_REACH_M = 0.2405 + 0.256 + 0.210 + 0.144


def _clean_urdf(source: Path, clean_path: Path) -> None:
    """Point meshes at the checked-in copies and drop collision geometry."""
    root = ET.parse(source).getroot()
    for child in list(root):
        if child.tag in {"transmission", "gazebo"}:
            root.remove(child)

    for link in root.findall("link"):
        _remove_children(link, "collision")
        for mesh in link.findall("./visual/geometry/mesh"):
            filename = Path(mesh.get("filename", "")).name
            mesh_path = VISUAL_MESH_DIR / filename
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing visual mesh: {mesh_path}")
            mesh.set("filename", str(mesh_path.resolve()))

    mujoco_tag = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mujoco_tag,
        "compiler",
        {"fusestatic": "false", "discardvisual": "false", "strippath": "false"},
    )
    _write_xml(root, clean_path)


def _add_defaults(root: ET.Element, insert_at: int) -> None:
    defaults = ET.Element("default")
    arm = ET.SubElement(defaults, "default", {"class": "realman_arm"})
    ET.SubElement(arm, "position", {"kp": "500", "kv": "70", "forcerange": "-60 60"})
    root.insert(insert_at, defaults)


def _add_actuators(root: ET.Element) -> None:
    actuator = ET.SubElement(root, "actuator")
    joint_limits = {
        joint.get("name"): joint.get("range")
        for joint in root.findall(".//joint")
        if joint.get("name") and joint.get("range")
    }
    for index in range(1, ARM_JOINT_COUNT + 1):
        name = joint_name(index)
        ET.SubElement(
            actuator,
            "position",
            {
                "class": "realman_arm",
                "name": name,
                "joint": name,
                "ctrlrange": joint_limits[name],
            },
        )


def _assemble_scene(body_path: Path, scene_path: Path, pedestal_height: float) -> None:
    root = ET.parse(body_path).getroot()
    root.set("model", "realman_rm75b_single")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib.clear()
    compiler.set("angle", "radian")

    index = list(root).index(compiler)
    # Framed on a single arm's own envelope, not the dual-arm torso.
    center_z = pedestal_height + WORK_RADIUS_M / 2.0
    root.insert(
        index + 1,
        ET.Element(
            "statistic",
            {"center": f"0 0 {center_z:.3f}", "extent": f"{2.0 * WORK_RADIUS_M:.3f}"},
        ),
    )
    root.insert(
        index + 2,
        ET.Element(
            "option",
            {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
        ),
    )
    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "global",
        {"azimuth": "135", "elevation": "-18", "offwidth": "1600", "offheight": "1200"},
    )
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.4 0.4 0.4",
            "diffuse": "0.75 0.75 0.75",
            "specular": "0.15 0.15 0.15",
        },
    )
    root.insert(index + 3, visual)
    _add_defaults(root, index + 4)

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Converted MJCF is missing asset/worldbody")
    for mesh in asset.findall("mesh"):
        mesh.set("file", f"{MESH_REL_DIR}/{Path(mesh.get('file', '')).name}")

    # Re-parent the arm base onto a pedestal body. No joint between world and
    # the base, so the arm is rigidly bolted down.
    base = _find_named(worldbody, "body", BASE_LINK)
    worldbody.remove(base)
    pedestal = ET.Element(
        "body", {"name": PEDESTAL_BODY, "pos": f"0 0 {pedestal_height:.4f}"}
    )
    pedestal.append(base)
    worldbody.insert(0, pedestal)

    # link_7 already is the flange face (joint_7 carries the manual's d7), so
    # the mount frame sits on it with no extra offset -- unlike the dual-arm
    # scene, which needs a hand-mount translation to get here.
    tip = _find_named(root, "body", TIP_LINK)
    mount = ET.SubElement(tip, "body", {"name": FLANGE_BODY, "pos": "0 0 0"})
    ET.SubElement(
        mount,
        "site",
        {"name": TCP_SITE, "pos": "0 0 0", "size": "0.008", "rgba": "1 0.2 0.1 1"},
    )

    for joint in root.findall(".//joint"):
        if joint.get("name", "").startswith("joint_"):
            joint.set("damping", "0.5")
            joint.set("armature", "0.001")

    ET.SubElement(
        worldbody,
        "light",
        {"pos": "1 -1 3", "dir": "-1 1 -2", "diffuse": "0.8 0.8 0.8", "specular": "0.2 0.2 0.2"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "3 3 0.1",
            "rgba": "0.16 0.18 0.20 1",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    _add_actuators(root)
    _write_xml(root, scene_path)


def _verify_geometry(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Check the built arm against the manual before trusting the home pose."""
    print("Link world positions at the zero pose:")
    names = [BASE_LINK] + [f"link_{i}" for i in range(1, ARM_JOINT_COUNT + 1)] + [FLANGE_BODY]
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Missing body in generated scene: {name}")
        x, y, z = data.xpos[body_id]
        print(f"  {name:<34} x={x:+.4f} y={y:+.4f} z={z:+.4f}")

    base_z = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_LINK)][2])
    shoulder_z = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_1")][2])
    flange_z = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY)][2])

    if flange_z <= base_z:
        raise ValueError(
            f"Arm does not extend upward from its base at the zero pose "
            f"(flange {flange_z - base_z:+.4f} m relative to base); the "
            f"pedestal needs an alignment quaternion."
        )
    if abs((shoulder_z - base_z) - MANUAL_D1_M) > 1e-3:
        raise ValueError(
            f"Shoulder height {shoulder_z - base_z:.4f} m does not match the "
            f"manual's d1 = {MANUAL_D1_M} m"
        )
    if abs((flange_z - base_z) - MANUAL_REACH_M) > 1e-3:
        raise ValueError(
            f"Base-to-flange {flange_z - base_z:.4f} m does not match the "
            f"manual's d1+d3+d5+d7 = {MANUAL_REACH_M:.4f} m"
        )
    print(
        f"Matches the manual: shoulder d1={shoulder_z - base_z:.4f} m, "
        f"base-to-flange {flange_z - base_z:.4f} m."
    )


def _add_home_keyframe(scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    expected = (ARM_JOINT_COUNT,) * 3
    if (model.nq, model.nv, model.nu) != expected:
        raise ValueError(
            f"Expected single-arm RM75-B dimensions {expected}, got "
            f"({model.nq}, {model.nv}, {model.nu})"
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _verify_geometry(model, data)

    # _set_joint_qpos rejects any value outside the joint's range, which is the
    # guard the dual-arm home values silently violate on r_joint1 (+227 deg).
    for index, value in enumerate(SINGLE_ARM_HOME_QPOS, 1):
        _set_joint_qpos(model, data, joint_name(index), value)
    mujoco.mj_forward(model, data)

    ctrl = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]

    root = ET.parse(scene_path).getroot()
    old_keyframe = root.find("keyframe")
    if old_keyframe is not None:
        root.remove(old_keyframe)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe, "key", {"name": "home", "qpos": _fmt(data.qpos), "ctrl": _fmt(ctrl)}
    )
    _write_xml(root, scene_path)

    final_model = mujoco.MjModel.from_xml_path(str(scene_path))
    if mujoco.mj_name2id(final_model, mujoco.mjtObj.mjOBJ_KEY, "home") == -1:
        raise ValueError("Generated scene is missing the home keyframe")
    final_data = mujoco.MjData(final_model)
    mujoco.mj_resetDataKeyframe(final_model, final_data, 0)
    mujoco.mj_forward(final_model, final_data)
    flange_id = mujoco.mj_name2id(final_model, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY)
    print(f"Home flange position: {_fmt(final_data.xpos[flange_id])}")


def build(pedestal_height: float) -> None:
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(f"Missing vendor single-arm URDF: {SOURCE_URDF}")

    with tempfile.TemporaryDirectory(prefix="rm75b_single_build_") as temp_string:
        temp = Path(temp_string)
        clean_urdf = temp / "clean.urdf"
        body_mjcf = temp / "body.xml"
        _clean_urdf(SOURCE_URDF, clean_urdf)
        model = mujoco.MjModel.from_xml_path(str(clean_urdf))
        mujoco.mj_saveLastXML(str(body_mjcf), model)
        _assemble_scene(body_mjcf, SCENE_PATH, pedestal_height)
        _add_home_keyframe(SCENE_PATH)
    print(f"Wrote {SCENE_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pedestal-height",
        type=float,
        default=0.0,
        help="Height of the arm base above the floor in metres (default: 0.0).",
    )
    args = parser.parse_args()
    build(args.pedestal_height)


if __name__ == "__main__":
    main()

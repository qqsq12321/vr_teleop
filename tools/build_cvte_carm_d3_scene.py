#!/usr/bin/env python3
"""Build the CVTE CARM D3 MuJoCo scene from the checked-in vendor URDF.

The checked-in scene has no runtime dependency on the vendor ROS package;
rebuild it any time from `assets/arm_body/cvte_carm_d3/source/urdf/carm_d3.urdf`:

    python tools/build_cvte_carm_d3_scene.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.robots.cvte_carm_d3 import CVTECArmD3Spec  # noqa: E402

ASSET_ROOT = ROOT / "assets" / "arm_body" / "cvte_carm_d3"
SOURCE_URDF = ASSET_ROOT / "source" / "urdf" / "carm_d3.urdf"
MESH_DIR_REL = "../../assets/arm_body/cvte_carm_d3/meshes"
SCENE_PATH = ROOT / "example" / "scene_config" / "scene_cvte_carm_d3.xml"

# (first_joint_index, last_joint_index, actuator class, kp, kv)
ARM_JOINT_TIERS = (
    (1, 2, "cvte_arm_heavy", "600", "60"),
    (3, 4, "cvte_arm_mid", "400", "40"),
    (5, 7, "cvte_arm_light", "150", "15"),
)
GRIPPER_CLASS = "cvte_gripper"

# The vendor kinematics hang both arms straight down (-Z) from base_link, as
# designed for a shoulder-height overhead mount. Raise base_link so the
# all-zero home pose keeps the grippers comfortably above a floor/tabletop.
MOUNT_HEIGHT = 1.3


def _fmt(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    element = root.find(f".//{tag}[@name='{name}']")
    if element is None:
        raise ValueError(f"Could not find {tag} named {name}")
    return element


def _convert_urdf_to_mjcf(urdf_path: Path, body_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(body_path), model)


def _arm_tier_for_index(index: int) -> tuple[str, str, str]:
    for lo, hi, cls, kp, kv in ARM_JOINT_TIERS:
        if lo <= index <= hi:
            return cls, kp, kv
    raise ValueError(f"No actuator tier configured for joint index {index}")


def _add_defaults(root: ET.Element, insert_at: int) -> None:
    defaults = ET.Element("default")
    top = ET.SubElement(defaults, "default", {"class": "cvte"})
    visual = ET.SubElement(top, "default", {"class": "cvte/visual"})
    ET.SubElement(
        visual,
        "geom",
        {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"},
    )
    for _, _, cls, kp, kv in ARM_JOINT_TIERS:
        child = ET.SubElement(defaults, "default", {"class": cls})
        ET.SubElement(child, "position", {"kp": kp, "kv": kv})
    gripper = ET.SubElement(defaults, "default", {"class": GRIPPER_CLASS})
    ET.SubElement(gripper, "position", {"kp": "50", "kv": "5"})
    root.insert(insert_at, defaults)


def _add_scene_dressing(root: ET.Element, insert_at: int) -> None:
    asset = root.find("asset")
    assert asset is not None
    skybox = ET.Element(
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.55 0.58 0.62",
            "rgb2": "0.06 0.07 0.08",
            "width": "512",
            "height": "3072",
        },
    )
    ground_tex = ET.Element(
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.45 0.47 0.47",
            "rgb2": "0.28 0.30 0.30",
            "markrgb": "0.38 0.40 0.40",
            "width": "300",
            "height": "300",
        },
    )
    ground_mat = ET.Element(
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "6 6",
            "reflectance": "0.18",
        },
    )
    asset.insert(0, skybox)
    asset.insert(1, ground_tex)
    asset.insert(2, ground_mat)

    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.7 0.7 0.7", "ambient": "0.28 0.28 0.28", "specular": "0.2 0.2 0.2"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.20 0.26 1"})
    ET.SubElement(
        visual,
        "global",
        {"azimuth": "135", "elevation": "-18", "offwidth": "1600", "offheight": "1200"},
    )
    root.insert(insert_at, visual)


def _assemble_scene(body_path: Path, scene_path: Path) -> None:
    root = ET.parse(body_path).getroot()
    root.set("model", "cvte_carm_d3")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib.clear()
    compiler.set("angle", "radian")
    compiler.set("balanceinertia", "true")
    compiler_index = list(root).index(compiler)

    root.insert(
        compiler_index + 1,
        ET.Element("statistic", {"center": "0 0 0.9", "extent": "1.8"}),
    )
    root.insert(
        compiler_index + 2,
        ET.Element("option", {"integrator": "implicitfast"}),
    )
    _add_scene_dressing(root, compiler_index + 3)
    _add_defaults(root, compiler_index + 4)

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Converted MJCF is missing asset/worldbody")

    for mesh in asset.findall("mesh"):
        filename = Path(mesh.get("file", "")).name
        mesh.set("file", f"{MESH_DIR_REL}/{filename}")

    for geom in worldbody.iter("geom"):
        if geom.get("mesh") is not None:
            geom.set("class", "cvte/visual")

    base = _find_named(worldbody, "body", "base_link")
    worldbody.remove(base)
    mount = ET.Element(
        "body",
        {"name": "cvte_mount", "pos": f"0 0 {MOUNT_HEIGHT:g}"},
    )
    mount.append(base)
    worldbody.insert(0, mount)

    worldbody.insert(
        0,
        ET.Element(
            "light",
            {"pos": "0 -1.2 2.2", "dir": "0 0.4 -1", "directional": "true"},
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "geom",
            {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"},
        ),
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "scene_cam",
            "pos": "1.8 0.0 0.70",
            "xyaxes": "0 1 0 0 0 1",
            "fovy": "45",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {"name": "front_cam", "pos": "1.3 1.3 1.0", "xyaxes": "-0.71 0.71 0 -0.1 -0.1 0.99", "fovy": "45"},
    )

    # Replicate the vendor URDF's dropped <mimic> tags as explicit equality
    # constraints (MuJoCo's URDF importer does not convert <mimic> on its own).
    equality = ET.SubElement(root, "equality")
    for side in ("LEFT", "RIGHT"):
        ET.SubElement(
            equality,
            "joint",
            {
                "joint1": f"gripper_left_joint-{side}",
                "joint2": f"gripper_right_joint-{side}",
                "polycoef": "0 1 0 0 0",
            },
        )

    _add_actuators(root)
    _write_xml(root, scene_path)


def _add_actuators(root: ET.Element) -> None:
    joint_info: dict[str, tuple[str, str]] = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name and joint.get("range"):
            joint_info[name] = (joint.get("range"), joint.get("actuatorfrcrange", ""))

    actuator = ET.SubElement(root, "actuator")
    for side in ("LEFT", "RIGHT"):
        for index in range(1, 8):
            name = f"joint{index}-{side}"
            ctrlrange, forcerange = joint_info[name]
            cls, _, _ = _arm_tier_for_index(index)
            attrs = {"class": cls, "name": name, "joint": name, "ctrlrange": ctrlrange}
            if forcerange:
                attrs["forcerange"] = forcerange
            ET.SubElement(actuator, "position", attrs)
        gripper_name = f"gripper_right_joint-{side}"
        ctrlrange, forcerange = joint_info[gripper_name]
        attrs = {
            "class": GRIPPER_CLASS,
            "name": gripper_name,
            "joint": gripper_name,
            "ctrlrange": ctrlrange,
        }
        if forcerange:
            attrs["forcerange"] = forcerange
        ET.SubElement(actuator, "position", attrs)


def _add_home_keyframe(scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    expected_nq = 2 * (7 + 2)  # 7 arm joints + 2 (mimicked) gripper joints, per side
    expected_nu = 2 * (7 + 1)  # 7 arm actuators + 1 driven gripper actuator, per side
    if (model.nq, model.nu, model.neq) != (expected_nq, expected_nu, 2):
        raise ValueError(
            f"Expected (nq, nu, neq) == ({expected_nq}, {expected_nu}, 2), got "
            f"({model.nq}, {model.nu}, {model.neq})"
        )
    data = mujoco.MjData(model)
    spec = CVTECArmD3Spec()
    for side in ("left", "right"):
        for name, value in zip(spec.arm_joint_names(side), spec.arm_home_qpos(side)):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Missing joint while building home keyframe: {name}")
            lower, upper = model.jnt_range[joint_id]
            if not lower <= value <= upper:
                raise ValueError(f"Home value {value} for {name} is outside [{lower}, {upper}]")
            data.qpos[model.jnt_qposadr[joint_id]] = value
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
        keyframe,
        "key",
        {"name": "home", "qpos": _fmt(data.qpos), "ctrl": _fmt(ctrl)},
    )
    _write_xml(root, scene_path)

    final_model = mujoco.MjModel.from_xml_path(str(scene_path))
    key_id = mujoco.mj_name2id(final_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id == -1:
        raise ValueError("Generated scene is missing the home keyframe")


def build() -> None:
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(f"Missing vendor URDF: {SOURCE_URDF}")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="cvte_carm_d3_build_") as temp_string:
        body_mjcf = Path(temp_string) / "body.xml"
        _convert_urdf_to_mjcf(SOURCE_URDF, body_mjcf)
        _assemble_scene(body_mjcf, SCENE_PATH)
        _add_home_keyframe(SCENE_PATH)
    print(f"Wrote {SCENE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    build()

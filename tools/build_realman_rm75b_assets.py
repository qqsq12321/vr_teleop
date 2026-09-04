#!/usr/bin/env python3
"""Build the RealMan RM75-B + Inspire RH56DFX MuJoCo scene.

The checked-in scene has no runtime dependency on the vendor ROS package.  Use
``--vendor-dir`` when importing a newer export; subsequent rebuilds can use the
checked-in xacro sources and visual meshes directly.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.robots.realman_rm75b import (  # noqa: E402
    LEFT_ARM_HOME_QPOS,
    RIGHT_ARM_HOME_QPOS,
)

ASSET_ROOT = ROOT / "assets" / "arm_body" / "realman_rm75b"
SOURCE_URDF_DIR = ASSET_ROOT / "source" / "urdf"
VISUAL_MESH_DIR = ASSET_ROOT / "meshes" / "visual"
SCENE_PATH = ROOT / "example" / "scene_config" / "scene_realman_rm75b_inspire.xml"
INSPIRE_SCENE_PATH = ROOT / "example" / "scene_config" / "scene_galbot_inspire.xml"

MODEL_XACROS = (
    "agv.urdf.xacro",
    "body_head.urdf.xacro",
    "left_hand.urdf.xacro",
    "right_hand.urdf.xacro",
    "rm75_B_left.urdf.xacro",
    "rm75_B_right.urdf.xacro",
    "joint.urdf.xacro",
)
FIXED_HAND_LINKS = {"l_hand_base_link", "r_hand_base_link"}
FIXED_HAND_JOINTS = {"l_arm_hand_joint", "r_arm_hand_joint"}
FIXED_HAND_MESHES = {"l_hand_base_link.STL", "r_hand_base_link.STL"}
ROOT_POSITION = "0 0 0.08"
ROOT_QUATERNION = "0.7071067811865476 0 0 0.7071067811865475"
HAND_MOUNT_POSITION = "0 0 0.03"
HAND_MOUNT_QUATERNION = "0.5 -0.5 -0.5 0.5"


def _fmt(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _import_vendor_sources(vendor_dir: Path, voxel_size: float) -> None:
    vendor_urdf = vendor_dir / "urdf"
    vendor_meshes = vendor_dir / "meshes"
    if not vendor_urdf.is_dir() or not vendor_meshes.is_dir():
        raise FileNotFoundError(
            f"Expected a ROS package with urdf/ and meshes/ under {vendor_dir}"
        )

    SOURCE_URDF_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(vendor_urdf.glob("*.xacro")):
        shutil.copy2(source, SOURCE_URDF_DIR / source.name)

    VISUAL_MESH_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(vendor_meshes.glob("*.STL")):
        if source.name in FIXED_HAND_MESHES:
            continue
        destination = VISUAL_MESH_DIR / source.name
        if source.name == "body_base_link.STL":
            _cluster_binary_stl(source, destination, voxel_size)
        else:
            shutil.copy2(source, destination)


def _read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"STL is too short: {path}")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if len(raw) != expected_size:
        raise ValueError(f"Only binary STL input is supported: {path}")
    records = np.frombuffer(
        raw,
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
        offset=84,
        count=triangle_count,
    )
    return np.asarray(records["vertices"], dtype=np.float64)


def _cluster_binary_stl(source: Path, destination: Path, voxel_size: float) -> None:
    """Reduce a binary STL using deterministic 3-D vertex clustering."""
    triangles = _read_binary_stl(source)
    flat_vertices = triangles.reshape(-1, 3)
    voxel_keys = np.rint(flat_vertices / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)

    vertex_count = int(inverse.max()) + 1
    clustered = np.empty((vertex_count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=vertex_count)
    for axis in range(3):
        clustered[:, axis] = (
            np.bincount(inverse, weights=flat_vertices[:, axis], minlength=vertex_count)
            / counts
        )

    faces = inverse.reshape(-1, 3)
    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[nondegenerate]
    # Remove coincident triangles independent of winding, keeping first winding.
    _, unique_indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique_indices)]
    output_triangles = clustered[faces]

    edges_a = output_triangles[:, 1] - output_triangles[:, 0]
    edges_b = output_triangles[:, 2] - output_triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    output_triangles = output_triangles[valid]
    normals = normals[valid] / lengths[valid, None]

    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"RealMan body_base_link clustered at {voxel_size:g} m by vr_teleop"
        .encode("ascii")[:80]
        .ljust(80, b"\0")
    )
    record_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.zeros(len(output_triangles), dtype=record_dtype)
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = output_triangles.astype(np.float32)
    with destination.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(records)))
        stream.write(records.tobytes())
    print(
        f"Simplified {source.name}: {len(triangles)} -> {len(records)} triangles "
        f"({vertex_count} clustered vertices)"
    )


def _make_xacro_wrapper(path: Path) -> None:
    root = ET.Element(
        "robot",
        {
            "name": "dual_75B_arm_robot",
            "xmlns:xacro": "http://ros.org/wiki/xacro",
        },
    )
    for name in MODEL_XACROS:
        source = SOURCE_URDF_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing source xacro: {source}")
        # Keep the literal xacro prefix. ROS xacro 2.x does not expand an
        # ElementTree-generated ns0:include even when it has the same URI.
        ET.SubElement(root, "xacro:include", {"filename": str(source.resolve())})
    _write_xml(root, path)


def _remove_children(parent: ET.Element, tag: str) -> None:
    for child in list(parent):
        if child.tag == tag:
            parent.remove(child)


def _clean_urdf(raw_path: Path, clean_path: Path) -> None:
    root = ET.parse(raw_path).getroot()

    for child in list(root):
        if child.tag in {"transmission", "gazebo"}:
            root.remove(child)
        elif child.tag == "link" and child.get("name") in FIXED_HAND_LINKS:
            root.remove(child)
        elif child.tag == "joint" and child.get("name") in FIXED_HAND_JOINTS:
            root.remove(child)

    for link in root.findall("link"):
        _remove_children(link, "collision")
        for mesh in link.findall("./visual/geometry/mesh"):
            filename = Path(mesh.get("filename", "")).name
            mesh_path = VISUAL_MESH_DIR / filename
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing imported visual mesh: {mesh_path}")
            mesh.set("filename", str(mesh_path.resolve()))

    for joint in root.findall("joint"):
        if "wheel" in joint.get("name", ""):
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "calibration", "safety_controller"):
                _remove_children(joint, tag)

    mujoco_tag = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mujoco_tag,
        "compiler",
        {"fusestatic": "false", "discardvisual": "false", "strippath": "false"},
    )
    _write_xml(root, clean_path)


def _convert_urdf_to_mjcf(clean_path: Path, body_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(clean_path))
    mujoco.mj_saveLastXML(str(body_path), model)


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    element = root.find(f".//{tag}[@name='{name}']")
    if element is None:
        raise ValueError(f"Could not find {tag} named {name}")
    return element


def _add_defaults(root: ET.Element, insert_at: int) -> None:
    defaults = ET.Element("default")
    for name, kp, kv, force in (
        ("realman_arm", "500", "70", "-60 60"),
        ("realman_hand", "5", "0.35", "-5 5"),
        ("realman_head", "20", "4", "-20 20"),
    ):
        child = ET.SubElement(defaults, "default", {"class": name})
        ET.SubElement(child, "position", {"kp": kp, "kv": kv, "forcerange": force})
    root.insert(insert_at, defaults)


def _add_inspire_hand(
    scene_root: ET.Element,
    inspire_root: ET.Element,
    side: str,
) -> None:
    hand_name = f"{side}_hand_base_link"
    source_hand = copy.deepcopy(_find_named(inspire_root, "body", hand_name))
    source_hand.attrib.pop("pos", None)
    source_hand.attrib.pop("quat", None)

    arm_prefix = "l" if side == "left" else "r"
    parent = _find_named(scene_root, "body", f"{arm_prefix}_link7")
    mount = ET.SubElement(
        parent,
        "body",
        {
            "name": f"{side}_arm_end_effector_mount_link",
            "pos": HAND_MOUNT_POSITION,
            "quat": HAND_MOUNT_QUATERNION,
        },
    )
    ET.SubElement(
        mount,
        "site",
        {"name": f"{side}_tcp", "pos": "0 0 0", "size": "0.008", "rgba": "1 0.2 0.1 1"},
    )
    mount.append(source_hand)

    referenced_meshes = {
        geom.get("mesh") for geom in source_hand.iter("geom") if geom.get("mesh")
    }
    target_asset = scene_root.find("asset")
    source_asset = inspire_root.find("asset")
    assert target_asset is not None and source_asset is not None
    source_meshes = {mesh.get("name"): mesh for mesh in source_asset.findall("mesh")}
    for mesh_name in sorted(referenced_meshes):
        target_asset.append(copy.deepcopy(source_meshes[mesh_name]))


def _add_actuators(root: ET.Element, inspire_root: ET.Element) -> None:
    actuator = ET.SubElement(root, "actuator")
    joint_limits: dict[str, str] = {}
    for joint in root.findall(".//joint"):
        if joint.get("name") and joint.get("range"):
            joint_limits[joint.get("name")] = joint.get("range")

    for joint_name in ("head_joint1", "head_joint2"):
        ET.SubElement(
            actuator,
            "position",
            {
                "class": "realman_head",
                "name": joint_name,
                "joint": joint_name,
                "ctrlrange": joint_limits[joint_name],
            },
        )
    for prefix in ("l", "r"):
        for index in range(1, 8):
            joint_name = f"{prefix}_joint{index}"
            ET.SubElement(
                actuator,
                "position",
                {
                    "class": "realman_arm",
                    "name": joint_name,
                    "joint": joint_name,
                    "ctrlrange": joint_limits[joint_name],
                },
            )

    source_actuator = inspire_root.find("actuator")
    assert source_actuator is not None
    for side in ("left", "right"):
        prefix = f"{side}_"
        for source in source_actuator:
            joint_name = source.get("joint", "")
            if not joint_name.startswith(prefix) or "_arm_" in joint_name:
                continue
            target = copy.deepcopy(source)
            target.set("class", "realman_hand")
            actuator.append(target)


def _decorate_joints(root: ET.Element) -> None:
    for joint in root.findall(".//joint"):
        name = joint.get("name", "")
        if name.startswith(("l_joint", "r_joint")):
            joint.set("damping", "0.5")
            joint.set("armature", "0.001")
        elif name.startswith("head_joint"):
            joint.set("damping", "0.2")
            joint.set("armature", "0.001")


def _assemble_scene(body_path: Path, scene_path: Path) -> None:
    root = ET.parse(body_path).getroot()
    root.set("model", "realman_rm75b_inspire")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib.clear()
    compiler.set("angle", "radian")

    compiler_index = list(root).index(compiler)
    root.insert(
        compiler_index + 1,
        ET.Element("statistic", {"center": "0.22 0 0.72", "extent": "2.2"}),
    )
    root.insert(
        compiler_index + 2,
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
    root.insert(compiler_index + 3, visual)
    _add_defaults(root, compiler_index + 4)

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Converted MJCF is missing asset/worldbody")
    for mesh in asset.findall("mesh"):
        filename = Path(mesh.get("file", "")).name
        mesh.set(
            "file",
            f"../../assets/arm_body/realman_rm75b/meshes/visual/{filename}",
        )

    base = _find_named(worldbody, "body", "base_link")
    worldbody.remove(base)
    platform = ET.Element(
        "body",
        {"name": "realman_teleop_base", "pos": ROOT_POSITION, "quat": ROOT_QUATERNION},
    )
    platform.append(base)
    worldbody.insert(0, platform)

    inspire_root = ET.parse(INSPIRE_SCENE_PATH).getroot()
    _add_inspire_hand(root, inspire_root, "left")
    _add_inspire_hand(root, inspire_root, "right")
    _decorate_joints(root)

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
    _add_actuators(root, inspire_root)
    _write_xml(root, scene_path)


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Missing joint while building home keyframe: {name}")
    lower, upper = model.jnt_range[joint_id]
    if not lower <= value <= upper:
        raise ValueError(f"Home value {value} for {name} is outside [{lower}, {upper}]")
    data.qpos[model.jnt_qposadr[joint_id]] = value


def _add_home_keyframe(scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu) != (40, 40, 40):
        raise ValueError(
            f"Expected RealMan+Inspire dimensions (40, 40, 40), got "
            f"({model.nq}, {model.nv}, {model.nu})"
        )
    data = mujoco.MjData(model)
    for side, prefix, values in (
        ("left", "l", LEFT_ARM_HOME_QPOS),
        ("right", "r", RIGHT_ARM_HOME_QPOS),
    ):
        del side
        for index, value in enumerate(values, 1):
            _set_joint_qpos(model, data, f"{prefix}_joint{index}", value)
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


def build(vendor_dir: Path | None, xacro_bin: str, voxel_size: float) -> None:
    if vendor_dir is not None:
        _import_vendor_sources(vendor_dir.resolve(), voxel_size)
    if not SOURCE_URDF_DIR.is_dir() or not VISUAL_MESH_DIR.is_dir():
        raise FileNotFoundError(
            "RealMan sources/assets are missing; pass --vendor-dir for the first import"
        )

    with tempfile.TemporaryDirectory(prefix="realman_rm75b_build_") as temp_string:
        temp = Path(temp_string)
        wrapper = temp / "dual_75B_arm_robot.absolute.urdf.xacro"
        raw_urdf = temp / "raw.urdf"
        clean_urdf = temp / "clean.urdf"
        body_mjcf = temp / "body.xml"
        _make_xacro_wrapper(wrapper)
        subprocess.run(
            [xacro_bin, str(wrapper), "-o", str(raw_urdf)],
            check=True,
            cwd=ROOT,
        )
        _clean_urdf(raw_urdf, clean_urdf)
        _convert_urdf_to_mjcf(clean_urdf, body_mjcf)
        _assemble_scene(body_mjcf, SCENE_PATH)
        _add_home_keyframe(SCENE_PATH)
    print(f"Wrote {SCENE_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=None,
        help="Vendor dual_75B_arm_robot ROS package to import before building.",
    )
    parser.add_argument("--xacro-bin", default=shutil.which("xacro") or "xacro")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.001,
        help="Vertex-clustering grid for body_base_link.STL in metres (default: 0.001).",
    )
    args = parser.parse_args()
    build(args.vendor_dir, args.xacro_bin, args.voxel_size)


if __name__ == "__main__":
    main()

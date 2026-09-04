import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from utils.core.hand.retarget import default_pico4_config_path
from utils.robots.dexforce import DexForceSpec

ROOT = Path(__file__).resolve().parents[1]
ANYDEX_ROOT = ROOT / "third_party" / "AnyDexRetarget"
ANYDEX_CONFIG = ANYDEX_ROOT / "example" / "config" / "adaptive" / "quest3" / "quest3_linker_l20.yaml"

# DexForce actuator order expressed with AnyDex's semantic joint names.
DEXFORCE_RETARGET_JOINT_NAMES = (
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "index_dip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "middle_dip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "ring_dip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "pinky_dip",
)

GEORT_L20_URDF = {
    side: ANYDEX_ROOT
    / "assets"
    / "linker_l20"
    / f"geort_{side}"
    / f"linkerhand_l20_{side}.urdf"
    for side in ("left", "right")
}
MCP_ROLL_JOINTS = {
    "IF": "index_mcp_roll",
    "MF": "middle_mcp_roll",
    "RF": "ring_mcp_roll",
    "LF": "pinky_mcp_roll",
}
DEXFORCE_L20_DESCRIPTIONS = (
    ("left", ROOT / "assets/dex_hand/linker_l20/left_hand.urdf.xacro"),
    ("left", ROOT / "assets/dex_hand/linker_l20/linker_l20_left_vis.urdf"),
    ("left", ROOT / "assets/dex_hand/linker_l20/linker_l20_left_mujoco.xml"),
    ("right", ROOT / "assets/dex_hand/linker_l20/right_hand.urdf.xacro"),
    ("right", ROOT / "assets/dex_hand/linker_l20/linker_l20_right_vis.urdf"),
    ("right", ROOT / "assets/dex_hand/linker_l20/linker_l20_right_mujoco.xml"),
    *(
        (side, ROOT / f"example/scene_config/{scene_name}")
        for scene_name in (
            "scene_dexforce.xml",
            "scene_dexforce_head2test.xml",
            "scene_dexforce_head_axes.xml",
        )
        for side in ("left", "right")
    ),
)


def _joint_axis(path: Path, joint_name: str) -> tuple[float, float, float]:
    matches = [
        joint
        for joint in ET.parse(path).getroot().iter("joint")
        if joint.get("name") == joint_name
    ]
    assert len(matches) == 1, (
        f"Expected one {joint_name} joint in {path}, found {len(matches)}"
    )

    joint = matches[0]
    axis_text = joint.get("axis")
    if axis_text is None:
        axis = joint.find("axis")
        assert axis is not None, f"Missing axis for {joint_name} in {path}"
        axis_text = axis.get("xyz")
    assert axis_text is not None
    values = tuple(float(value) for value in axis_text.split())
    assert len(values) == 3
    return values


def _anydex_l20_joint_names(side: str) -> tuple[str, ...]:
    sys.path.insert(0, str(ANYDEX_ROOT))
    try:
        from anydexretarget import Retargeter
    finally:
        sys.path.pop(0)

    retargeter = Retargeter.from_yaml(str(ANYDEX_CONFIG), side)
    return tuple(retargeter.optimizer.robot.dof_joint_names)


def test_dexforce_mapping_matches_current_anydex_l20_order() -> None:
    spec = DexForceSpec()

    for side in ("left", "right"):
        source_joint_names = _anydex_l20_joint_names(side)
        expected_mapping = tuple(
            source_joint_names.index(name) for name in DEXFORCE_RETARGET_JOINT_NAMES
        )
        assert spec.hand_qpos_mapping == expected_mapping


def test_pico4_uses_side_aware_shared_l20_config() -> None:
    spec = DexForceSpec()

    for side in ("left", "right"):
        core_path = default_pico4_config_path(side)
        spec_path = spec.hand_config_path("pico4", side)

        assert core_path == spec_path
        assert core_path.name == "pico4_linker_l20.yaml"
        assert core_path.is_file()


@pytest.mark.parametrize(
    ("side", "description_path"),
    DEXFORCE_L20_DESCRIPTIONS,
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_dexforce_mcp_roll_axes_match_current_geort_l20(
    side: str, description_path: Path
) -> None:
    geort_path = GEORT_L20_URDF[side]
    prefix = side.upper()

    for dexforce_finger, geort_joint_name in MCP_ROLL_JOINTS.items():
        dexforce_joint_name = f"{prefix}_{dexforce_finger}_MCP_ROLL"
        assert _joint_axis(description_path, dexforce_joint_name) == _joint_axis(
            geort_path, geort_joint_name
        )

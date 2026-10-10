"""DexForce robot specification and default asset paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT_DIR / "example" / "scene_config" / "scene_dexforce.xml"
ANYDEX_ADAPTIVE_CONFIG_DIR = ROOT_DIR / "third_party" / "AnyDexRetarget" / "example" / "config" / "adaptive"


@dataclass(frozen=True)
class DexForceSpec:
    scene_path: Path = DEFAULT_SCENE
    hand_type: str = "linker_l20"
    base_body_name: str = "base_link"
    arm_joint_suffixes: tuple[str, ...] = tuple(f"J{i}" for i in range(1, 8))
    hand_joint_suffixes: tuple[str, ...] = (
        "T_CMC_ROLL",
        "T_CMC_YAW",
        "T_CMC_PITCH",
        "T_MCP",
        "T_IP",
        "IF_MCP_ROLL",
        "IF_MCP_PITCH",
        "IF_PIP",
        "IF_DIP",
        "MF_MCP_ROLL",
        "MF_MCP_PITCH",
        "MF_PIP",
        "MF_DIP",
        "RF_MCP_ROLL",
        "RF_MCP_PITCH",
        "RF_PIP",
        "RF_DIP",
        "LF_MCP_ROLL",
        "LF_MCP_PITCH",
        "LF_PIP",
        "LF_DIP",
    )
    # AnyDex Linker L20 output order is index, middle, pinky, ring, thumb;
    # reorder it to the DexForce actuator order above.
    hand_qpos_mapping: tuple[int, ...] = (
        16,
        17,
        18,
        19,
        20,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        12,
        13,
        14,
        15,
        8,
        9,
        10,
        11,
    )
    torso_joint_names: tuple[str, ...] = ("ANKLE", "KNEE", "BUTTOCK", "WAIST")
    head_joint_names: tuple[str, ...] = ("NECK1", "NECK2")
    torso_home_qpos: tuple[float, ...] = ()
    head_home_qpos: tuple[float, ...] = ()

    # -- Real robot (ROS 2 /control/joint_position) ------------------------
    #
    # Startup pose taken from the vendor's deployed robot_config, measured on
    # hardware.  The simulation home above is a different, visualisation-only
    # pose and must not be used to command the real arms.
    real_arm_home_qpos: tuple[tuple[float, ...], tuple[float, ...]] = (
        # left, right
        (-0.3065, -0.7689, 0.5767, -1.5671, 0.4924, -0.1750, -0.0063),
        (0.3065, 0.7689, -0.5767, 1.5671, -0.4924, 0.1750, 0.0063),
    )
    real_torso_home_qpos: tuple[float, ...] = (-0.6, 1.2, -0.6, 0.0)
    real_head_home_qpos: tuple[float, ...] = (0.0, -0.5590)

    def real_home_qpos(self, side: str) -> np.ndarray:
        index = 0 if side == "left" else 1
        return np.array(self.real_arm_home_qpos[index], dtype=np.float64)

    def real_body_joint_names(self) -> tuple[str, ...]:
        """Torso and head joints, in the order the driver expects them."""
        return tuple(self.torso_joint_names) + tuple(self.head_joint_names)

    def real_body_home_qpos(self) -> np.ndarray:
        return np.array(
            tuple(self.real_torso_home_qpos) + tuple(self.real_head_home_qpos),
            dtype=np.float64,
        )

    def full_body_joint_names(self) -> tuple[str, ...]:
        """All 20 joints in the order /control/joint_position receives them."""
        return (
            *self.arm_joint_names("right"),
            *self.arm_joint_names("left"),
            *self.real_body_joint_names(),
        )

    def arm_joint_names(self, side: str) -> tuple[str, ...]:
        prefix = side.upper()
        return tuple(f"{prefix}_{s}" for s in self.arm_joint_suffixes)

    def hand_joint_names(self, side: str) -> tuple[str, ...]:
        prefix = side.upper()
        return tuple(f"{prefix}_{s}" for s in self.hand_joint_suffixes)

    def ee_body_name(self, side: str) -> str:
        return f"{side}_j7"

    def arm_home_qpos(self, side: str) -> np.ndarray:
        if side == "left":
            return np.array([0.0, -0.6, 0.0, -1.57, 0.0, -0.4, 0.0], dtype=np.float64)
        return np.array([0.0, 0.6, 0.0, 1.57, 0.0, 0.4, 0.0], dtype=np.float64)

    def hand_config_path(self, input_source: str, side: str) -> Path:
        source = input_source if input_source in {"quest3", "avp", "pico4"} else "quest3"
        return ANYDEX_ADAPTIVE_CONFIG_DIR / source / f"{source}_linker_l20.yaml"

    def apply_head_torso(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        set_joint_fn,
        torso_joints: dict,
        yaw: float,
        pitch: float,
        t: float,
    ) -> None:
        set_joint_fn(model, data, torso_joints, "ANKLE",  0.6 + t * 0.6)
        set_joint_fn(model, data, torso_joints, "KNEE",  -1.2 + t * (-0.8))
        set_joint_fn(model, data, torso_joints, "BUTTOCK", 0.6 - pitch * 0.5)
        set_joint_fn(model, data, torso_joints, "WAIST",  yaw)
        set_joint_fn(model, data, torso_joints, "NECK1",  0.85 * yaw)
        set_joint_fn(model, data, torso_joints, "NECK2",  0.75 * pitch)

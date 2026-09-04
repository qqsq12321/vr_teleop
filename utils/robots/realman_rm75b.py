"""RealMan RM75-B dual arms with the fixed Inspire RH56DFX hand pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XACRO = (
    ROOT_DIR
    / "assets"
    / "arm_body"
    / "realman_rm75b"
    / "source"
    / "urdf"
    / "dual_75B_arm_robot.urdf.xacro"
)
DEFAULT_SCENE = ROOT_DIR / "example" / "scene_config" / "scene_realman_rm75b_inspire.xml"
ANYDEX_ADAPTIVE_CONFIG_DIR = (
    ROOT_DIR / "third_party" / "AnyDexRetarget" / "example" / "config" / "adaptive"
)

# A symmetric, folded-forward pose expressed in the vendor RM75-B joint frames.
# It keeps the elbows outside the torso and points both Inspire hands forward.
LEFT_ARM_HOME_QPOS = (-0.47, 0.0, -0.50, -2.04, 0.62, 1.09, -0.59)
RIGHT_ARM_HOME_QPOS = (0.47, 0.0, 0.50, 2.04, -0.62, -1.09, 0.59)


@dataclass(frozen=True)
class RealManRM75BInspireSpec:
    """Fixed RealMan RM75-B + Inspire RH56DFX simulation specification."""

    xacro_path: Path = DEFAULT_XACRO
    scene_path: Path = DEFAULT_SCENE
    hand_type: str = "inspire_rh56dfx"
    retarget_hand_type: str = "inspire_hand"
    # The vendor model is wrapped in a +90 deg world-alignment rotation. Pico/VR
    # translation is already expressed in project world axes, so do not apply the
    # rotated vendor base frame a second time to position residuals.
    base_body_name: str | None = None
    arm_asset_dir: Path = ROOT_DIR / "assets" / "arm_body" / "realman_rm75b"
    hand_asset_dir: Path = ROOT_DIR / "assets" / "dex_hand" / "inspire_rh56dfx"
    arm_joint_count: int = 7
    torso_joint_names: tuple[str, ...] = ()
    head_joint_names: tuple[str, ...] = ("head_joint1", "head_joint2")
    torso_home_qpos: tuple[float, ...] = ()
    head_home_qpos: tuple[float, ...] = (0.0, 0.0)
    hand_joint_suffixes: tuple[str, ...] = (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "thumb_intermediate_joint",
        "thumb_distal_joint",
        "index_proximal_joint",
        "index_intermediate_joint",
        "middle_proximal_joint",
        "middle_intermediate_joint",
        "ring_proximal_joint",
        "ring_intermediate_joint",
        "pinky_proximal_joint",
        "pinky_intermediate_joint",
    )
    hand_driver_joint_suffixes: tuple[str, ...] = (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint",
    )
    # AnyDex inspire_hand output order is index, middle, pinky, ring, thumb.
    hand_qpos_mapping: tuple[int, ...] = (8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5)
    inspire_channel_indices: tuple[int, ...] = (4, 6, 2, 0, 9, 8)
    inspire_channel_max_rad: tuple[float, ...] = (1.47, 1.47, 1.47, 1.47, 0.6, 1.308)
    inspire_channel_invert: tuple[bool, ...] = (True, True, True, True, True, True)

    def arm_joint_names(self, side: str) -> tuple[str, ...]:
        prefix = "l" if side == "left" else "r"
        return tuple(f"{prefix}_joint{i}" for i in range(1, self.arm_joint_count + 1))

    def ee_body_name(self, side: str) -> str:
        return f"{side}_arm_end_effector_mount_link"

    def arm_home_qpos(self, side: str) -> np.ndarray:
        values = LEFT_ARM_HOME_QPOS if side == "left" else RIGHT_ARM_HOME_QPOS
        return np.asarray(values, dtype=np.float64)

    def hand_joint_names(self, side: str) -> tuple[str, ...]:
        return tuple(f"{side}_{suffix}" for suffix in self.hand_joint_suffixes)

    def hand_driver_joint_names(self, side: str) -> tuple[str, ...]:
        return tuple(f"{side}_{suffix}" for suffix in self.hand_driver_joint_suffixes)

    def hand_config_path(self, input_source: str, side: str) -> Path:
        source = input_source if input_source in {"quest3", "avp", "pico4"} else "quest3"
        return ANYDEX_ADAPTIVE_CONFIG_DIR / source / f"{source}_{self.retarget_hand_type}.yaml"

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
        del t
        set_joint_fn(
            model,
            data,
            torso_joints,
            "head_joint1",
            self.head_home_qpos[0] + 0.85 * yaw,
        )
        set_joint_fn(
            model,
            data,
            torso_joints,
            "head_joint2",
            self.head_home_qpos[1] + 0.75 * pitch,
        )

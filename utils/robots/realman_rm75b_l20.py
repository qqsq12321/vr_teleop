"""RealMan RM75-B dual arms with the Linker L20 dexterous hand pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT_DIR / "example" / "scene_config" / "scene_realman_rm75b_l20.xml"
ANYDEX_ADAPTIVE_CONFIG_DIR = (
    ROOT_DIR / "third_party" / "AnyDexRetarget" / "example" / "config" / "adaptive"
)

# Derived from realman_rm75b.py's home, with two deliberate changes.
#
# r_joint1 is unwrapped by -2*pi so it lands inside the model's +/-3.11 limit.
# Forward kinematics are bit-identical; staying in range keeps
# --control-mode ctrl from clamping the right arm to a different startup pose
# than --control-mode qpos produces.
#
# l_joint7 (wrist roll) is rotated to +75 deg so the left palm faces down like
# the right one. The Inspire spec's -88.6 deg leaves the left palm facing up --
# measurable there too, not something the L20 graft introduced; the scene's
# checked-in keyframe predates that home change and hides it. J7's axis passes
# through the TCP site, so this rotates the palm without moving the
# end-effector position at all.
LEFT_ARM_HOME_QPOS = (-0.81834998, -0.17645279, 0.00000000, -1.37451660, -0.87545715, -0.99700188, 1.30899694)
RIGHT_ARM_HOME_QPOS = (3.96542297 - 2.0 * np.pi, -0.17640043, 0.00000000, -1.44113582, -2.08996197, 0.92622878, -1.55081231)


@dataclass(frozen=True)
class RealManRM75BL20Spec:
    """Fixed RealMan RM75-B + Linker L20 simulation specification.

    Simulation only. The L20 pair has no real-robot backend here; the Inspire
    hardware path lives in utils/robots/realman_rm75b.py.
    """

    scene_path: Path = DEFAULT_SCENE
    hand_type: str = "linker_l20"
    retarget_hand_type: str = "linker_l20"
    # The vendor model is wrapped in a +90 deg world-alignment rotation. Pico/VR
    # translation is already expressed in project world axes, so do not apply the
    # rotated vendor base frame a second time to position residuals.
    base_body_name: str | None = None
    arm_asset_dir: Path = ROOT_DIR / "assets" / "arm_body" / "realman_rm75b"
    hand_asset_dir: Path = ROOT_DIR / "assets" / "dex_hand" / "linker_l20"
    arm_joint_count: int = 7
    torso_joint_names: tuple[str, ...] = ()
    head_joint_names: tuple[str, ...] = ("head_joint1", "head_joint2")
    torso_home_qpos: tuple[float, ...] = ()
    head_home_qpos: tuple[float, ...] = (0.0, 0.0)
    # Declaration order of the canonical linker_l20_{side}_mujoco.xml joints,
    # which matches DexForceSpec.hand_joint_suffixes.
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
    # AnyDex linker_l20 output order is index, middle, pinky, ring, thumb;
    # reorder it to the joint order above. Same table as DexForceSpec, since
    # both scenes embed the same canonical hand.
    hand_qpos_mapping: tuple[int, ...] = (
        16, 17, 18, 19, 20,
        0, 1, 2, 3,
        4, 5, 6, 7,
        12, 13, 14, 15,
        8, 9, 10, 11,
    )

    def arm_joint_names(self, side: str) -> tuple[str, ...]:
        prefix = "l" if side == "left" else "r"
        return tuple(f"{prefix}_joint{i}" for i in range(1, self.arm_joint_count + 1))

    def ee_body_name(self, side: str) -> str:
        return f"{side}_arm_end_effector_mount_link"

    def arm_home_qpos(self, side: str) -> np.ndarray:
        values = LEFT_ARM_HOME_QPOS if side == "left" else RIGHT_ARM_HOME_QPOS
        return np.asarray(values, dtype=np.float64)

    def hand_joint_names(self, side: str) -> tuple[str, ...]:
        prefix = side.upper()
        return tuple(f"{prefix}_{suffix}" for suffix in self.hand_joint_suffixes)

    def hand_config_path(self, input_source: str, side: str) -> Path:
        del side  # One shared config per input source, as for DexForce.
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

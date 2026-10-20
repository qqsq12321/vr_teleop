"""CVTE CARM D3 dual-arm with a parallel gripper per side."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT_DIR / "example" / "scene_config" / "scene_cvte_carm_d3.xml"

# Upper arms hang down, elbows bent 90 deg so forearms and grippers point
# forward (+X). Same values for both sides: joint1/joint4 share the +Y axis on
# each arm, and the wrist joints stay at mid-range for IK headroom.
ARM_HOME_QPOS = (0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CVTECArmD3Spec:
    """Fixed-base CVTE CARM D3 dual-arm + single-DOF parallel gripper."""

    scene_path: Path = DEFAULT_SCENE
    hand_type: str = "cvte_gripper"
    base_body_name: str = "base_link"
    arm_asset_dir: Path = ROOT_DIR / "assets" / "arm_body" / "cvte_carm_d3"
    arm_joint_count: int = 7
    torso_joint_names: tuple[str, ...] = ()
    head_joint_names: tuple[str, ...] = ()
    torso_home_qpos: tuple[float, ...] = ()
    head_home_qpos: tuple[float, ...] = ()
    hand_qpos_mapping: tuple[int, ...] | None = None
    # Factory controller address (manual §3.2); the PC must sit on 10.42.0.x.
    real_ip: str = "10.42.0.101"

    def arm_joint_names(self, side: str) -> tuple[str, ...]:
        suffix = "LEFT" if side == "left" else "RIGHT"
        return tuple(f"joint{i}-{suffix}" for i in range(1, self.arm_joint_count + 1))

    def hand_joint_names(self, side: str) -> tuple[str, ...]:
        # The single driven DOF; `gripper_left_joint-{SUFFIX}` mimics it via the
        # scene's equality constraint and is not separately controlled.
        suffix = "LEFT" if side == "left" else "RIGHT"
        return (f"gripper_right_joint-{suffix}",)

    def ee_body_name(self, side: str) -> str:
        suffix = "LEFT" if side == "left" else "RIGHT"
        return f"gripper_base-{suffix}"

    def flange_body_name(self, side: str) -> str:
        # Tool flange: fixed `joint8` at the link7 tip, the frame the controller
        # reports as its Cartesian pose.
        suffix = "LEFT" if side == "left" else "RIGHT"
        return f"link8-{suffix}"

    def arm_home_qpos(self, side: str) -> np.ndarray:
        del side
        return np.asarray(ARM_HOME_QPOS, dtype=np.float64)

    def hand_config_path(self, input_source: str, side: str) -> None:
        # The single-DOF gripper is not retargeted by AnyDexRetarget; it is
        # driven by TeleopArmController.update_hand_from_pinch instead.
        del input_source, side
        return None

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
        del model, data, set_joint_fn, torso_joints, yaw, pitch, t

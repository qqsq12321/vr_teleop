"""Single RealMan RM75-B arm bolted to the floor, with no end effector.

Built from the vendor's standalone single-arm package (``RM75-B.urdf``) rather
than by carving one arm out of the dual-arm xacro.  Besides the cleaner
``joint_1..joint_7`` naming, the standalone model places ``joint_7`` at the
manual's d7 = 144 mm, so ``link_7`` is the true flange face; the dual-arm
``*_link7`` sits part-way along the wrist and only reaches the mount frame via
an extra hand-mount offset.

Unlike every other spec here, this one declares which sides it supports.  The
dual-arm specs map any non-``left`` side onto the right arm, which silently
accepts a side that does not exist; a single-arm spec has to reject it instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT_DIR / "example" / "scene_config" / "scene_realman_rm75b_single.xml"

ARM_JOINT_COUNT = 7
# The lone arm is addressed as the right side; VR input and the real-robot
# adapters both default to "right" for single-arm work.
SIDE = "right"
FLANGE_BODY = "right_arm_end_effector_mount_link"
TCP_SITE = "right_tcp"
PEDESTAL_BODY = "rm75_single_base"


def joint_name(index: int) -> str:
    """Vendor joint name for a 1-based joint index."""
    return f"joint_{index}"


# A natural forward-reaching posture: shoulder raised, elbow above the wrist,
# forearm angling down and forward, flange 0.52 m ahead of the base at 0.28 m
# height with the tool axis pointing straight forward (+X) so VR wrist tracking
# starts aligned. Tuned interactively in the MuJoCo viewer.
#
# The elbow sitting above the flange is what makes this read as an arm rather
# than a slumped one; poses with the elbow below the wrist look wrong even when
# the flange lands in the same place.
#
# J1/J3/J5/J7 stay at zero, keeping the arm in the X-Z plane with no lateral
# offset. J2 and J6 are non-zero because the manual (6.2.5) lists J2 = 0
# together with J6 = 0 as a singularity, and the all-zero pose is itself
# singular (manipulability 0, so IK cannot start). The tightest joint sits at
# 54% of its travel, inside the manual's ranges (J1 +-178, J2 +-130, J3 +-178,
# J4 +-135, J5 +-178, J6 +-128, J7 +-360 deg).
#
# The scene builder re-checks each value against the model's jnt_range, which
# is the guard the dual-arm home values violate on r_joint1 (+227 deg).
SINGLE_ARM_HOME_QPOS = (0.0, 0.93, 0.0, 1.27, 0.0, -0.70, 0.0)


@dataclass(frozen=True)
class RealManRM75BSingleSpec:
    """Floor-mounted single RM75-B arm, flange only."""

    scene_path: Path = DEFAULT_SCENE
    # No dexterous hand or gripper: teleoperation drives the arm to the flange
    # and nothing downstream of it.
    hand_type: str = "none"
    # The pedestal carries no rotation, so the base frame already matches the
    # project world frame (X forward, Y left, Z up) and there is no base
    # rotation to re-apply to position residuals. Revisit if the scene is ever
    # rebuilt with an alignment quaternion on the pedestal.
    base_body_name: str | None = None
    arm_asset_dir: Path = ROOT_DIR / "assets" / "arm_body" / "realman_rm75b_single"
    arm_joint_count: int = ARM_JOINT_COUNT
    sides: tuple[str, ...] = (SIDE,)
    torso_joint_names: tuple[str, ...] = ()
    head_joint_names: tuple[str, ...] = ()
    torso_home_qpos: tuple[float, ...] = ()
    head_home_qpos: tuple[float, ...] = ()
    hand_qpos_mapping: tuple[int, ...] | None = None

    def _check_side(self, side: str) -> None:
        if side not in self.sides:
            raise ValueError(
                f"RealMan RM75-B single-arm supports {self.sides}, got {side!r}"
            )

    def arm_joint_names(self, side: str) -> tuple[str, ...]:
        self._check_side(side)
        return tuple(joint_name(i) for i in range(1, self.arm_joint_count + 1))

    def ee_body_name(self, side: str) -> str:
        self._check_side(side)
        return FLANGE_BODY

    def arm_home_qpos(self, side: str) -> np.ndarray:
        self._check_side(side)
        return np.asarray(SINGLE_ARM_HOME_QPOS, dtype=np.float64)

    def hand_joint_names(self, side: str) -> tuple[str, ...]:
        self._check_side(side)
        return ()

    def hand_config_path(self, input_source: str, side: str) -> None:
        # Nothing to retarget without an end effector.
        del input_source
        self._check_side(side)
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

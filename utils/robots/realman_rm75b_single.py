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

from utils.robots.realman_rm75b import RealManRM75BInspireSpec

if TYPE_CHECKING:
    import mujoco

ROOT_DIR = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT_DIR / "example" / "scene_config"
DEFAULT_SCENE = SCENE_DIR / "scene_realman_rm75b_single.xml"
INSPIRE_SCENE = SCENE_DIR / "scene_realman_rm75b_single_inspire.xml"
ROHAND_SCENE = SCENE_DIR / "scene_realman_rm75b_single_rohand.xml"
ANYDEX_ADAPTIVE_CONFIG_DIR = (
    ROOT_DIR / "third_party" / "AnyDexRetarget" / "example" / "config" / "adaptive"
)

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

# Hand bodies, joints and actuators share the arm's side prefix so a hand's own
# names (ROHand calls its root mesh "base_link") cannot collide with the arm's.
HAND_PREFIX = f"{SIDE}_"

# ROHand joint names in the order its MJCF declares them, which is the order
# the scene keeps them in. The retargeter emits URDF order instead.
ROHAND_JOINT_NAMES = (
    "if_slider_link", "if_slider_abpart_link", "if_proximal_link", "if_distal_link", "if_connecting_link",
    "mf_slider_link", "mf_slider_abpart_link", "mf_proximal_link", "mf_distal_link", "mf_connecting_link",
    "rf_slider_link", "rf_slider_abpart_link", "rf_proximal_link", "rf_distal_link", "rf_connecting_link",
    "lf_slider_link", "lf_slider_abpart_link", "lf_proximal_link", "lf_distal_link", "lf_connecting_link",
    "th_root_link", "th_proximal_link", "th_slider_link", "th_connecting_link", "th_distal_link",
)
# Retargeter (URDF order) -> scene (MJCF order). Same permutation AnyDex's own
# MuJoCo output uses; tests re-derive it from joint names rather than trust it.
ROHAND_QPOS_MAPPING = (
    3, 4, 1, 2, 0, 13, 14, 11, 12, 10, 18, 19, 16, 17, 15,
    8, 9, 6, 7, 5, 20, 21, 23, 24, 22,
)
# The six motors, in ROH_FINGER_POS_TARGET0..5 register order (thumb bend,
# index, middle, ring, little, thumb rotation), each read from one retarget
# joint by name and normalised between its open and closed angle.
#
# The URDF gives the proximal joints +-pi, which is no limit at all, so the
# ranges come from replaying AnyDex's AVP recording (example/data/avp1.pkl)
# through both this hand and the Inspire hand. Each ROHand finger tracks the
# matching Inspire finger at corr -0.95..-0.99: the ROHand joint grows MORE
# NEGATIVE as the finger closes. The wide span (~3 rad) is because this joint is
# a link in the finger's four-bar linkage, not the anatomical knuckle. Values
# are the Inspire-open and Inspire-closed medians, rounded. Thumb rotation
# correlates weakly, so it uses the URDF's own 0..1.57.
#
# These are first estimates from a recording, not hardware measurements.
ROHAND_CHANNELS = (
    ("th_proximal_link", 1.0, -1.3),
    ("if_proximal_link", 0.6, -2.6),
    ("mf_proximal_link", 0.1, -2.5),
    ("rf_proximal_link", -0.1, -2.6),
    ("lf_proximal_link", -0.4, -2.6),
    ("th_root_link", 0.0, 1.57),
)


@dataclass(frozen=True)
class RealManRM75BSingleSpec:
    """Floor-mounted single RM75-B arm, flange only."""

    scene_path: Path = DEFAULT_SCENE
    # No dexterous hand or gripper: teleoperation drives the arm to the flange
    # and nothing downstream of it. Hand subclasses override the hand fields.
    hand_type: str = "none"
    retarget_hand_type: str | None = None
    hand_joint_suffixes: tuple[str, ...] = ()
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
        return tuple(f"{HAND_PREFIX}{suffix}" for suffix in self.hand_joint_suffixes)

    def hand_config_path(self, input_source: str, side: str) -> Path | None:
        self._check_side(side)
        if self.retarget_hand_type is None:
            return None  # Nothing to retarget without an end effector.
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
        del model, data, set_joint_fn, torso_joints, yaw, pitch, t


# The dual-arm spec already carries the Inspire joint list, retarget-output
# reordering and hardware channel mapping, all verified on that robot. Read
# them from there rather than keeping a second copy that could drift.
_DUAL_INSPIRE = RealManRM75BInspireSpec()


@dataclass(frozen=True)
class RealManRM75BSingleInspireSpec(RealManRM75BSingleSpec):
    """Single RM75-B with an Inspire RH56DFX right hand on the flange."""

    scene_path: Path = INSPIRE_SCENE
    hand_type: str = _DUAL_INSPIRE.hand_type
    retarget_hand_type: str | None = _DUAL_INSPIRE.retarget_hand_type
    hand_joint_suffixes: tuple[str, ...] = _DUAL_INSPIRE.hand_joint_suffixes
    hand_qpos_mapping: tuple[int, ...] | None = _DUAL_INSPIRE.hand_qpos_mapping
    # Read by utils.adapters.output.inspire_mapping.raw_qpos_to_channels.
    inspire_channel_indices: tuple[int, ...] = _DUAL_INSPIRE.inspire_channel_indices
    inspire_channel_max_rad: tuple[float, ...] = _DUAL_INSPIRE.inspire_channel_max_rad
    inspire_channel_invert: tuple[bool, ...] = _DUAL_INSPIRE.inspire_channel_invert


@dataclass(frozen=True)
class RealManRM75BSingleROHandSpec(RealManRM75BSingleSpec):
    """Single RM75-B with an OYMotion ROHand right hand on the flange."""

    scene_path: Path = ROHAND_SCENE
    hand_type: str = "rohand"
    retarget_hand_type: str | None = "rohand"
    hand_joint_suffixes: tuple[str, ...] = ROHAND_JOINT_NAMES
    hand_qpos_mapping: tuple[int, ...] | None = ROHAND_QPOS_MAPPING
    rohand_channels: tuple[tuple[str, float, float], ...] = ROHAND_CHANNELS


HAND_SPECS = {
    "none": RealManRM75BSingleSpec,
    "inspire": RealManRM75BSingleInspireSpec,
    "rohand": RealManRM75BSingleROHandSpec,
}

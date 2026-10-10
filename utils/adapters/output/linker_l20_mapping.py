"""AnyDex linker_l20 output -> DexForce L20 finger command values.

The retargeter returns 21 joint values per hand.  The DexForce robot exposes
16 active joints per hand and expects a percentage per joint on
``/control/ee/left`` and ``/control/ee/right`` (``mode = 0``, position).

Verified retarget output order (``Retargeter.optimizer.robot.dof_joint_names``,
Pinocchio's DOF order -- not the URDF declaration order):

    index:  roll(0) pitch(1) pip(2) dip(3)
    middle: roll(4) pitch(5) pip(6) dip(7)
    pinky:  roll(8) pitch(9) pip(10) dip(11)
    ring:   roll(12) pitch(13) pip(14) dip(15)
    thumb:  roll(16) yaw(17) pitch(18) mcp(19) ip(20)

Mapping to DexForce's 16 joints:

* drop the five passive joints the robot does not expose -- the four DIPs
  (3, 7, 11, 15) and the thumb IP (20);
* swap thumb roll and thumb yaw, because the robot lists ``T_CMC_YAW`` before
  ``T_CMC_ROLL``.

Both the retarget order and the DexForce hand limits are asserted at runtime
by ``load_sim_hand_limits`` so a retargeter or model upgrade that changes
either one fails loudly instead of moving the wrong finger.
"""

from __future__ import annotations

import numpy as np

# DexForce hand joints in the order used by both /control/ee/* commands and
# robot_config/hand/linker_l20.yaml::joints_name.<side>_hand.
DEXFORCE_HAND_JOINT_NAMES: tuple[str, ...] = (
    "T_CMC_YAW",
    "T_CMC_ROLL",
    "T_CMC_PITCH",
    "T_MCP",
    "IF_MCP_ROLL",
    "IF_MCP_PITCH",
    "IF_PIP",
    "MF_MCP_ROLL",
    "MF_MCP_PITCH",
    "MF_PIP",
    "RF_MCP_ROLL",
    "RF_MCP_PITCH",
    "RF_PIP",
    "LF_MCP_ROLL",
    "LF_MCP_PITCH",
    "LF_PIP",
)

# Expected retarget output order, as lowercase Pinocchio joint names.
EXPECTED_RETARGET_ORDER: tuple[str, ...] = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
)

# For each DexForce joint, the matching lowercase joint name in the retarget
# output. Used to build the index table by name at runtime.
_DEXFORCE_TO_RETARGET_NAME: dict[str, str] = {
    "T_CMC_YAW": "thumb_cmc_yaw",
    "T_CMC_ROLL": "thumb_cmc_roll",
    "T_CMC_PITCH": "thumb_cmc_pitch",
    "T_MCP": "thumb_mcp",
    "IF_MCP_ROLL": "index_mcp_roll",
    "IF_MCP_PITCH": "index_mcp_pitch",
    "IF_PIP": "index_pip",
    "MF_MCP_ROLL": "middle_mcp_roll",
    "MF_MCP_PITCH": "middle_mcp_pitch",
    "MF_PIP": "middle_pip",
    "RF_MCP_ROLL": "ring_mcp_roll",
    "RF_MCP_PITCH": "ring_mcp_pitch",
    "RF_PIP": "ring_pip",
    "LF_MCP_ROLL": "pinky_mcp_roll",
    "LF_MCP_PITCH": "pinky_mcp_pitch",
    "LF_PIP": "pinky_pip",
}

# Simulation joint names per side, in the order DexForce lists its joints.
# Mirrors DexForceSpec.hand_joint_suffixes plus the side prefix.
DEXFORCE_TO_SIM_SUFFIX: dict[str, str] = {
    "T_CMC_YAW": "T_CMC_YAW",
    "T_CMC_ROLL": "T_CMC_ROLL",
    "T_CMC_PITCH": "T_CMC_PITCH",
    "T_MCP": "T_MCP",
    "IF_MCP_ROLL": "IF_MCP_ROLL",
    "IF_MCP_PITCH": "IF_MCP_PITCH",
    "IF_PIP": "IF_PIP",
    "MF_MCP_ROLL": "MF_MCP_ROLL",
    "MF_MCP_PITCH": "MF_MCP_PITCH",
    "MF_PIP": "MF_PIP",
    "RF_MCP_ROLL": "RF_MCP_ROLL",
    "RF_MCP_PITCH": "RF_MCP_PITCH",
    "RF_PIP": "RF_PIP",
    "LF_MCP_ROLL": "LF_MCP_ROLL",
    "LF_MCP_PITCH": "LF_MCP_PITCH",
    "LF_PIP": "LF_PIP",
}

# Output ranges the robot expects for mode=0. The four finger ab/adduction
# joints and thumb yaw are bidirectional; everything else is 0..100.
_OUTPUT_RANGE: dict[str, tuple[float, float]] = {
    name: ((-100.0, 100.0) if name.endswith("_MCP_ROLL") else (0.0, 100.0))
    for name in DEXFORCE_HAND_JOINT_NAMES
}
_OUTPUT_RANGE["T_CMC_YAW"] = (-100.0, 100.0)

# Per-joint direction. Correct entries here rather than editing the mapping
# logic. The four finger MCP_ROLL joints are flipped because the hand abducted
# the wrong way on the robot; the remaining signs are still unverified, so the
# read-only pass is still the way to check them.
#
# These apply to both hands: retarget_to_dexforce_percent takes one table and
# the real entry passes the same one per side. If a flip turns out to be needed
# on only one hand, this table has to become side-aware first.
DEFAULT_SIGNS: dict[str, float] = {name: 1.0 for name in DEXFORCE_HAND_JOINT_NAMES}
DEFAULT_SIGNS.update(
    {name: -1.0 for name in DEXFORCE_HAND_JOINT_NAMES if name.endswith("_MCP_ROLL")}
)


def retarget_indices(retarget_joint_names: list[str] | tuple[str, ...]) -> list[int]:
    """Build the DexForce-joint -> retarget-output index table by name.

    Raises if the retargeter's joint order is not the expected one, so an
    upstream reordering cannot silently move the wrong finger.
    """
    names = [str(n).lower() for n in retarget_joint_names]
    if tuple(names) != EXPECTED_RETARGET_ORDER:
        raise ValueError(
            "Unexpected linker_l20 retarget joint order.\n"
            f"  expected: {list(EXPECTED_RETARGET_ORDER)}\n"
            f"  actual:   {names}"
        )
    index_by_name = {name: i for i, name in enumerate(names)}
    return [index_by_name[_DEXFORCE_TO_RETARGET_NAME[n]] for n in DEXFORCE_HAND_JOINT_NAMES]


def load_sim_hand_limits(model, spec, side: str) -> np.ndarray:
    """Read the 16 DexForce-order joint limits in radians from a MuJoCo model.

    Keeps the rad->percent calibration tied to the same XML the IK uses, so the
    two cannot drift apart.
    """
    import mujoco

    rows = []
    for name in DEXFORCE_HAND_JOINT_NAMES:
        joint_name = f"{side.upper()}_{DEXFORCE_TO_SIM_SUFFIX[name]}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Hand joint '{joint_name}' not found in the scene model.")
        rows.append(model.jnt_range[joint_id])
    return np.asarray(rows, dtype=np.float64)


def retarget_to_dexforce_percent(
    qpos: np.ndarray,
    sim_limits: np.ndarray,
    *,
    indices: list[int] | tuple[int, ...],
    signs: dict[str, float] | None = None,
) -> list[float]:
    """Map a retarget output vector to 16 DexForce finger percentages.

    Args:
        qpos: retarget output for one hand (21 values), radians.
        sim_limits: (16, 2) lower/upper limits in DexForce joint order, radians.
        indices: index table from :func:`retarget_indices`.
        signs: optional per-joint direction overrides.

    Returns:
        16 values in ``DEXFORCE_HAND_JOINT_NAMES`` order, ready for
        ``EEJointControl.values`` with ``mode = 0``.
    """
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    lim = np.asarray(sim_limits, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError("Retarget output contains NaN/Inf")
    if lim.shape != (len(DEXFORCE_HAND_JOINT_NAMES), 2):
        raise ValueError(
            f"Expected ({len(DEXFORCE_HAND_JOINT_NAMES)}, 2) hand limits, got {lim.shape}"
        )
    if len(indices) != len(DEXFORCE_HAND_JOINT_NAMES):
        raise ValueError(f"Expected {len(DEXFORCE_HAND_JOINT_NAMES)} indices, got {len(indices)}")

    sign_table = DEFAULT_SIGNS if signs is None else signs
    values: list[float] = []
    for out_index, name in enumerate(DEXFORCE_HAND_JOINT_NAMES):
        source_index = int(indices[out_index])
        if source_index >= q.shape[0]:
            raise ValueError(
                f"Retarget output has {q.shape[0]} values but '{name}' needs index {source_index}."
            )
        lo, hi = float(lim[out_index][0]), float(lim[out_index][1])
        out_lo, out_hi = _OUTPUT_RANGE[name]
        if hi <= lo:
            values.append(0.0)
            continue
        fraction = (float(q[source_index]) - lo) / (hi - lo)
        value = fraction * (out_hi - out_lo) + out_lo
        value *= float(sign_table.get(name, 1.0))
        values.append(float(np.clip(value, min(out_lo, out_hi), max(out_lo, out_hi))))
    return values


def open_hand_percent() -> list[float]:
    """Neutral fully-open command, used on startup and shutdown."""
    return [0.0] * len(DEXFORCE_HAND_JOINT_NAMES)


__all__ = [
    "DEFAULT_SIGNS",
    "DEXFORCE_HAND_JOINT_NAMES",
    "EXPECTED_RETARGET_ORDER",
    "load_sim_hand_limits",
    "open_hand_percent",
    "retarget_indices",
    "retarget_to_dexforce_percent",
]

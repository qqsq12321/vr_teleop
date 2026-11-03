"""AnyDex raw ROHand joints -> the six ROHand motor positions.

Units are radians in, ROHand logical positions 0..65535 out, in the order of the
``ROH_FINGER_POS_TARGET0..5`` registers (1135..1140): thumb bend, index,
middle, ring, little, thumb rotation.

Joints are looked up by name in the retargeter's own output order, so the raw
URDF-ordered vector is required -- not the MuJoCo-reordered hand qpos.
"""
from __future__ import annotations

import numpy as np

ROHAND_POSITION_MAX = 65535
ROHAND_CHANNEL_COUNT = 6


def raw_qpos_to_positions(qpos, joint_names, spec, *, closed_is_max: bool = True) -> list[int]:
    """Normalise each motor's source joint between its open and closed angle.

    ``closed_is_max`` says which end of the logical range is a closed finger.
    The vendor protocol does not state it, so it is a parameter rather than a
    baked-in assumption; confirm it on the hand before trusting either value.
    """
    q = np.asarray(qpos, dtype=np.float64)
    names = list(joint_names)
    if q.ndim != 1 or q.shape[0] != len(names) or not np.all(np.isfinite(q)):
        raise ValueError(
            f"Expected a finite raw ROHand qpos matching {len(names)} joint names, got {q.shape}"
        )
    result = []
    for joint, open_rad, closed_rad in spec.rohand_channels:
        try:
            value = q[names.index(joint)]
        except ValueError:
            raise ValueError(f"Retarget output has no joint {joint!r}") from None
        fraction = float(np.clip((value - open_rad) / (closed_rad - open_rad), 0.0, 1.0))
        if not closed_is_max:
            fraction = 1.0 - fraction
        result.append(int(round(fraction * ROHAND_POSITION_MAX)))
    if len(result) != ROHAND_CHANNEL_COUNT:
        raise ValueError(f"ROHand needs {ROHAND_CHANNEL_COUNT} channels, spec gives {len(result)}")
    return result

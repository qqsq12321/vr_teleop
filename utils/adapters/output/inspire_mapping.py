"""AnyDex raw 12-joint ordering -> Inspire six hardware channels.

Do not pass MuJoCo's reordered hand qpos here. Units are radians in,
0..1000 out (0 closed, 1000 open with the current robot specs).
"""
from __future__ import annotations

import numpy as np


def raw_qpos_to_channels(qpos, spec, *, scale: int = 1000) -> list[int]:
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != (12,) or not np.all(np.isfinite(q)):
        raise ValueError(f"Expected finite raw AnyDex Inspire qpos (12,), got {q.shape}")
    result = []
    for index, maximum, invert in zip(
        spec.inspire_channel_indices, spec.inspire_channel_max_rad,
        spec.inspire_channel_invert, strict=True,
    ):
        fraction = float(np.clip(q[index] / maximum, 0, 1))
        result.append(int((1 - fraction if invert else fraction) * scale))
    return result

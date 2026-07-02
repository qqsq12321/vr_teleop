"""Small MuJoCo output helpers for simulation backends."""

from __future__ import annotations

import numpy as np


def write_qpos(data, qpos_indices: np.ndarray, values: np.ndarray) -> None:
    """Write joint positions into a MuJoCo data.qpos vector."""
    if len(qpos_indices) != len(values):
        return
    for qadr, value in zip(qpos_indices, values):
        data.qpos[qadr] = value

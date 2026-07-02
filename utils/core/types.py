"""Shared data types passed between input, core, and output layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    position: np.ndarray
    quaternion: np.ndarray


@dataclass
class HandInput:
    wrist: Pose | None = None
    landmarks: np.ndarray | None = None


@dataclass
class TeleopFrame:
    timestamp: float
    left: HandInput
    right: HandInput
    head_pose: Pose | None = None
    stop_requested: bool = False


@dataclass
class SideCommand:
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None


@dataclass
class TeleopCommand:
    left: SideCommand
    right: SideCommand
    body_qpos: dict[str, float] | None = None
    stop_requested: bool = False

"""Strict Pico observations for RealMan only; simulation keeps cached input.

No cached side can become a fresh hardware observation. Local receive time is
not the device capture time and cannot establish end-to-end latency.
"""
from dataclasses import dataclass
import time
from typing import Optional

import numpy as np

from utils.adapters.input.pico4 import (
    Pico4, JOINT_21_INDICES, _parse_tracking_json, transform_pico4_to_robot_pose,
)


def _parse_csv_floats(s):
    values = []
    for part in s.split(','):
        if part.strip():
            try:
                values.append(float(part))
            except ValueError:
                values.append(float('nan'))
    return values


def _parse_hand_state(
    hand_j: dict,
    wrist_joint_index: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(hand_j, dict):
        return None, None, None
    is_active = hand_j.get('isActive', 0)
    try:
        is_active = int(float(str(is_active)))
    except (ValueError, TypeError, OverflowError):
        return None, None, None
    if is_active != 1:
        return None, None, None

    joints = hand_j.get('HandJointLocations')
    if not isinstance(joints, list) or len(joints) != 26:
        return None, None, None

    all_joints = np.full((26, 7), np.nan, dtype=np.float32)
    for i, joint in enumerate(joints):
        if not isinstance(joint, dict):
            continue
        p = joint.get('p')
        if not isinstance(p, str):
            continue
        vals = _parse_csv_floats(p)
        if len(vals) >= 7:
            all_joints[i] = vals[:7]
        elif len(vals) >= 3 and i != wrist_joint_index:
            # Finger orientation is not used by retargeting. A wrist without
            # orientation, however, must not become a synthetic identity pose.
            all_joints[i] = [*vals[:3], 0.0, 0.0, 0.0, 1.0]

    # Missing/invalid joints are not a new usable tracking observation.
    if not np.all(np.isfinite(all_joints)):
        return None, None, None
    if np.any(np.linalg.norm(all_joints[:, 3:7], axis=1) < 1e-6):
        return None, None, None

    result = np.zeros((21, 3), dtype=np.float32)
    for dst_i, src_i in enumerate(JOINT_21_INDICES):
        result[dst_i] = all_joints[src_i, :3]

    result -= result[0].copy()
    wrist_index = int(np.clip(wrist_joint_index, 0, all_joints.shape[0] - 1))
    wrist_pose = all_joints[wrist_index].copy()
    return result, wrist_pose, all_joints



@dataclass(frozen=True)
class PicoFrame:
    """Atomic latest observation; timestamps are local monotonic receive times.

    No inference is made about device clock or end-to-end network latency.
    """
    sequence: int
    received_at: float
    hands: dict[str, Optional[np.ndarray]]
    wrists: dict[str, Optional[np.ndarray]]

    def wrist_pose(self, side: str):
        pose = self.wrists[side]
        if pose is None or pose.shape != (7,) or not np.all(np.isfinite(pose)):
            return None
        if np.linalg.norm(pose[3:]) < 1e-6:
            return None
        return transform_pico4_to_robot_pose(pose[:3], pose[3:])


class TrackedPico4(Pico4):
    def __init__(self, *args, **kwargs):
        self._sequence = 0  # Initialized before the parent starts its receiver.
        super().__init__(*args, **kwargs)

    def _parse_tracking(self, payload):
        return _parse_tracking_json(payload, wrist_joint_index=self._wrist_joint_index,
                                    hand_parser=_parse_hand_state)

    def _publish_tracking(self, left, right, left_wrist, right_wrist,
                          left_joints, right_joints, head_pose) -> None:
        with self._lock:
            # Explicitly clear inactive sides: a right-hand frame must never
            # refresh or replay a cached left wrist.
            self._left_hand, self._right_hand = left, right
            self._left_wrist_pose, self._right_wrist_pose = left_wrist, right_wrist
            self._left_joints, self._right_joints = left_joints, right_joints
            self._head_pose = head_pose
            self._last_update = time.monotonic()
            self._sequence += 1

    def latest_frame(self, after_sequence: int = 0) -> Optional[PicoFrame]:
        with self._lock:
            if self._sequence <= after_sequence:
                return None
            def clone(value):
                return None if value is None else value.copy()
            return PicoFrame(
                self._sequence, self._last_update,
                {"left": clone(self._left_hand), "right": clone(self._right_hand)},
                {"left": clone(self._left_wrist_pose), "right": clone(self._right_wrist_pose)},
            )


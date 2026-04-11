"""Apple Vision Pro input via avp_stream (Tracking Streamer app).

Provides wrist pose, hand landmarks, and pinch distance in formats
compatible with the existing Quest 3 teleoperation pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from util.quaternion import matrix_to_quaternion

# ---------------------------------------------------------------------------
# VisionPro 25-joint → MediaPipe 21-landmark mapping
# Skips metacarpal indices (5, 10, 15, 20) not present in MediaPipe.
# Source: third_party/AnyDexRetarget/example/input/visionpro.py
# ---------------------------------------------------------------------------

VP_TO_MEDIAPIPE = (
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24
)


# ---------------------------------------------------------------------------
# Coordinate transform: AVP Z-up → Robot frame
# ---------------------------------------------------------------------------
#
# avp_stream applies YUP2ZUP internally, so output is Z-up RH:
#   AVP frame: X=right, Y=forward, Z=up
# Robot frame (same as transform_vr_to_robot_pose output):
#   X=forward, Y=left, Z=up
#
# Mapping: robot = Rz(-90°) @ avp
#   Robot X =  AVP Y   (forward)
#   Robot Y = -AVP X   (left)
#   Robot Z =  AVP Z   (up)

_RZ = np.array(
    [[ 0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0]],
    dtype=np.float64,
)

_RZ_T = _RZ.T


def transform_avp_to_robot_pose(
    avp_pos: np.ndarray,
    avp_rot: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Convert AVP Z-up wrist pose to robot frame.

    Args:
        avp_pos: (3,) position in AVP Z-up frame.
        avp_rot: (3, 3) rotation matrix in AVP Z-up frame.

    Returns:
        (robot_position, robot_quaternion) as tuples, matching the output
        format of ``quaternion.transform_vr_to_robot_pose()``.
    """
    robot_pos = _RZ @ avp_pos
    robot_rot = _RZ @ avp_rot @ _RZ_T
    robot_quat = matrix_to_quaternion(
        tuple(tuple(float(v) for v in row) for row in robot_rot)
    )
    return (float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])), robot_quat


# Extra Rz(+90°) clockwise correction for real Kinova
_RZ_CW = np.array(
    [[ 0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_RZ_CW_T = _RZ_CW.T


def apply_rz90cw(
    pos: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Apply an additional Rz(+90°) clockwise rotation on top of existing pose."""
    p = np.array(pos, dtype=np.float64)
    rp = _RZ_CW @ p
    # Reconstruct rotation matrix from quaternion, rotate, convert back
    from util.quaternion import quaternion_to_matrix
    rot = np.array(quaternion_to_matrix(quat), dtype=np.float64)
    rr = _RZ_CW @ rot @ _RZ_CW_T
    rq = matrix_to_quaternion(
        tuple(tuple(float(v) for v in row) for row in rr)
    )
    return (float(rp[0]), float(rp[1]), float(rp[2])), rq


# ---------------------------------------------------------------------------
# AVPInput class
# ---------------------------------------------------------------------------


class AVPInput:
    """Wraps ``avp_stream.VisionProStreamer`` for the teleop pipeline."""

    def __init__(self, ip: str) -> None:
        from avp_stream import VisionProStreamer

        self._streamer = VisionProStreamer(ip=ip, record=False)
        self._latest = None
        self._dual_pinch_start = None

    # -- per-frame refresh --------------------------------------------------

    def poll(self) -> bool:
        """Fetch the latest tracking frame. Returns True if new data arrived."""
        data = self._streamer.get_latest()
        if data is None:
            return False
        self._latest = data
        return True

    # -- wrist pose ---------------------------------------------------------

    def get_wrist_pose(
        self, side: str = "right"
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """Extract wrist (position, quaternion) in robot frame.

        Returns None if no data is available.
        """
        if self._latest is None:
            return None
        hand = self._latest.right if side == "right" else self._latest.left
        if hand is None:
            return None
        wrist_mat = hand.wrist  # (4, 4) in AVP Z-up frame
        if wrist_mat is None:
            return None
        pos = np.asarray(wrist_mat[:3, 3], dtype=np.float64)
        rot = np.asarray(wrist_mat[:3, :3], dtype=np.float64)
        return transform_avp_to_robot_pose(pos, rot)

    def get_wrist_pose_raw(
        self, side: str = "right"
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """Extract wrist (position, quaternion) in AVP Z-up frame directly.

        No additional rotation applied. For real robot where the base frame
        aligns with AVP Z-up without the Rz(-90) correction.
        """
        if self._latest is None:
            return None
        hand = self._latest.right if side == "right" else self._latest.left
        if hand is None:
            return None
        wrist_mat = hand.wrist
        if wrist_mat is None:
            return None
        pos = wrist_mat[:3, 3]
        rot = wrist_mat[:3, :3]
        quat = matrix_to_quaternion(
            tuple(tuple(float(v) for v in row) for row in rot)
        )
        return (float(pos[0]), float(pos[1]), float(pos[2])), quat

    # -- hand landmarks -----------------------------------------------------

    def get_landmarks_mediapipe(
        self, side: str = "right"
    ) -> Optional[np.ndarray]:
        """Return (21, 3) MediaPipe-format landmarks in AVP Z-up frame.

        These can be fed directly to ``Retargeter.retarget()`` since the
        AVP and Quest3 retarget configs are identical.
        """
        if self._latest is None:
            return None
        hand = self._latest.right if side == "right" else self._latest.left
        if hand is None:
            return None
        # hand is HandData with shape (N, 4, 4), N >= 25
        if hand.shape[0] < 25:
            return None
        mediapipe = np.zeros((21, 3), dtype=np.float32)
        for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
            mediapipe[mp_idx] = hand[vp_idx][:3, 3]
        if np.allclose(mediapipe, 0):
            return None
        return mediapipe

    # -- pinch distance -----------------------------------------------------

    def get_pinch_distance(self, side: str = "right") -> Optional[float]:
        """Return thumb-index pinch distance in meters."""
        if self._latest is None:
            return None
        hand = self._latest.right if side == "right" else self._latest.left
        if hand is None:
            return None
        return float(hand.pinch_distance)

    # -- dual pinch stop gesture --------------------------------------------

    _DUAL_PINCH_THRESHOLD = 0.02  # meters
    _DUAL_PINCH_HOLD_S = 3.0     # seconds

    def check_dual_pinch_stop(self) -> bool:
        """Return True if both hands are pinching for 3+ seconds (stop gesture).

        Call once per loop iteration. Prints countdown warnings.
        """
        if self._latest is None:
            self._dual_pinch_start = None
            return False
        left = self._latest.left
        right = self._latest.right
        if left is None or right is None:
            self._dual_pinch_start = None
            return False

        both_pinching = (
            left.pinch_distance < self._DUAL_PINCH_THRESHOLD
            and right.pinch_distance < self._DUAL_PINCH_THRESHOLD
        )

        if not both_pinching:
            if self._dual_pinch_start is not None:
                self._dual_pinch_start = None
            return False

        import time
        now = time.time()
        if self._dual_pinch_start is None:
            self._dual_pinch_start = now
            print("Dual pinch detected — hold 3s to stop...")
            return False

        elapsed = now - self._dual_pinch_start
        # Print countdown every second
        remaining = self._DUAL_PINCH_HOLD_S - elapsed
        if remaining > 0 and int(remaining) < int(remaining + 0.02):
            pass  # avoid spam
        if elapsed >= self._DUAL_PINCH_HOLD_S:
            print("Dual pinch stop triggered!")
            return True
        return False

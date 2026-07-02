"""Wrist residual tracker: VR wrist pose -> robot target pose."""

from __future__ import annotations

import math

import numpy as np

from utils.core.quaternion import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
)


class WristTracker:
    """Track VR wrist pose residuals with EMA smoothing.

    Shared by teleop_sim and teleop_real.
    """

    def __init__(
        self,
        initial_site_pos: np.ndarray,
        initial_site_quat: np.ndarray,
        *,
        position_scale: float = 1.5,
        ema_alpha: float = 0.8,
        negate_rot_xy: bool = False,
        base_xmat: np.ndarray | None = None,
        position_deadband: float = 0.0,
        rotation_deadband_deg: float = 0.0,
    ):
        self._initial_site_pos = np.asarray(initial_site_pos, dtype=np.float64).copy()
        self._initial_site_quat = np.asarray(initial_site_quat, dtype=np.float64).copy()
        self._position_scale = position_scale
        self._ema_alpha = ema_alpha
        self._negate_rot_xy = negate_rot_xy
        self._base_xmat = base_xmat.copy() if base_xmat is not None else None
        self._position_deadband = position_deadband
        self._rotation_deadband_deg = rotation_deadband_deg

        self._initial_wrist_position: tuple | None = None
        self._initial_wrist_quaternion: tuple | None = None
        self._smoothed_residual: np.ndarray | None = None

        self.target_position: np.ndarray = self._initial_site_pos.copy()
        self.target_quaternion: np.ndarray = self._initial_site_quat.copy()
        self.residual: np.ndarray | None = None
        self.euler_residual: tuple | None = None

    @property
    def initialized(self) -> bool:
        return self._initial_wrist_position is not None

    def reset_baseline(
        self,
        site_pos: np.ndarray,
        site_quat: np.ndarray,
        *,
        base_xmat: np.ndarray | None = None,
    ) -> None:
        """Reset the robot-side baseline pose (used by real-robot sync).

        Keeps the VR-side initial reference intact so that residuals continue
        from the same locked VR pose. Only the robot target pose and (optionally)
        the base rotation are updated.
        """
        self._initial_site_pos = np.asarray(site_pos, dtype=np.float64).copy()
        self._initial_site_quat = np.asarray(site_quat, dtype=np.float64).copy()
        if base_xmat is not None:
            self._base_xmat = np.asarray(base_xmat, dtype=np.float64).copy()
        self._smoothed_residual = None
        self.target_position = self._initial_site_pos.copy()
        self.target_quaternion = self._initial_site_quat.copy()
        self.residual = None
        self.euler_residual = None

    def update(self, robot_position: tuple, robot_quaternion: tuple) -> None:
        """Update target pose from a new VR wrist observation."""
        if self._initial_wrist_position is None:
            self._initial_wrist_position = robot_position
            self._initial_wrist_quaternion = robot_quaternion
            return

        # Position residual
        residual = np.array(
            [
                robot_position[0] - self._initial_wrist_position[0],
                robot_position[1] - self._initial_wrist_position[1],
                robot_position[2] - self._initial_wrist_position[2],
            ],
            dtype=np.float64,
        )
        if self._base_xmat is not None:
            residual = self._base_xmat @ residual

        # EMA smoothing
        if self._smoothed_residual is None:
            self._smoothed_residual = residual
        else:
            self._smoothed_residual = (
                self._ema_alpha * residual
                + (1.0 - self._ema_alpha) * self._smoothed_residual
            )

        # Position deadband
        if self._position_deadband > 0.0:
            if np.linalg.norm(self._smoothed_residual) < self._position_deadband:
                self._smoothed_residual = np.zeros_like(self._smoothed_residual)

        self.target_position = (
            self._initial_site_pos + self._position_scale * self._smoothed_residual
        )

        # Rotation residual
        relative_quaternion = quaternion_multiply(
            robot_quaternion,
            quaternion_inverse(self._initial_wrist_quaternion),
        )
        if self._negate_rot_xy:
            relative_quaternion = (
                -relative_quaternion[0],
                -relative_quaternion[1],
                relative_quaternion[2],
                relative_quaternion[3],
            )

        # Rotation deadband
        if self._rotation_deadband_deg > 0.0:
            deadband_rad = math.radians(self._rotation_deadband_deg)
            x, y, z, w = relative_quaternion
            sin_half = math.sqrt(x * x + y * y + z * z)
            angle = 2.0 * math.atan2(sin_half, abs(w))
            if angle < deadband_rad:
                relative_quaternion = (0.0, 0.0, 0.0, 1.0)

        self.target_quaternion = np.array(
            quaternion_multiply(relative_quaternion, self._initial_site_quat),
            dtype=np.float64,
        )
        norm = np.linalg.norm(self.target_quaternion)
        if norm > 0.0:
            self.target_quaternion /= norm

        self.residual = self._smoothed_residual
        self.euler_residual = quaternion_to_euler_xyz(
            relative_quaternion[0],
            relative_quaternion[1],
            relative_quaternion[2],
            relative_quaternion[3],
        )

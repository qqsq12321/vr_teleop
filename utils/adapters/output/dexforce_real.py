"""DexForce real-robot output backend placeholder."""

from __future__ import annotations

import numpy as np


class DexForceRealArm:
    """Placeholder for a real DexForce arm + hand SDK connection.

    Replace these methods with calls to the DexForce SDK during real-robot
    adaptation. The contract mirrors the teleop loop: connect, read current
    joints, send arm targets, send hand targets, and disconnect cleanly.
    """

    def __init__(self, side: str, ip: str, num_arm_joints: int = 7, num_hand_joints: int = 21):
        self.side = side
        self.ip = ip
        self.num_arm_joints = num_arm_joints
        self.num_hand_joints = num_hand_joints
        self._connected = False

    def connect(self) -> None:
        print(f"[{self.side}] TODO: connect DexForce arm/hand at {self.ip}")
        self._connected = True

    def get_joint_positions_rad(self) -> np.ndarray:
        return np.zeros(self.num_arm_joints, dtype=np.float64)

    def send_arm_qpos(self, q_rad: np.ndarray) -> None:
        pass

    def send_hand_qpos(self, q_rad: np.ndarray) -> None:
        pass

    def close(self) -> None:
        self._connected = False

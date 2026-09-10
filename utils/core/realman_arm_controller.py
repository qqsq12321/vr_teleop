"""RealMan-only feedback validation and input re-alignment.

The simulation controller and WristTracker keep their original behavior.
"""
import mujoco
import numpy as np

from utils.core.arm_controller import TeleopArmController


class RealManArmController(TeleopArmController):
    def sync_from_joint_positions(self, current_q):
        current_q = np.asarray(current_q, dtype=float)
        if current_q.shape != (len(self.arm_dof_indices),) or not np.all(np.isfinite(current_q)):
            raise ValueError('Measured joint positions must match arm DoF and be finite')
        self.data.qpos[self.arm_dof_indices] = current_q
        mujoco.mj_forward(self.model, self.data)
        self.reset_baseline_from_current_pose()

    def reset_baseline_from_current_pose(self, *, reset_input_reference=False):
        super().reset_baseline_from_current_pose()
        if reset_input_reference:
            # Only the hardware session may discard its VR reference. Preserve
            # all tracker tuning, while requiring the next fresh frame to align.
            self.tracker._initial_wrist_position = None
            self.tracker._initial_wrist_quaternion = None

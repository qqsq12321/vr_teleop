"""RM_API2 adapter for a single floor-mounted RealMan RM75-B arm.

No SDK import or connection happens at module import; ``Robotic_Arm`` is only
imported inside ``connect``.  That keeps ``--check`` and the offline tests from
loading the vendor native library.

This is deliberately separate from ``realman_real.RealManOutput``: that adapter
drives the dual-arm torso robot through the vendored *legacy* SDK (an ``Arm``
class with CamelCase methods), while RM_API2 is a handle-based ``rm_*`` API.
The two are not interchangeable, so the legacy path is left untouched.

Like the legacy adapter, this is not a hard-real-time transport: a blocked
native call cannot be cancelled from Python.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from utils.adapters.output.realman_real import ArmFeedback, RealManError, _joints
from utils.core.safety import InputTrackingFault

LOG = logging.getLogger(__name__)

ARM_DOF = 7
DEFAULT_PORT = 8080
# rm_movej for the planned startup move, matching the legacy path's profile.
INITIAL_MOVE_SPEED = 20


def _check(code, operation):
    """RM_API2 returns 0 on success; negatives are transport/parse errors."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RealManError(f"{operation}: RM_API2 returned {code!r}")


class RM75BSingleOutput:
    """One RM75-B arm over RM_API2.

    ``rm_destroy`` tears down the SDK for the whole process, so only one
    adapter may own it at a time -- the same constraint the legacy adapter has.
    """

    _owner_lock = threading.Lock()

    def __init__(
        self,
        ip: str,
        *,
        port: int = DEFAULT_PORT,
        factory=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(ip, str) or not ip:
            raise ValueError("A single arm IP address is required")
        self.ip = ip
        self.port = int(port)
        self.clock = clock
        self._factory = factory
        self._arm = None
        self._lock = threading.RLock()
        self._owns_sdk = False
        self._motion_enabled = False
        self._fault = None

    # -- unit conversion ----------------------------------------------------

    def model_to_sdk_degrees(self, q_rad) -> np.ndarray:
        """Model radians to controller degrees.

        A floor-mounted arm's base frame matches the model's, so there is no
        joint offset to apply -- unlike the dual-arm torso, whose right arm
        needs a J1 correction for its mount.
        """
        return np.rad2deg(_joints(q_rad))

    def sdk_degrees_to_model(self, q_deg) -> np.ndarray:
        return np.deg2rad(_joints(q_deg))

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> ArmFeedback:
        with self._lock:
            if self._arm is not None:
                raise RealManError("Already connected")
            if not self._owner_lock.acquire(blocking=False):
                raise RealManError("Another RM_API2 adapter owns the SDK in this process")
            self._owns_sdk = True
            try:
                arm = self._build_arm()
                self._arm = arm
                handle = arm.rm_create_robot_arm(self.ip, self.port)
                # A failed connection reports a non-positive handle id rather
                # than raising, so check it before trusting anything else.
                handle_id = getattr(handle, "id", None)
                if handle_id is None or int(handle_id) <= 0:
                    raise RealManError(f"Could not connect to {self.ip}:{self.port} (handle={handle_id!r})")

                dof = getattr(arm, "arm_dof", None)
                if dof is None or int(dof) != ARM_DOF:
                    raise RealManError(f"Expected a {ARM_DOF}-DoF RM75 arm, controller reports arm_dof={dof!r}")

                code, info = arm.rm_get_robot_info()
                _check(code, "robot info")
                LOG.info("Connected to %s:%s; robot info: %s", self.ip, self.port, info)
                return self._read_feedback_locked()
            except Exception:
                self.close()
                raise

    def _build_arm(self):
        if self._factory is not None:
            return self._factory(self.ip)
        # Imported here so --check and the offline tests never load the vendor
        # native library.
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        return RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    def enable_motion(self):
        """Only call after the runner selected motion mode and checked feedback."""
        with self._lock:
            self._require_connected()
            if self._fault:
                raise RealManError(f"Latched fault: {self._fault}")
            self._motion_enabled = True

    def _require_connected(self):
        if self._arm is None:
            raise RealManError("Arm is not connected")

    def _require_motion(self, guard):
        self._require_connected()
        if not self._motion_enabled or self._fault:
            raise RealManError(f"Motion disabled or faulted: {self._fault}")
        # Called immediately before EACH SDK write, including after blocking I/O.
        try:
            guard(self.clock())
        except InputTrackingFault:
            # Skip this write; a later fresh frame may resume the same session.
            raise
        except Exception as exc:
            self._fault = str(exc)
            raise

    # -- feedback -----------------------------------------------------------

    def read_feedback(self) -> ArmFeedback:
        with self._lock:
            self._require_connected()
            return self._read_feedback_locked()

    def _read_feedback_locked(self) -> ArmFeedback:
        try:
            # Stamp the request start, not the reply: a slow response must not
            # be able to look fresh.
            sampled_at = self.clock()
            code, degrees = self._arm.rm_get_joint_degree()
            _check(code, "joint feedback")
            return ArmFeedback(self.sdk_degrees_to_model(degrees), sampled_at)
        except Exception as exc:
            self._fault = str(exc)
            raise

    # -- motion -------------------------------------------------------------

    def move_to_initial_pose(self, q_rad, *, guard: Callable[[float], None]):
        """Planned, blocking startup move. Separate from the CANFD teleop path.

        Waits for the controller to report completion so the caller can read the
        measured pose afterwards rather than assuming the target was reached.
        """
        degrees = self.model_to_sdk_degrees(q_rad)
        with self._lock:
            self._require_motion(guard)
            try:
                _check(
                    self._arm.rm_movej(degrees.tolist(), INITIAL_MOVE_SPEED, 0, 0, 1),
                    "initial rm_movej",
                )
            except Exception as exc:
                self._fault = str(exc)
                raise

    def send_joints(self, q_rad, *, guard: Callable[[float], None], follow: bool = False):
        degrees = self.model_to_sdk_degrees(q_rad)
        with self._lock:
            self._require_motion(guard)
            try:
                _check(
                    self._arm.rm_movej_canfd(degrees.tolist(), bool(follow)),
                    "rm_movej_canfd",
                )
            except Exception as exc:
                self._fault = str(exc)
                raise

    def stop(self):
        """Best-effort halt. This is NOT an emergency stop; use the physical one."""
        with self._lock:
            if self._arm is None or not self._motion_enabled:
                return
            self._fault = self._fault or "Stopped; restart to re-arm"
            try:
                _check(self._arm.rm_set_arm_stop(), "rm_set_arm_stop")
            except Exception:
                LOG.exception("Unable to confirm stop; use the physical emergency stop")
            self._motion_enabled = False

    def close(self):
        with self._lock:
            self.stop()
            arm, self._arm = self._arm, None
            if arm is not None:
                try:
                    arm.rm_delete_robot_arm()
                except Exception:
                    LOG.exception("Arm handle delete failed")
            try:
                if arm is not None:
                    # Global to the process; only after the handle is gone.
                    arm.rm_destroy()
            except Exception:
                LOG.exception("SDK destroy failed")
            finally:
                if self._owns_sdk:
                    self._owns_sdk = False
                    self._owner_lock.release()


__all__ = ["RM75BSingleOutput", "RealManError", "ArmFeedback"]

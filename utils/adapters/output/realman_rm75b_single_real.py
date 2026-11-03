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

# End tool port (末端工具口). The hand is wired here; the controller relays
# its RS485 traffic.
HAND_KINDS = ("none", "inspire")
TOOL_VOLTAGE_LABELS = {0: "0V", 1: "5V", 2: "12V", 3: "24V"}
# Fourth-generation rm_set_tool_rs485_mode: switch the port to the
# controller's built-in dexterous-hand protocol.
TOOL_RS485_HAND = 1

INSPIRE_CHANNEL_MAX = 1000
HAND_CHANNELS = 6


def _check(code, operation):
    """RM_API2 returns 0 on success; negatives are transport/parse errors."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RealManError(f"{operation}: RM_API2 returned {code!r}")


def _hand_channels(values, maximum):
    channels = [int(v) for v in values]
    if len(channels) != HAND_CHANNELS or any(
        int(v) != v or not 0 <= v <= maximum for v in values
    ):
        raise ValueError(f"Expected {HAND_CHANNELS} integers in 0..{maximum}, got {list(values)}")
    return channels


class RM75BSingleOutput:
    """One RM75-B arm over RM_API2, plus an optional hand on its tool port.

    ``rm_destroy`` tears down the SDK for the whole process, so only one
    adapter may own it at a time -- the same constraint the legacy adapter has.

    Arm motion and hand commands are gated separately: ``--hand-only`` drives
    the hand without ever enabling the arm.
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
        self.controller_version = None
        self._hand = None
        self._hand_enabled = False

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
                # Tool-port APIs differ between controller generations.
                version = (info or {}).get("robot_controller_version")
                self.controller_version = int(version) if version is not None else None
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

    # -- hand on the end tool port -------------------------------------------

    def tool_voltage(self) -> str:
        """Read the tool port supply. Never sets it: a wrong value can damage a hand."""
        with self._lock:
            self._require_connected()
            code, voltage = self._arm.rm_get_tool_voltage()
            _check(code, "rm_get_tool_voltage")
            return TOOL_VOLTAGE_LABELS.get(int(voltage), f"code {voltage}")

    def configure_hand(self, kind: str, *, baudrate: int = 115200):
        """Put the tool port into the mode this hand speaks. Sends no hand target.

        Refuses when the tool port is unpowered rather than switching power on:
        the supply must be set on the teach pendant for the hand actually fitted.
        """
        if kind not in HAND_KINDS:
            raise ValueError(f"Unknown hand {kind!r}; expected one of {HAND_KINDS}")
        with self._lock:
            self._require_connected()
            if kind == "none":
                self._hand = None
                return
            voltage = self.tool_voltage()
            if voltage == "0V":
                raise RealManError(
                    "Tool port outputs 0V. Set the hand's supply voltage on the teach "
                    "pendant first; this program never changes it."
                )
            generation = self.controller_version
            if generation not in (3, 4):
                raise RealManError(
                    f"Unknown controller generation {generation!r}; cannot pick the tool-port API"
                )
            # The controller speaks the Inspire protocol natively; the fourth
            # generation needs the port switched to hand mode first.
            if generation == 4:
                try:
                    _check(self._arm.rm_set_tool_rs485_mode(TOOL_RS485_HAND, int(baudrate)),
                           "rm_set_tool_rs485_mode(hand)")
                except Exception as exc:
                    self._fault = str(exc)
                    raise
            self._hand = {"kind": kind, "generation": generation}
            LOG.info("Tool port %s, controller gen %s, configured for %s", voltage, generation, kind)

    def enable_hand(self):
        """Allow hand commands. Independent of arm motion."""
        with self._lock:
            self._require_connected()
            if self._hand is None:
                raise RealManError("No hand configured")
            if self._fault:
                raise RealManError(f"Latched fault: {self._fault}")
            self._hand_enabled = True

    def send_hand(self, channels, *, guard: Callable[[float], None]):
        """Send one Inspire target: six channels, 0..1000."""
        with self._lock:
            self._require_connected()
            if self._hand is None or not self._hand_enabled or self._fault:
                raise RealManError(f"Hand disabled or faulted: {self._fault}")
            values = _hand_channels(channels, INSPIRE_CHANNEL_MAX)
            try:
                guard(self.clock())
            except InputTrackingFault:
                raise
            except Exception as exc:
                self._fault = str(exc)
                raise
            try:
                _check(self._arm.rm_set_hand_angle(values, False, 0), "rm_set_hand_angle")
            except Exception as exc:
                # A hand that stops answering fails closed like the arm does.
                self._fault = str(exc)
                raise

    def stop(self):
        """Best-effort halt. This is NOT an emergency stop; use the physical one."""
        with self._lock:
            # The hand keeps its last target: opening it on stop could drop
            # whatever it is holding.
            self._hand_enabled = False
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
            self._hand = None
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

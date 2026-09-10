"""Legacy RealMan Python SDK adapter. No SDK import/connect at module import.

One adapter owns both sockets and the process-global legacy API. Calls are
serialized: this SDK mutates shared ctypes function signatures. This is not a
hard-real-time transport; a blocked native call cannot be cancelled by Python.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable

import numpy as np

from utils.core.safety import InputTrackingFault

LOG = logging.getLogger(__name__)
SIDES = ("left", "right")


class RealManError(RuntimeError):
    pass


def _check(result, operation):
    if not isinstance(result, (int, np.integer)) or result != 0:
        raise RealManError(f"{operation}: SDK returned {result!r}")


def _joints(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise RealManError("Expected seven finite joint values")
    return q


@dataclass(frozen=True)
class ArmFeedback:
    q_rad: np.ndarray
    # Request start, not response completion: slow replies cannot look fresh.
    sampled_at: float


class RealManOutput:
    _owner_lock = threading.Lock()

    def __init__(self, left_ip: str, right_ip: str, *, factory=None,
                 clock: Callable[[], float] = time.monotonic,
                 right_j1_offset_deg: float = 0.0):
        if left_ip == right_ip:
            raise ValueError("Left/right arm IPs must differ")
        self.right_j1_offset_deg = float(right_j1_offset_deg)
        if not np.isfinite(self.right_j1_offset_deg):
            raise ValueError("Right J1 offset must be finite")
        self.ips = dict(zip(SIDES, (left_ip, right_ip)))
        self.clock = clock
        self._factory = factory
        self._arms = {}
        self._lock = threading.RLock()
        self._owns_api = False
        self._motion_enabled = False
        self._fault = None
        self._hand_config = None

    def model_to_sdk_degrees(self, side, q_rad):
        """Fixed calibration, never wrap angles at +/-180 (would jump targets)."""
        if side not in SIDES:
            raise ValueError(f"Unknown arm side: {side}")
        q = np.rad2deg(_joints(q_rad))
        if side == "right":
            q[0] += self.right_j1_offset_deg
        return q

    def sdk_degrees_to_model(self, side, q_deg):
        if side not in SIDES:
            raise ValueError(f"Unknown arm side: {side}")
        q = _joints(q_deg).copy()
        if side == "right":
            q[0] -= self.right_j1_offset_deg
        return np.deg2rad(q)

    def connect(self):
        with self._lock:
            if self._owns_api:
                raise RealManError("Already connected")
            if not self._owner_lock.acquire(blocking=False):
                raise RealManError("Only one RealManOutput may own the legacy API")
            self._owns_api = True
            try:
                factory = self._factory
                if factory is None:
                    from third_party.realman_api.robotic_arm import Arm, RM75

                    def factory(ip):
                        # Track a partly initialized object so constructor errors
                        # do not silently leak its native socket.
                        arm = Arm.__new__(Arm)
                        try:
                            Arm.__init__(arm, RM75, ip)
                        except BaseException:
                            if getattr(arm, "nSocket", -1) > 0:
                                arm.Arm_Socket_Close()
                            if not self._arms:
                                arm.RM_API_UnInit()
                            raise
                        return arm

                for side in SIDES:
                    arm = factory(self.ips[side])
                    self._arms[side] = arm
                    _check(arm.Arm_Socket_State(), f"{side} connection")
                    if getattr(arm, "code", None) != 7:
                        raise RealManError(f"{side}: SDK did not identify a 7-DOF arm")
                    tag, info = arm.Get_Arm_Software_Info()
                    _check(tag, f"{side} product information")
                    product = info.product_version.decode(errors="replace")
                    if "RM75" not in product:
                        raise RealManError(f"{side}: expected RM75, got {product!r}")
                    LOG.info("%s %s: %s", side, self.ips[side], product)
                return self.read_feedback()
            except BaseException:
                self.close()
                raise

    def read_feedback(self) -> dict[str, ArmFeedback]:
        with self._lock:
            self._require_connected()
            result = {}
            try:
                for side, arm in self._arms.items():
                    sampled_at = self.clock()
                    tag, joints, _pose, errors = arm.Get_Current_Arm_State(retry=0)
                    _check(tag, f"{side} feedback")
                    if any(int(err, 0) if isinstance(err, str) else int(err) for err in errors):
                        raise RealManError(f"{side} controller fault: {errors}")
                    result[side] = ArmFeedback(self.sdk_degrees_to_model(side, joints), sampled_at)
            except Exception as exc:
                self._fault = str(exc)
                raise
            return result

    def enable_motion(self):
        """Only call when the runner selected hardware motion mode and checked feedback."""
        with self._lock:
            self._require_connected()
            if self._fault:
                raise RealManError(f"Latched fault: {self._fault}")
            self._motion_enabled = True

    def _require_connected(self):
        if set(self._arms) != set(SIDES):
            raise RealManError("Both arm connections are required, even for single-arm control")

    def _require_motion(self, side, guard):
        self._require_connected()
        if side not in SIDES or not self._motion_enabled or self._fault:
            raise RealManError(f"Motion disabled or faulted: {side}, {self._fault}")
        # Called immediately before EACH SDK write, including after blocking I/O.
        try:
            guard(self.clock())
        except InputTrackingFault:
            # Skip this write; a later fresh frame may resume the same session.
            raise
        except Exception as exc:
            self._fault = str(exc)
            raise

    def move_to_initial_pose(self, side: str, q_rad, *, guard: Callable[[float], None]):
        """Planned startup move, matching master's MoveJ v=20, r=0, block=1.

        This is separate from CANFD teleop; wait for the SDK to report completion
        before reading the measured pose and establishing the wrist baseline.
        """
        q = _joints(q_rad)
        with self._lock:
            self._require_motion(side, guard)
            try:
                _check(self._arms[side].Movej_Cmd(
                    self.model_to_sdk_degrees(side, q).tolist(), v=20, r=0, trajectory_connect=0, block=True),
                    f"{side} initial Movej_Cmd")
            except Exception as exc:
                self._fault = str(exc)
                raise

    def send_joints(self, side: str, q_rad, *, guard: Callable[[float], None], follow=False):
        q = _joints(q_rad)
        with self._lock:
            self._require_motion(side, guard)
            try:
                _check(self._arms[side].Movej_CANFD(self.model_to_sdk_degrees(side, q).tolist(), bool(follow)),
                       f"{side} Movej_CANFD")
            except Exception as exc:
                self._fault = str(exc)
                raise

    def configure_hands(self, config: dict, *, guard):
        """Explicit backend selection; never probe hands by writing registers.

        Tool power/voltage are deliberately NOT changed by this adapter.
        """
        backend = config["backend"]
        if backend not in {"native", "modbus"}:
            raise ValueError("Hand backend must be native or modbus")
        with self._lock:
            if backend == "modbus":
                for side in config["sides"]:
                    self._require_motion(side, guard)
                    _check(self._arms[side].Set_Modbus_Mode(
                        1, int(config["baudrate"]), int(config["timeout_100ms"]), True),
                        f"{side} Set_Modbus_Mode")
            self._hand_config = dict(config)

    def configure_gripper_speed(self, side: str, speed: int, *, guard):
        """Set gripper speed by sending an initial open position command.

        RealMan grippers don't have a standalone speed-setting API; speed is
        embedded in Pick/Release commands. We initialize with a Release to set
        speed without closing the gripper.
        """
        if side not in SIDES:
            raise ValueError(f"Unknown arm side: {side}")
        if not 1 <= speed <= 1000:
            raise ValueError("Gripper speed must be in 1..1000")
        with self._lock:
            self._require_motion(side, guard)
            try:
                _check(self._arms[side].Set_Gripper_Release(speed, block=False, timeout=0),
                       f"{side} gripper speed config")
            except Exception as exc:
                self._fault = str(exc)
                raise

    def send_gripper_position(self, side: str, position, *, guard):
        """Set the single-axis RealMan gripper opening, 1..1000."""
        if side not in SIDES:
            raise ValueError(f"Unknown arm side: {side}")
        position = int(position)
        if not 1 <= position <= 1000:
            raise ValueError("Gripper position must be in 1..1000")
        with self._lock:
            self._require_motion(side, guard)
            try:
                _check(self._arms[side].Set_Gripper_Position(
                    position, block=False, timeout=0),
                    f"{side} gripper command")
            except Exception as exc:
                self._fault = str(exc)
                raise

    def send_hand(self, side: str, channels, *, guard):
        values = np.asarray(channels)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("Hand command must contain six finite channels")
        if np.any(values < 0) or np.any(values > 1000) or np.any(values != np.floor(values)):
            raise ValueError("Hand channels must be integers in 0..1000")
        with self._lock:
            config = self._hand_config
            if config is None or side not in config["sides"]:
                raise RealManError("Hand output not explicitly configured for this side")
            self._require_motion(side, guard)
            try:
                angles = values.astype(int).tolist()
                if config["backend"] == "native":
                    tag = self._arms[side].Set_Hand_Angle(angles, block=False)
                else:
                    # Legacy Modbus Write_Registers expects big-endian bytes,
                    # unlike the separate Inspire direct-serial packet encoder.
                    data = [byte for angle in angles for byte in (angle >> 8, angle & 255)]
                    tag = self._arms[side].Write_Registers(
                        1, int(config["address"]), 6, data, int(config["device_id"]), False)
                _check(tag, f"{side} hand command")
            except Exception as exc:
                self._fault = str(exc)
                raise

    def stop(self):
        """Best effort arm stop; NOT power-off, NOT a hardware emergency stop.

        A hand may continue toward its last angle; do not assume stopping the
        arm cancels an Inspire command. No automatic opening/releasing objects.
        """
        with self._lock:
            if not self._motion_enabled:
                return
            self._fault = self._fault or "Stopped; restart to re-arm"
            for side, arm in self._arms.items():
                try:
                    _check(arm.Move_Stop_Cmd(block=True), f"{side} stop")
                except Exception:
                    LOG.exception("Unable to confirm stop for %s; use physical emergency stop", side)
            self._motion_enabled = False

    def close(self):
        with self._lock:
            self.stop()
            arms = list(self._arms.values())
            self._arms.clear()
            for arm in arms:
                try:
                    arm.Arm_Socket_Close()
                except Exception:
                    LOG.exception("Socket close failed")
            try:
                if arms:
                    arms[-1].RM_API_UnInit()  # Global; only after both sockets close.
            finally:
                if self._owns_api:
                    self._owns_api = False
                    self._owner_lock.release()

"""CVTE CARM D3 dual-arm adapter over the vendor ``carm`` Python SDK.

No SDK import or network access happens at module import. The SDK talks to the
controller over WebSocket (default ``10.42.0.101:8090``) and reconnects in the
background on its own, so freshness is judged from the local receive time of
each pushed joint state, never from "is connected".

Joint values are passed through unchanged: the vendor's ``joints_bind.json``
binds URDF ``jointN-LEFT/RIGHT`` 1:1 to the controller's ``arm[0/1]`` joint
``N-1``, and the manual's joint limits equal the URDF limits.

The D3 joints have no brakes. Servo-off or a hardware E-stop drops the arms,
so this adapter only ever stops the tracking task; it never disables servos.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

LOG = logging.getLogger(__name__)
SIDES = ("left", "right")
DOF = 7
# SDK error code that only reports a warning (carm_dual_bot_demo.on_carm_error).
WARNING_CODE = 3001


class CVTEError(RuntimeError):
    pass


class CVTEFeedbackError(CVTEError):
    pass


def _joints(value, what="joint vector") -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (DOF,) or not np.all(np.isfinite(q)):
        raise CVTEError(f"Expected {DOF} finite values for {what}, got {value!r}")
    return q


def sim_gripper_to_opening(q_sim: float, sim_range, eeff_range) -> float:
    """Map the sim finger joint (0 = closed) to the controller's opening in metres."""
    sim_lo, sim_hi = (float(v) for v in sim_range)
    eeff_lo, eeff_hi = (float(v) for v in eeff_range)
    ratio = np.clip((float(q_sim) - sim_lo) / (sim_hi - sim_lo), 0.0, 1.0)
    return float(eeff_lo + ratio * (eeff_hi - eeff_lo))


class CVTERealOutput:
    def __init__(
        self,
        ip: str,
        *,
        sides=SIDES,
        read_only: bool = False,
        factory: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
        connect_timeout_s: float = 5.0,
        feedback_timeout_s: float = 0.5,
        joint_margin_rad: float = 0.01,
    ):
        selected = tuple(dict.fromkeys(str(side).lower() for side in sides))
        if not selected or any(side not in SIDES for side in selected):
            raise ValueError(f"CVTE sides must be left/right, got {selected}")
        for name, value in (
            ("connect_timeout_s", connect_timeout_s),
            ("feedback_timeout_s", feedback_timeout_s),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(joint_margin_rad) or joint_margin_rad < 0:
            raise ValueError("joint_margin_rad must be finite and non-negative")
        self.ip = ip
        self.sides = selected
        self.read_only = bool(read_only)
        self.clock = clock
        self.connect_timeout_s = float(connect_timeout_s)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.joint_margin_rad = float(joint_margin_rad)
        self._factory = factory
        self._bot = None
        self._lock = threading.RLock()
        self._feedback: dict[str, tuple[np.ndarray, float]] = {}
        self._limits: dict[str, np.ndarray] = {}
        self._controller_limits: dict[str, np.ndarray] = {}
        self._eeff_config: dict[str, dict] = {}
        self._last_command: dict[str, tuple[np.ndarray, float]] = {}
        self._motion_enabled = False
        self._fault: str | None = None
        self.commands_sent = 0

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            if self._bot is not None:
                raise CVTEError("Already connected")
            factory = self._factory
            if factory is None:
                from carm import CArmDualBot

                factory = CArmDualBot
            self._bot = factory(self.ip)
            deadline = self.clock() + self.connect_timeout_s
            while not self._bot.is_connected():
                if self.clock() > deadline:
                    raise CVTEError(
                        f"No connection to the D3 controller at {self.ip} within "
                        f"{self.connect_timeout_s:.1f}s (check cable, 10.42.0.x address, web UI)"
                    )
                time.sleep(0.05)
            self._bot.register_error_cbk("vr_teleop", self._on_error)
            for side in SIDES:
                getattr(self._bot, f"register_{side}_joint_cbk")(
                    lambda t, p, v, tau, side=side: self._on_joint(side, p)
                )
                config = getattr(self._bot, f"get_{side}_config")() or {}
                lower, upper = config.get("limit_lower"), config.get("limit_upper")
                if lower is not None and upper is not None:
                    self._controller_limits[side] = np.column_stack(
                        [_joints(lower, f"{side} limit_lower"), _joints(upper, f"{side} limit_upper")]
                    )
                self._eeff_config[side] = getattr(self._bot, f"get_{side}_eeff_config")() or {}

    def wait_for_feedback(self, timeout_s: float = 2.0) -> bool:
        deadline = self.clock() + timeout_s
        while self.clock() <= deadline:
            if all(self._fresh(side) for side in SIDES):
                return True
            time.sleep(0.02)
        return False

    def _on_joint(self, side: str, positions) -> None:
        try:
            q = _joints(positions, f"{side} feedback")
        except CVTEError:
            return  # Stale entry stays; freshness checks catch a silent feed.
        with self._lock:
            self._feedback[side] = (q, self.clock())

    def _on_error(self, code, message) -> None:
        if code == WARNING_CODE:
            LOG.warning("D3 controller warning: %s", message)
            return
        with self._lock:
            self._fault = self._fault or f"D3 controller error {code}: {message}"
        LOG.error("D3 controller error %s: %s", code, message)

    # -- state --------------------------------------------------------------

    def _fresh(self, side: str) -> bool:
        sample = self._feedback.get(side)
        return sample is not None and 0.0 <= self.clock() - sample[1] <= self.feedback_timeout_s

    @property
    def fault(self) -> str | None:
        return self._fault

    def feedback_age_s(self, side: str) -> float:
        sample = self._feedback.get(side)
        return float("inf") if sample is None else self.clock() - sample[1]

    def read_arm_positions(self, side: str) -> np.ndarray:
        with self._lock:
            sample = self._feedback.get(side)
            if sample is None:
                raise CVTEFeedbackError(f"{side}: no joint feedback received yet")
            if not self._fresh(side):
                raise CVTEFeedbackError(
                    f"{side}: joint feedback stale for {self.feedback_age_s(side):.2f}s "
                    f"(limit {self.feedback_timeout_s:.2f}s)"
                )
            return sample[0].copy()

    def status(self, side: str) -> dict:
        return dict(getattr(self._bot, f"get_{side}_status")() or {})

    def read_cart_pose(self, side: str) -> np.ndarray | None:
        pose = getattr(self._bot, f"get_{side}_cart_pose")()
        pose = np.asarray(pose if pose is not None else [], dtype=np.float64)
        return pose if pose.shape == (7,) and np.all(np.isfinite(pose)) else None

    def check_startup_state(self) -> list[str]:
        problems = []
        if self._bot is None or not self._bot.is_connected():
            return [f"not connected to {self.ip}"]
        if self._fault:
            problems.append(self._fault)
        for side in SIDES:
            status = self.status(side)
            if status.get("arm_dof", DOF) != DOF:
                problems.append(f"{side}: controller reports {status.get('arm_dof')} DOF, expected {DOF}")
            if status.get("state", 0) == -1:
                problems.append(f"{side}: controller is in an error state (clear it in the web UI, 复位)")
            if not self._fresh(side):
                problems.append(f"{side}: no fresh joint feedback")
        return problems

    def gripper_available(self, side: str) -> bool:
        if self._bot is None:
            return False
        try:
            eeff_type = getattr(self._bot, f"get_{side}_eeff_type")()
            connected = getattr(self._bot, f"get_{side}_eeff_connect")()
            dof = getattr(self._bot, f"get_{side}_eeff_dof")()
        except Exception:  # Optional feature: missing state just means "no gripper".
            return False
        return bool(connected) and eeff_type == "gripper" and dof == 1

    def gripper_range(self, side: str) -> tuple[float, float]:
        config = self._eeff_config.get(side, {})
        lower = (config.get("eeff_lower") or [0.0])[0]
        upper = (config.get("eeff_upper") or [0.077])[0]
        lower, upper = float(lower), float(upper)
        if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
            raise CVTEError(f"{side}: invalid gripper range {config!r}")
        return lower, upper

    # -- limits -------------------------------------------------------------

    def load_limits_from_model(self, model, joint_names_by_side: dict[str, tuple[str, ...]]) -> None:
        """Command limits = model range ∩ controller soft limits, shrunk by the margin."""
        for side in SIDES:
            ranges = np.array([model.jnt_range[model.joint(name).id] for name in joint_names_by_side[side]])
            if side in self._controller_limits:
                controller = self._controller_limits[side]
                ranges = np.column_stack(
                    [np.maximum(ranges[:, 0], controller[:, 0]), np.minimum(ranges[:, 1], controller[:, 1])]
                )
            ranges = ranges + np.array([self.joint_margin_rad, -self.joint_margin_rad])
            if np.any(ranges[:, 0] >= ranges[:, 1]):
                raise CVTEError(f"{side}: joint margin consumes an entire joint range")
            self._limits[side] = ranges

    def limits(self, side: str) -> np.ndarray:
        return self._limits[side].copy()

    # -- motion -------------------------------------------------------------

    def _require_motion(self, side: str) -> None:
        if self.read_only:
            raise CVTEError("Read-only session: motion commands are disabled")
        if self._bot is None or not self._motion_enabled:
            raise CVTEError("Motion not enabled")
        if side not in self.sides:
            raise CVTEError(f"{side} arm is not selected for this session")
        if self._fault:
            raise CVTEError(f"Latched fault: {self._fault}")
        if side not in self._limits:
            raise CVTEError("Joint limits not loaded")

    def enable_motion(self) -> None:
        """Clear errors, enable servos, enter POSITION mode (SDK ``set_ready``)."""
        with self._lock:
            if self.read_only:
                raise CVTEError("Read-only session: motion commands are disabled")
            if self._bot is None:
                raise CVTEError("Not connected")
            if self._fault:
                raise CVTEError(f"Latched fault: {self._fault}")
            if self._bot.set_ready() != 1:
                raise CVTEError("set_ready failed (servo enable / POSITION mode)")
            for side in self.sides:
                status = self.status(side)
                if status.get("servo") != 1 or status.get("fsm_state") != "POSITION":
                    raise CVTEError(
                        f"{side}: not ready after set_ready (servo={status.get('servo')}, "
                        f"fsm_state={status.get('fsm_state')!r}, state={status.get('state')})"
                    )
            self._motion_enabled = True

    def send_joints(self, side: str, q_rad, *, gripper_m: float | None = None) -> np.ndarray:
        """Stream a joint target (clamped to limits); returns what was sent."""
        q = _joints(q_rad, f"{side} target")
        with self._lock:
            self._require_motion(side)
            limits = self._limits[side]
            q = np.clip(q, limits[:, 0], limits[:, 1])
            eeff = -1.0
            if gripper_m is not None:
                lower, upper = self.gripper_range(side)
                eeff = float(np.clip(gripper_m, lower, upper))
            result = getattr(self._bot, f"track_{side}_joint")(q.tolist(), eeff)
            if result != 1:
                self._fault = self._fault or f"{side}: track_{side}_joint returned {result}"
                raise CVTEError(self._fault)
            self._last_command[side] = (q.copy(), self.clock())
            self.commands_sent += 1
            return q

    def last_command(self, side: str) -> np.ndarray | None:
        with self._lock:
            sample = self._last_command.get(side)
            return None if sample is None else sample[0].copy()

    def check_tracking(self, side: str) -> float:
        """Return tracking error without latching faults (monitoring only)."""
        with self._lock:
            if side not in self._last_command:
                return 0.0
            measured = self.read_arm_positions(side)
            error = float(np.max(np.abs(measured - self._last_command[side][0])))
            return error

    def stop(self) -> None:
        """Stop the tracking task; servos stay enabled so the arms hold (no brakes)."""
        with self._lock:
            if self._bot is None or not self._motion_enabled:
                return
            self._motion_enabled = False
            try:
                if self._bot.task_stop() != 1:
                    # The controller also declines this when no task is running.
                    LOG.warning("task_stop not acknowledged (no running task?); arms keep their last target")
            except Exception:
                LOG.exception("task_stop failed; use the web UI or the E-stop if needed")

    def close(self) -> None:
        with self._lock:
            self.stop()
            bot, self._bot = self._bot, None
            if bot is None:
                return
            # Callbacks are not released first: the SDK's receive thread iterates
            # its callback dict unlocked, so releasing raises inside it.
            try:
                bot.disconnect()
            except Exception:
                LOG.exception("Disconnect failed")

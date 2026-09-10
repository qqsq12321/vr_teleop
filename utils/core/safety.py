"""Fail-closed host-side checks, not a certified robot safety controller."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np


class SafetyFault(RuntimeError):
    pass


class InputTrackingFault(SafetyFault):
    """Missing, explicitly invalid, or expired tracking (still latched)."""

@dataclass(frozen=True)
class SafetyLimits:
    rate_hz: float = 50.0
    input_timeout_s: float = 0.25
    feedback_timeout_s: float = 0.2
    command_timeout_s: float = 0.08
    joint_margin_rad: float = 0.0872664626
    tracking_error_rad: float = 0.1745329252
    tracking_error_grace_s: float = 0.2
    collision_margin_m: float = 0.02

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")


def finite_vector(value, length=7):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise SafetyFault(f"Expected finite vector ({length},)")
    return vector


class SafetySupervisor:
    """Per-side freshness + persistent tracking fault + target validation.

    No automatic recovery. Create a new session after a fault. Joint limits must come from the matching hardware/URDF joint frames.
    """
    def __init__(self, limits: SafetyLimits, joint_ranges, sides, *, hands=False, recover_tracking=False):
        self.limits = limits
        self.sides = tuple(sides)
        if not self.sides or not set(self.sides) <= {"left", "right"}:
            raise ValueError("Invalid controlled arm selection")
        self.hands = hands
        self.recover_tracking = recover_tracking
        self.ranges = {}
        for side in ("left", "right"):
            r = np.asarray(joint_ranges[side], dtype=float)
            if r.shape != (7, 2) or not np.all(np.isfinite(r)):
                raise ValueError("Expected finite joint ranges (7,2)")
            r = r + np.array([limits.joint_margin_rad, -limits.joint_margin_rad])
            if np.any(r[:, 0] >= r[:, 1]):
                raise ValueError("Joint margins consume entire range")
            self.ranges[side] = r
        self.wrist_times = {}
        self.hand_times = {}
        self.feedback = {}
        self.q_command = {}
        self.error_since = {}
        self.fault = None

    def fail(self, reason, fault_type=SafetyFault):
        self.fault = self.fault or reason
        raise fault_type(self.fault)

    def observe_input(self, result):
        self.wrist_times.update(result.wrist_received_at)
        self.hand_times.update(result.hand_received_at)
        for side in result.invalid_wrist_sides:
            self.wrist_times.pop(side, None)
        for side in result.invalid_hand_sides:
            self.hand_times.pop(side, None)

    def _fresh(self, timestamp, now, timeout):
        return timestamp is not None and math.isfinite(timestamp) and 0 <= now - timestamp <= timeout

    def input_ready(self, now):
        return all(self._fresh(self.wrist_times.get(s), now, self.limits.input_timeout_s)
                   and (not self.hands or self._fresh(self.hand_times.get(s), now, self.limits.input_timeout_s))
                   for s in self.sides)

    def check_bounds(self, side, q):
        q = finite_vector(q)
        r = self.ranges[side]
        if np.any(q < r[:, 0]) or np.any(q > r[:, 1]):
            self.fail(f"{side}: joint soft limit exceeded")
        return q

    def observe_feedback(self, feedback, now):
        if set(feedback) != {"left", "right"}:
            self.fail("Both arm feedback samples are required")
        self.feedback = feedback
        for side, state in feedback.items():
            self.check_bounds(side, state.q_rad)
            if not self._fresh(state.sampled_at, now, self.limits.feedback_timeout_s):
                self.fail(f"{side}: feedback timeout")
            if side in self.q_command:
                error = np.max(np.abs(state.q_rad - self.q_command[side]))
                if error > self.limits.tracking_error_rad:
                    self.error_since.setdefault(side, now)
                    if now - self.error_since[side] >= self.limits.tracking_error_grace_s:
                        self.fail(f"{side}: persistent joint tracking error")
                else:
                    self.error_since.pop(side, None)

    def start(self, now):
        self.check_live(now)
        self.q_command = {s: self.feedback[s].q_rad.copy() for s in self.sides}

    def input_problem(self, now):
        for side in self.sides:
            for kind, times in (("wrist", self.wrist_times), ("hand", self.hand_times)):
                if kind == "hand" and not self.hands:
                    continue
                stamp = times.get(side)
                if self._fresh(stamp, now, self.limits.input_timeout_s):
                    continue
                if stamp is None:
                    detail = "input marked invalid or no valid sample"
                elif not math.isfinite(stamp) or not math.isfinite(now) or now < stamp:
                    detail = "invalid receive timestamp"
                else:
                    detail = (f"no fresh sample for {(now - stamp) * 1000:.1f} ms "
                              f"(limit {self.limits.input_timeout_s * 1000:.1f} ms)")
                return f"{side}: {kind} tracking lost/stale: {detail}"
        return None

    def check_live(self, now, *, generated_at=None):
        if self.fault:
            raise SafetyFault(self.fault)
        if not math.isfinite(now):
            self.fail("Invalid monotonic time")
        problem = self.input_problem(now)
        if problem:
            if self.recover_tracking:
                raise InputTrackingFault(problem)
            self.fail(problem, InputTrackingFault)
        for side in ("left", "right"):
            state = self.feedback.get(side)
            if state is None or not self._fresh(state.sampled_at, now, self.limits.feedback_timeout_s):
                self.fail(f"{side}: feedback timeout")
        if generated_at is not None and not self._fresh(generated_at, now, self.limits.command_timeout_s):
            self.fail("Command expired during computation or SDK I/O")

    def validate_targets(self, targets, now):
        """Check IK targets without changing their joint angles or ramping them."""
        self.check_live(now)
        if not self.q_command or set(targets) != set(self.sides):
            self.fail("Session not started or incomplete command")
        commands = {side: self.check_bounds(side, targets[side]).copy()
                    for side in self.sides}
        self.q_command = commands
        return {side: q.copy() for side, q in commands.items()}

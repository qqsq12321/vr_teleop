"""Galbot G1 real-arm output backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from utils.robots.galbot import GalbotInspireSpec


class GalbotRealRobotOutput:
    """Dispatch arm joint commands to a Galbot G1 through the official SDK.

    The Galbot SDK import is intentionally delayed until ``connect()`` so the
    rest of the project can be imported on machines without the robot runtime.
    """

    def __init__(
        self,
        spec: GalbotInspireSpec,
        *,
        skip_controller_switch: bool = False,
        strict_joint_names: bool = True,
        command_time_from_start_s: float = 0.0,
    ):
        self.spec = spec
        self.skip_controller_switch = bool(skip_controller_switch)
        self.strict_joint_names = bool(strict_joint_names)
        self.command_time_from_start_s = float(command_time_from_start_s)

        self.robot: Any | None = None
        self._control_status: Any | None = None
        self._controller_name: Any | None = None
        self._joint_group: Any | None = None
        self._joint_command: Any | None = None
        self._joint_names = {
            "left": list(spec.arm_joint_names("left")),
            "right": list(spec.arm_joint_names("right")),
        }
        self._warned_command_failure: set[str] = set()

    def connect(self) -> None:
        try:
            from galbot_sdk.g1 import (
                ControlStatus,
                G1ControllerName,
                G1JointGroup,
                GalbotRobot,
                JointCommand,
            )
        except ImportError as exc:
            raise ImportError(
                "galbot_sdk is required for Galbot real teleop. "
                "Install the official SDK and source its setup.sh before running."
            ) from exc

        self._control_status = ControlStatus
        self._controller_name = G1ControllerName
        self._joint_group = G1JointGroup
        self._joint_command = JointCommand

        self.robot = GalbotRobot()
        if not self.robot.init(enable_sync_mode=False):
            raise RuntimeError("GalbotRobot.init() failed.")

        self._verify_joint_names("left")
        self._verify_joint_names("right")

        if not self.skip_controller_switch:
            self._switch_arm_controller("left")
            self._switch_arm_controller("right")

    def get_joint_positions_rad(self, side: str) -> np.ndarray:
        self._require_connected()
        names = self._names_for_side(side)
        values = self.robot.get_joint_positions(joint_names=names)
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] != len(names):
            raise RuntimeError(f"Galbot returned {arr.shape[0]} {side} joints, expected {len(names)}.")
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Galbot returned invalid {side} joint positions: {arr}")
        return arr

    def send_arm_qpos(self, side: str, q_rad: Sequence[float] | np.ndarray) -> None:
        self._require_connected()
        names = self._names_for_side(side)
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape[0] != len(names):
            raise ValueError(f"{side} arm command has {q.shape[0]} joints, expected {len(names)}.")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{side} arm command contains NaN/Inf: {q}")

        commands = []
        for value in q:
            command = self._joint_command()
            command.position = float(value)
            commands.append(command)

        status = self.robot.set_joint_commands(
            commands,
            joint_names=names,
            time_from_start_s=self.command_time_from_start_s,
        )
        if not self._status_ok(status) and side not in self._warned_command_failure:
            print(f"Warning: Galbot {side} arm command failed with status {status}")
            self._warned_command_failure.add(side)

    def close(self) -> None:
        if self.robot is None:
            return
        for method_name in ("request_shutdown", "wait_for_shutdown", "destroy"):
            method = getattr(self.robot, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception as exc:
                print(f"Warning: Galbot {method_name} failed: {exc}")
        self.robot = None

    def _switch_arm_controller(self, side: str) -> None:
        controller = (
            self._controller_name.LEFT_ARM_PVT_CTRL
            if side == "left"
            else self._controller_name.RIGHT_ARM_PVT_CTRL
        )
        status = self.robot.switch_controller(controller)
        self._require_status_ok(status, f"switch {side} arm controller to {controller}")

    def _verify_joint_names(self, side: str) -> None:
        if not self.strict_joint_names:
            return
        group = self._joint_group.left_arm if side == "left" else self._joint_group.right_arm
        active_names = list(self.robot.get_joint_names(only_active_joint=True, joint_groups=[group]))
        expected = self._names_for_side(side)
        if active_names != expected:
            raise RuntimeError(
                f"Galbot {side} active joint names mismatch.\n"
                f"  expected: {expected}\n"
                f"  actual:   {active_names}"
            )

    def _names_for_side(self, side: str) -> list[str]:
        if side not in self._joint_names:
            raise ValueError(f"Unknown Galbot side: {side!r}")
        return self._joint_names[side]

    def _require_connected(self) -> None:
        if self.robot is None:
            raise RuntimeError("Galbot robot is not connected.")

    def _require_status_ok(self, status: Any, operation: str) -> None:
        if not self._status_ok(status):
            raise RuntimeError(f"Galbot {operation} failed with status {status}.")

    def _status_ok(self, status: Any) -> bool:
        if self._control_status is None:
            return bool(status)
        for attr in ("SUCCESS", "OK"):
            if hasattr(self._control_status, attr) and status == getattr(self._control_status, attr):
                return True
        if isinstance(status, bool):
            return status
        return str(status).lower().endswith("success") or str(status).lower().endswith("ok")

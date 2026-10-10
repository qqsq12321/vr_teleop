"""DexForce W1 real-robot output backend over ROS 2.

DexForce exposes no Python SDK in this project: the arms, torso, head and hands
are driven by publishing to the vendor's driver topics.

    /control/joint_position       joint_interfaces/JointPositionControl
    /control/ee/left              end_effector_interfaces/EEJointControl
    /control/ee/right             end_effector_interfaces/EEJointControl

and state is read from

    /feedback/robot_server_state  std_msgs/String   (a JSON document)
    /feedback/ee/left|right       end_effector_interfaces/EEFeedback

This module owns one rclpy node and never imports the robot stack at module
scope, so the file can be imported (and unit tested) on a machine without ROS.

Safety behaviour implemented here, because the driver does neither:

* ``joint_margin_rad`` keeps commands inside the model's joint ranges;
* a stall watchdog stops publishing when teleop frames go stale;
* ``read_only=True`` never creates a publisher at all.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from utils.adapters.output.linker_l20_mapping import DEXFORCE_HAND_JOINT_NAMES
from utils.robots.dexforce import DexForceSpec

_CONTROL_TOPIC = "/control/joint_position"
_FEEDBACK_TOPIC = "/feedback/robot_server_state"
_HAND_CONTROL_TOPIC = "/control/ee/{}"
_HAND_FEEDBACK_TOPIC = "/feedback/ee/{}"

# EEJointControlMode.POSITION
_MODE_POSITION = 0

# run_mode strings that mean the driver will refuse or misbehave on commands.
_BLOCKED_RUN_MODES = {"Drag", "Impedance", "Sleep"}
_HEALTHY_STATUS = {"Running", "Idle"}


class DexForceFeedbackError(RuntimeError):
    """Raised when the robot server state cannot be used."""


def parse_server_state(payload: str) -> dict[str, Any]:
    """Parse the JSON document carried by /feedback/robot_server_state."""
    try:
        state = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DexForceFeedbackError(f"robot_server_state is not valid JSON: {exc}") from exc
    if not isinstance(state, dict):
        raise DexForceFeedbackError("robot_server_state JSON is not an object")

    names = state.get("joint_name")
    positions = state.get("joint_position")
    if not isinstance(names, list) or not isinstance(positions, list):
        raise DexForceFeedbackError("robot_server_state is missing joint_name/joint_position")
    if len(names) != len(positions) or not names:
        raise DexForceFeedbackError(
            f"robot_server_state length mismatch: {len(names)} names, {len(positions)} positions"
        )
    for value in positions:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DexForceFeedbackError(f"robot_server_state has a non-finite position: {value!r}")
    return state


def joint_positions_by_name(state: dict[str, Any]) -> dict[str, float]:
    """Return {joint name: position in radians} from a parsed server state."""
    return {str(n): float(p) for n, p in zip(state["joint_name"], state["joint_position"])}


class DexForceRealOutput:
    """Command DexForce arms/body and hands, and read back robot state."""

    def __init__(
        self,
        spec: DexForceSpec,
        *,
        read_only: bool = False,
        publish_hz: float = 50.0,
        joint_margin_rad: float = 0.05,
        watchdog_timeout_s: float = 0.25,
        node_name: str = "vr_teleop_dexforce",
    ):
        self.spec = spec
        self.read_only = bool(read_only)
        self.publish_hz = float(publish_hz)
        self.joint_margin_rad = float(joint_margin_rad)
        self.watchdog_timeout_s = float(watchdog_timeout_s)
        self.node_name = node_name

        self.joint_names: tuple[str, ...] = spec.full_body_joint_names()
        self.joint_index: dict[str, int] = {n: i for i, n in enumerate(self.joint_names)}
        self.hand_joint_names: tuple[str, ...] = DEXFORCE_HAND_JOINT_NAMES

        self.node: Any | None = None
        self._pub: Any | None = None
        self._hand_pubs: dict[str, Any] = {}
        self._subs: list[Any] = []
        self._msg_types: tuple[Any, Any] | None = None

        self._latest_state: dict[str, Any] | None = None
        self._latest_state_time: float = 0.0
        self._latest_hand: dict[str, Any] = {}
        self._latest_hand_time: dict[str, float] = {}

        self._limits = np.zeros((len(self.joint_names), 2), dtype=np.float64)
        self._watched_run_mode: str | None = None
        self.commands_sent = 0
        self.blocked_reason: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def load_limits_from_model(self, model) -> None:
        """Populate joint limits from the MuJoCo model used for IK."""
        import mujoco

        rows = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Joint '{name}' not found in the scene model.")
            rows.append(model.jnt_range[joint_id])
        self._limits = np.asarray(rows, dtype=np.float64)

    def connect(self) -> None:
        """Create the ROS node, publishers and subscriptions."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from joint_interfaces.msg import JointPositionControl
            from end_effector_interfaces.msg import EEJointControl, EEFeedback
            from std_msgs.msg import String
        except ImportError as exc:
            raise ImportError(
                "ROS 2 Python packages are required for the DexForce real backend. "
                "Source /opt/ros/humble/setup.bash and this project's install/setup.bash, "
                "then run with a Python 3.10 interpreter."
            ) from exc

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._msg_types = (JointPositionControl, EEJointControl)
        self.node = Node(self.node_name)

        # /control/joint_position is published RELIABLE by the vendor's own
        # controller (body_controller.cpp), so match it.
        command_qos = QoSProfile(depth=5)
        command_qos.reliability = ReliabilityPolicy.RELIABLE
        command_qos.durability = DurabilityPolicy.VOLATILE

        # The driver publishes feedback BEST_EFFORT; a reliable subscription
        # would fail to match it silently.
        feedback_qos = QoSProfile(depth=10)
        feedback_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        feedback_qos.durability = DurabilityPolicy.VOLATILE

        if not self.read_only:
            self._pub = self.node.create_publisher(
                JointPositionControl, _CONTROL_TOPIC, command_qos
            )
            for side in ("left", "right"):
                self._hand_pubs[side] = self.node.create_publisher(
                    EEJointControl, _HAND_CONTROL_TOPIC.format(side), command_qos
                )

        self._subs.append(
            self.node.create_subscription(
                String, _FEEDBACK_TOPIC,
                lambda msg: self._on_state(msg), feedback_qos,
            )
        )
        for side in ("left", "right"):
            self._subs.append(
                self.node.create_subscription(
                    EEFeedback, _HAND_FEEDBACK_TOPIC.format(side),
                    lambda msg, s=side: self._on_hand(s, msg), feedback_qos,
                )
            )

        mode = "READ-ONLY (no publishers created)" if self.read_only else "commanding"
        print(f"  DexForce ROS node '{self.node_name}': {mode}", flush=True)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """Process incoming feedback without blocking the control loop."""
        if self.node is None:
            return
        self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self) -> None:
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception as exc:  # pragma: no cover - shutdown best effort
                print(f"Warning: DexForce node destroy failed: {exc}")
            self.node = None
            self._pub = None
            self._hand_pubs = {}
            self._subs = []
        if getattr(self, "_rclpy", None) is not None and self._rclpy.ok():
            self._rclpy.shutdown()
            self._rclpy = None

    # -- feedback ----------------------------------------------------------

    def _on_state(self, msg) -> None:
        try:
            state = parse_server_state(msg.data)
        except DexForceFeedbackError as exc:
            print(f"Warning: {exc}")
            return
        self._latest_state = state
        self._latest_state_time = time.monotonic()

    def _on_hand(self, side: str, msg) -> None:
        self._latest_hand[side] = msg
        self._latest_hand_time[side] = time.monotonic()

    def has_state(self) -> bool:
        return self._latest_state is not None

    def state_age_s(self) -> float:
        if self._latest_state_time == 0.0:
            return float("inf")
        return time.monotonic() - self._latest_state_time

    def run_mode(self) -> str | None:
        if self._latest_state is None:
            return None
        value = self._latest_state.get("run_mode")
        return str(value) if value is not None else None

    def server_status(self) -> str | None:
        if self._latest_state is None:
            return None
        value = self._latest_state.get("status")
        return str(value) if value is not None else None

    def read_joint_positions(self) -> dict[str, float]:
        """Current joint angles in radians, keyed by joint name."""
        if self._latest_state is None:
            raise DexForceFeedbackError("No robot_server_state received yet.")
        return joint_positions_by_name(self._latest_state)

    def read_body_positions(self) -> np.ndarray:
        """Torso and head angles in degrees of freedom order (radians)."""
        by_name = self.read_joint_positions()
        missing = [n for n in self.spec.real_body_joint_names() if n not in by_name]
        if missing:
            raise DexForceFeedbackError(f"robot_server_state is missing body joints: {missing}")
        return np.array([by_name[n] for n in self.spec.real_body_joint_names()], dtype=np.float64)

    def read_arm_positions(self, side: str) -> np.ndarray:
        by_name = self.read_joint_positions()
        names = self.spec.arm_joint_names(side)
        missing = [n for n in names if n not in by_name]
        if missing:
            raise DexForceFeedbackError(f"robot_server_state is missing {side} arm joints: {missing}")
        return np.array([by_name[n] for n in names], dtype=np.float64)

    def hand_state(self, side: str) -> Any | None:
        return self._latest_hand.get(side)

    def check_startup_state(self) -> list[str]:
        """Return a list of blocking problems, empty when it is safe to start."""
        problems: list[str] = []
        if not self.has_state():
            problems.append("no /feedback/robot_server_state received")
            return problems

        status = self.server_status()
        if status not in _HEALTHY_STATUS:
            problems.append(f"robot server status is {status!r}, expected one of {sorted(_HEALTHY_STATUS)}")

        run_mode = self.run_mode()
        if run_mode in _BLOCKED_RUN_MODES:
            problems.append(
                f"run_mode is {run_mode!r}; switch the robot to Position mode before teleop"
            )

        error_code = self._latest_state.get("motor_error_code")
        if isinstance(error_code, list) and any(int(c) != 0 for c in error_code):
            problems.append(f"motor_error_code has non-zero entries: {error_code}")

        try:
            positions = self.read_joint_positions()
        except DexForceFeedbackError as exc:
            problems.append(str(exc))
            return problems

        unknown = [n for n in self.joint_names if n not in positions]
        if unknown:
            problems.append(
                "robot_server_state does not report these expected joints, so commands "
                f"for them would be ignored: {unknown}"
            )
        return problems

    # -- commands ----------------------------------------------------------

    def build_command(
        self,
        arm_qpos: dict[str, np.ndarray],
        body_qpos: np.ndarray | None,
        *,
        body_from_feedback: bool = True,
    ) -> np.ndarray:
        """Assemble a 20-value command vector in ``self.joint_names`` order.

        Body joints default to their last measured values so the arms are never
        commanded against a stale torso pose.
        """
        command = np.zeros(len(self.joint_names), dtype=np.float64)
        if body_from_feedback:
            measured = self.read_joint_positions()
            for name in self.spec.real_body_joint_names():
                command[self.joint_index[name]] = measured[name]
        elif body_qpos is not None:
            for offset, name in enumerate(self.spec.real_body_joint_names()):
                command[self.joint_index[name]] = float(body_qpos[offset])

        for side, values in arm_qpos.items():
            names = self.spec.arm_joint_names(side)
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if values.shape[0] != len(names):
                raise ValueError(
                    f"{side} arm command has {values.shape[0]} values, expected {len(names)}"
                )
            for offset, name in enumerate(names):
                command[self.joint_index[name]] = float(values[offset])
        return command

    def clamp_command(self, command: np.ndarray, *, clip_to_limits: bool = True) -> np.ndarray:
        """Apply joint-margin clipping so no target sits on a joint limit."""
        command = np.asarray(command, dtype=np.float64).copy()
        if not np.all(np.isfinite(command)):
            raise ValueError(f"Command contains NaN/Inf: {command}")

        if clip_to_limits and self._limits.size > 0 and bool(np.all(self._limits[:, 1] > self._limits[:, 0])):
            lower = self._limits[:, 0] + self.joint_margin_rad
            upper = self._limits[:, 1] - self.joint_margin_rad
            command = np.clip(command, lower, upper)
        return command

    def send_command(self, command: np.ndarray) -> bool:
        """Publish one 20-joint target, clipped inside the joint limits.

        Returns False when publishing was skipped (read-only, or no publisher).
        """
        # Validate before the read-only short-circuit: a caller passing garbage
        # should learn about it even when nothing is published.
        command = np.asarray(command, dtype=np.float64)
        if not np.all(np.isfinite(command)):
            raise ValueError(f"Command contains NaN/Inf: {command}")

        if self.read_only or self._pub is None or self.node is None:
            return False
        command = self.clamp_command(command)
        JointPositionControl = self._msg_types[0]
        msg = JointPositionControl()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in command]
        self._pub.publish(msg)
        self.commands_sent += 1
        return True

    def send_hand(self, side: str, values: list[float] | np.ndarray) -> bool:
        """Publish a 16-joint hand target in position mode (percent)."""
        if self.read_only:
            return False
        publisher = self._hand_pubs.get(side)
        if publisher is None:
            return False
        values = list(np.asarray(values, dtype=np.float64).reshape(-1))
        if len(values) != len(self.hand_joint_names):
            raise ValueError(
                f"{side} hand command has {len(values)} values, "
                f"expected {len(self.hand_joint_names)}"
            )
        EEJointControl = self._msg_types[1]
        msg = EEJointControl()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.mode = _MODE_POSITION
        msg.joint_names = list(self.hand_joint_names)
        msg.values = [float(v) for v in values]
        publisher.publish(msg)
        return True

    def is_fresh(self) -> bool:
        """False when the host has been silent long enough to warrant a stop."""
        return self.state_age_s() <= self.watchdog_timeout_s

__all__ = [
    "DexForceFeedbackError",
    "DexForceRealOutput",
    "joint_positions_by_name",
    "parse_server_state",
]

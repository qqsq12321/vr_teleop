"""CVTE CARM D3 real-robot adapter and runner, against a fake ``carm`` SDK."""

import inspect
import threading
import time

import mujoco
import numpy as np
import pytest

from example import teleop_cvte_real as runner
from utils.adapters.output.cvte_real import (
    CVTEError,
    CVTEFeedbackError,
    CVTERealOutput,
    sim_gripper_to_opening,
)
from utils.core.input_stream import InputPollResult
from utils.robots.cvte_carm_d3 import CVTECArmD3Spec

SPEC = CVTECArmD3Spec()
# Manual §2.2 soft limits, in radians.
LIMITS = {
    "left": (np.radians([-160, -15, -160, -120, -160, -90, -90]), np.radians([160, 180, 160, 120, 160, 90, 90])),
    "right": (np.radians([-160, -180, -160, -120, -160, -90, -90]), np.radians([160, 15, 160, 120, 160, 90, 90])),
}
MOTION_CALLS = {"set_ready", "track_left_joint", "track_right_joint", "move_left_joint", "move_right_joint", "task_stop"}


class FakeDualBot:
    """The subset of ``carm.CArmDualBot`` the adapter uses, with recorded calls."""

    def __init__(self, ip, *, push_thread=False, start_q=None):
        self.ip = ip
        self.calls = []
        self.connected = True
        self.servo = 0
        self.fsm_state = "IDLE"
        self.state = 0
        self.q = {side: np.array(start_q if start_q is not None else SPEC.arm_home_qpos(side)) for side in ("left", "right")}
        self.joint_cbks = {}
        self.error_cbks = {}
        self._stop = threading.Event()
        if push_thread:
            threading.Thread(target=self._push_loop, daemon=True).start()

    def _record(self, name, *args):
        self.calls.append((name, args))

    def _push_loop(self):
        while not self._stop.is_set():
            self.push()
            time.sleep(0.01)

    def push(self):
        for side, cbk in list(self.joint_cbks.items()):
            cbk(0.0, self.q[side].tolist(), [0.0] * 7, [0.0] * 7)

    def is_connected(self):
        return self.connected

    def disconnect(self):
        self._record("disconnect")
        self._stop.set()
        self.connected = False
        return 1

    def register_error_cbk(self, key, cbk):
        self.error_cbks[key] = cbk

    def release_error_cbk(self, key):
        self.error_cbks.pop(key, None)

    def set_ready(self):
        self._record("set_ready")
        self.servo, self.fsm_state = 1, "POSITION"
        return 1

    def task_stop(self):
        self._record("task_stop")
        return 1

    def set_servo_enable(self, enable):
        raise AssertionError("adapter must never toggle servos (the D3 has no brakes)")

    def emergency_stop(self):
        raise AssertionError("adapter must never trigger an emergency stop (the arms would drop)")


def _add_side_methods(side):
    def register(self, cbk):
        self.joint_cbks[side] = cbk

    def release(self):
        self.joint_cbks.pop(side, None)

    def config(self):
        lower, upper = LIMITS[side]
        return {"dof": 7, "limit_lower": lower.tolist(), "limit_upper": upper.tolist()}

    def eeff_config(self):
        return {"dof": 1, "eeff_type": "gripper", "eeff_lower": [0.0], "eeff_upper": [0.077]}

    def status(self):
        return {"arm_name": f"D3-{side}", "arm_dof": 7, "state": self.state, "servo": self.servo,
                "fsm_state": self.fsm_state, "speed_percentage": 50}

    def cart_pose(self):
        return [0.3, 0.2, -0.3, 0.0, 0.0, 0.0, 1.0]

    def track(self, targets, eeff_pos=-1):
        self._record(f"track_{side}_joint", list(targets), eeff_pos)
        self.q[side] = np.array(targets)
        return 1

    def move(self, target_pos, desire_time=-1, is_sync=True):
        self._record(f"move_{side}_joint", list(target_pos), desire_time, is_sync)
        self.q[side] = np.array(target_pos)
        return 1

    for name, fn in (
        (f"register_{side}_joint_cbk", register),
        (f"release_{side}_joint_cbk", release),
        (f"get_{side}_config", config),
        (f"get_{side}_eeff_config", eeff_config),
        (f"get_{side}_eeff_type", lambda self: "gripper"),
        (f"get_{side}_eeff_connect", lambda self: True),
        (f"get_{side}_eeff_dof", lambda self: 1),
        (f"get_{side}_status", status),
        (f"get_{side}_cart_pose", cart_pose),
        (f"track_{side}_joint", track),
        (f"move_{side}_joint", move),
    ):
        setattr(FakeDualBot, name, fn)


for _side in ("left", "right"):
    _add_side_methods(_side)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SPEC.scene_path))


def _output(model, *, read_only=False, clock=None, sides=("left", "right"), **kwargs):
    bot = FakeDualBot("10.42.0.101")
    output = CVTERealOutput("10.42.0.101", sides=sides, read_only=read_only, factory=lambda ip: bot,
                            clock=clock or FakeClock(), **kwargs)
    output.connect()
    bot.push()
    output.load_limits_from_model(model, {s: SPEC.arm_joint_names(s) for s in ("left", "right")})
    return output, bot


def test_read_only_session_rejects_every_motion_call(model):
    output, bot = _output(model, read_only=True)
    assert output.check_startup_state() == []
    np.testing.assert_allclose(output.read_arm_positions("left"), SPEC.arm_home_qpos("left"))
    with pytest.raises(CVTEError, match="Read-only"):
        output.enable_motion()
    with pytest.raises(CVTEError, match="Read-only"):
        output.send_joints("left", SPEC.arm_home_qpos("left"))
    output.close()
    assert not MOTION_CALLS & {name for name, _ in bot.calls}
    assert ("disconnect", ()) in bot.calls


def test_send_joints_clamps_to_limits(model):
    output, bot = _output(model, joint_margin_rad=0.05)
    output.enable_motion()
    home = SPEC.arm_home_qpos("left")

    # Commands are clamped to limits.
    sent = output.send_joints("left", home + 1.0)
    assert np.all(sent <= output.limits("left")[:, 1])

    # Limits (minus margin) win over the target, e.g. left J2 lower bound -15 deg.
    sent = output.send_joints("left", np.full(7, -10.0))
    assert sent[1] == pytest.approx(np.radians(-15) + 0.05, abs=1e-5)  # Scene stores 6 digits.
    assert np.all(sent >= output.limits("left")[:, 0] - 1e-12)
    name, (targets, eeff) = bot.calls[-1]
    assert name == "track_left_joint" and eeff == -1.0


def test_gripper_opening_is_clamped_into_the_controller_range(model):
    output, bot = _output(model)
    output.enable_motion()
    output.send_joints("right", SPEC.arm_home_qpos("right"), gripper_m=0.5)
    assert bot.calls[-1][1][1] == pytest.approx(0.077)
    assert sim_gripper_to_opening(0.0, (0.0, 0.037), (0.0, 0.077)) == 0.0
    assert sim_gripper_to_opening(0.037, (0.0, 0.037), (0.0, 0.077)) == pytest.approx(0.077)
    assert sim_gripper_to_opening(0.0185, (0.0, 0.037), (0.0, 0.077)) == pytest.approx(0.0385)
    assert sim_gripper_to_opening(-1.0, (0.0, 0.037), (0.01, 0.077)) == pytest.approx(0.01)


def test_unselected_arm_is_never_commanded(model):
    output, bot = _output(model, sides=("left",))
    output.enable_motion()
    with pytest.raises(CVTEError, match="not selected"):
        output.send_joints("right", SPEC.arm_home_qpos("right"))
    assert not any(name == "track_right_joint" for name, _ in bot.calls)


def test_stale_feedback_raises(model):
    clock = FakeClock()
    output, _bot = _output(model, clock=clock, feedback_timeout_s=0.5)
    clock.now += 0.6
    with pytest.raises(CVTEFeedbackError, match="stale"):
        output.read_arm_positions("left")
    assert "left: no fresh joint feedback" in output.check_startup_state()


def test_controller_error_latches_and_warnings_do_not(model):
    output, bot = _output(model)
    output.enable_motion()
    bot.error_cbks["vr_teleop"](3001, "just a warning")
    assert output.fault is None
    bot.error_cbks["vr_teleop"](2005, "joint over current")
    assert "2005" in output.fault
    with pytest.raises(CVTEError, match="Latched fault"):
        output.send_joints("left", SPEC.arm_home_qpos("left"))


def test_check_tracking_returns_error_without_latching(model):
    output, bot = _output(model)
    output.enable_motion()
    output.send_joints("left", SPEC.arm_home_qpos("left"))
    bot.q["left"] = SPEC.arm_home_qpos("left") + 0.5  # Arm blocked away from the command.
    bot.push()
    # Returns error but does not raise or latch fault.
    assert output.check_tracking("left") == pytest.approx(0.5)
    assert output.fault is None
    # Next command should still work.
    output.send_joints("left", SPEC.arm_home_qpos("left"))


def test_close_after_motion_stops_task_but_keeps_servos(model):
    output, bot = _output(model)
    output.enable_motion()
    output.send_joints("left", SPEC.arm_home_qpos("left"))
    output.close()
    names = [name for name, _ in bot.calls]
    assert names[-2:] == ["task_stop", "disconnect"]
    # FakeDualBot.set_servo_enable/emergency_stop raise if they were ever called.


def test_enable_motion_requires_position_mode(model):
    output, bot = _output(model)
    bot.set_ready = lambda: (bot._record("set_ready"), 1)[1]  # Acknowledged but never ready.
    with pytest.raises(CVTEError, match="not ready"):
        output.enable_motion()


def test_error_state_blocks_startup(model):
    output, bot = _output(model)
    bot.state = -1
    assert any("error state" in p for p in output.check_startup_state())


class FakeInput:
    """Moves the left wrist forward 1 mm per frame and pinches the gripper open."""

    def __init__(self, args):
        self.frame = 0

    def poll_arms(self, left_arm, right_arm, _converter, *, allow_stop_gesture=False):
        self.frame += 1
        for arm in (left_arm, right_arm):
            arm.update_from_pose((0.001 * self.frame, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            arm.update_hand_from_pinch(0.10)
        return InputPollResult(saw_valid_data=True)

    def close(self):
        pass


def _run(argv, start_q=None):
    bots = []

    def factory(ip, **kwargs):
        def make_bot(address):
            bots.append(FakeDualBot(address, push_thread=True, start_q=start_q))
            return bots[-1]

        return CVTERealOutput(ip, factory=make_bot, **kwargs)

    args = runner._parse_args(argv)
    assert runner.run(args, output_factory=factory, input_factory=FakeInput) == 0
    return bots[0]


def test_runner_read_only_sends_nothing():
    bot = _run(["--read-only", "--report", "--duration", "0.3"])
    assert not MOTION_CALLS & {name for name, _ in bot.calls}


def test_runner_commands_only_the_selected_arm_with_the_gripper():
    bot = _run(["--arm", "left", "--duration", "0.4"])
    names = [name for name, _ in bot.calls]
    assert names[0] == "set_ready"
    tracks = [args for name, args in bot.calls if name == "track_left_joint"]
    assert tracks, "left arm was never commanded"
    assert "track_right_joint" not in names
    # Pinch 0.10 m = fully open; the gripper moves with the arm command.
    assert tracks[-1][1] == pytest.approx(0.077)
    assert names[-2:] == ["task_stop", "disconnect"]


def test_runner_follows_hand_translation(monkeypatch):
    """Regression: base_link xmat read before mj_forward zeroed all translation."""
    trackers = {}
    original = runner._make_controller

    def capture(model, data, ik_data, side, args):
        controller = original(model, data, ik_data, side, args)
        trackers[side] = controller.tracker
        return controller

    monkeypatch.setattr(runner, "_make_controller", capture)
    _run(["--arm", "left", "--duration", "0.6", "--position-deadband", "0"])
    tracker = trackers["left"]
    # FakeInput moves the hand +X by 1 mm per frame.
    assert tracker.raw_residual[0] > 0.01
    assert tracker.target_position[0] - tracker._initial_site_pos[0] == pytest.approx(
        tracker.residual[0], abs=1e-9
    )
    assert tracker.residual[0] > 0.01


def test_runner_home_ramps_smoothly_from_the_measured_zero_pose():
    bot = _run(["--arm", "right", "--home-only", "--home-duration", "1"], start_q=np.zeros(7))
    names = [name for name, _ in bot.calls]
    assert not any(name.startswith("move_") for name in names)  # Controller rejects TASK_PVT.
    assert "track_left_joint" not in names
    path = np.array([args[0] for name, args in bot.calls if name == "track_right_joint"])
    assert len(path) > 10
    np.testing.assert_allclose(path[-1], SPEC.arm_home_qpos("right"), atol=1e-9)
    # Never faster than the default 1 rad/s limit over the 0.05 s step window.
    steps = np.abs(np.diff(np.vstack([np.zeros(7), path]), axis=0))
    assert np.max(steps) <= 1.0 * 0.05 + 1e-9
    assert names[-2:] == ["task_stop", "disconnect"]


def test_runner_rejects_home_in_read_only():
    with pytest.raises(SystemExit):
        runner._parse_args(["--read-only", "--home"])


def test_adapter_calls_match_the_installed_carm_sdk():
    carm = pytest.importorskip("carm")
    bot_cls = carm.CArmDualBot
    # (method, positional args the adapter passes, excluding self)
    used = [("is_connected", 0), ("disconnect", 0), ("set_ready", 0), ("task_stop", 0),
            ("register_error_cbk", 2)]
    for side in ("left", "right"):
        used += [(f"register_{side}_joint_cbk", 1),
                 (f"get_{side}_config", 0), (f"get_{side}_eeff_config", 0), (f"get_{side}_status", 0),
                 (f"get_{side}_cart_pose", 0), (f"get_{side}_eeff_type", 0),
                 (f"get_{side}_eeff_connect", 0), (f"get_{side}_eeff_dof", 0),
                 (f"track_{side}_joint", 2)]
    for name, nargs in used:
        method = getattr(bot_cls, name, None)
        assert method is not None, f"carm.CArmDualBot.{name} is missing"
        inspect.signature(method).bind(object(), *([None] * nargs))
    # The joint callback is invoked as cbk(t, pos, vel, tau).
    source = inspect.getsource(carm.CArmSingleCol._dispatch_joint)
    assert "reality.get(\"pose\"" in source

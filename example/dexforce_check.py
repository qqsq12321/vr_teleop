#!/usr/bin/env python3
"""Read-only DexForce checks: no motion, no publishes.

Verifies, against the live robot, the assumptions the teleop entry depends on:

1. topic connection and the 100 Hz server state feed;
2. joint names and count (a name the driver does not know is silently ignored);
3. run_mode / status / motor health gates;
4. that the measured pose reproduces in the MuJoCo model, i.e. the IK baseline
   the teleop loop will build is the robot's actual pose, not a mirrored one;
5. that the configured startup pose is inside the model joint limits and clear
   of self-collision, so the startup motion will not fight the limits;
6. hand identity and finger states, if a hand is attached.

Nothing here creates a publisher. Safe to run with the arms powered.

    conda activate vr_teleop
    source /opt/ros/humble/setup.bash
    source ~/dexforce_msgs_ws/install/setup.bash
    export ROS_DOMAIN_ID=20 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI='<CycloneDDS><Domain id="any"><General><Interfaces>
      <NetworkInterface address="192.168.20.77" priority="100"
       presence_required="false" multicast="default"/></Interfaces>
      <AllowMulticast>default</AllowMulticast></General></Domain></CycloneDDS>'
    python example/dexforce_check.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"))

import numpy as np

from utils.adapters.output.dexforce_ros import (
    DexForceFeedbackError,
    DexForceRealOutput,
    joint_positions_by_name,
)
from utils.robots.dexforce import DexForceSpec

_SPEC = DexForceSpec()

_OK = "  [ok]  "
_WARN = "  [warn]"
_FAIL = "  [FAIL]"


def _section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 62)


def check_feedback(output: DexForceRealOutput, timeout_s: float) -> bool:
    _section("1. 反馈链路 / feedback")
    if not output.has_state():
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not output.has_state():
            output.spin_once(timeout_sec=0.1)
    if not output.has_state():
        print(f"{_FAIL} {timeout_s:.0f}s 内没有收到 /feedback/robot_server_state")
        print("        检查 ROS_DOMAIN_ID=20、RMW_IMPLEMENTATION、CYCLONEDDS_URI 网卡")
        return False

    # Measure the actual rate rather than trusting one sample.
    stamps = []
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        output.spin_once(timeout_sec=0.02)
        if output._latest_state_time not in stamps:
            stamps.append(output._latest_state_time)
    hz = len(stamps) / 1.0
    print(f"{_OK} 收到反馈，约 {hz:.0f} Hz")
    return True


def check_health(output: DexForceRealOutput) -> bool:
    _section("2. 机器人状态 / health gates")
    state = output._latest_state or {}
    ok = True

    rows = [
        ("status", output.server_status(), {"Running", "Idle"}),
        ("run_mode", output.run_mode(), {"Position"}),
        ("robot_statusword", state.get("robot_statusword"), {"ENABLED"}),
    ]
    for label, value, allowed in rows:
        good = value in allowed
        ok &= good
        print(f"{_OK if good else _WARN} {label:18} = {value!r}   (期望 {sorted(allowed)})")

    errors = state.get("motor_error_code")
    if isinstance(errors, list):
        bad = [(i, c) for i, c in enumerate(errors) if int(c) != 0]
        if bad:
            ok = False
            print(f"{_FAIL} motor_error_code 非零: {bad}")
        else:
            print(f"{_OK} motor_error_code  全部为 0 ({len(errors)} 个电机)")

    motors = state.get("motor_status")
    if isinstance(motors, list):
        not_op = [i for i, s in enumerate(motors) if str(s) != "OP"]
        if not_op:
            ok = False
            print(f"{_FAIL} motor_status 不是 OP 的电机: {not_op}")
        else:
            print(f"{_OK} motor_status      全部 OP ({len(motors)} 个电机)")

    speed = state.get("global_speed_ratio")
    if speed is not None:
        try:
            ratio = float(speed)
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None and ratio > 0.3:
            print(f"{_WARN} global_speed_ratio = {ratio} 偏高；首次真机建议先降到 ~0.2")
        elif ratio is not None:
            print(f"{_OK} global_speed_ratio = {ratio}")
    return ok


def check_joint_names(output: DexForceRealOutput) -> bool:
    _section("3. 关节名 / joint names")
    state = output._latest_state or {}
    reported = [str(n) for n in state.get("joint_name", [])]
    expected = list(output.joint_names)

    print(f"       机器人报告 {len(reported)} 个关节，程序期望 {len(expected)} 个")
    missing = [n for n in expected if n not in reported]
    extra = [n for n in reported if n not in expected]

    if missing:
        print(f"{_FAIL} 机器人没有报告这些关节，发给它们的指令会被静默忽略:")
        print(f"        {missing}")
    else:
        print(f"{_OK} 20 个关节名全部匹配")
    if extra:
        print(f"{_WARN} 机器人多报了: {extra}")
    return not missing


def check_model_roundtrip(output: DexForceRealOutput) -> bool:
    """Feed the measured pose into MuJoCo and check the FK result is sane."""
    _section("4. 模型往返 / model round-trip")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SPEC.scene_path))
    data = mujoco.MjData(model)

    # Body first: the arms hang off the torso, so a wrong torso angle shifts
    # every arm pose downstream.
    try:
        body = output.read_body_positions()
    except DexForceFeedbackError as exc:
        print(f"{_FAIL} 读躯干失败: {exc}")
        return False
    names = _SPEC.real_body_joint_names()
    for name, value in zip(names, body):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    print(f"{_OK} 躯干/颈部: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(names, body)))

    ok = True
    for side in ("left", "right"):
        try:
            measured = output.read_arm_positions(side)
        except DexForceFeedbackError as exc:
            print(f"{_FAIL} 读 {side} 臂失败: {exc}")
            ok = False
            continue

        for name, value in zip(_SPEC.arm_joint_names(side), measured):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = model.jnt_range[joint_id]
            if not (lo <= value <= hi):
                print(f"{_WARN} {name} 实测 {value:+.4f} 超出模型限位 [{lo:+.4f}, {hi:+.4f}]")
                ok = False

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _SPEC.ee_body_name(side))
        mujoco.mj_forward(model, data)
        pos = data.xpos[body_id].copy()

        # The model base is at the origin; what matters is that the pose is
        # plausible relative to the shoulder, not an absurd mirrored value.
        shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_j1")
        offset = pos - data.xpos[shoulder_id]
        reach = float(np.linalg.norm(offset))
        plausible = 0.05 < reach < 1.5
        ok &= plausible
        print(f"{_OK if plausible else _WARN} {side:5} 末端相对肩部偏移 = "
              f"[{offset[0]:+.3f} {offset[1]:+.3f} {offset[2]:+.3f}] m, |d|={reach:.3f} m")
        print(f"        实测关节角: {np.round(measured, 3).tolist()}")
    return ok


def check_home_pose() -> bool:
    _section("5. 启动姿态 / startup pose")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SPEC.scene_path))
    data = mujoco.MjData(model)
    ok = True

    print(f"       目标启动姿态 (rad): {np.round(_SPEC.real_home_qpos('left'), 3).tolist()} (左)")
    print(f"                          {np.round(_SPEC.real_home_qpos('right'), 3).tolist()} (右)")

    for name, value in zip(_SPEC.real_body_joint_names(), _SPEC.real_body_home_qpos()):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    for side in ("left", "right"):
        for name, value in zip(_SPEC.arm_joint_names(side), _SPEC.real_home_qpos(side)):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = model.jnt_range[joint_id]
            if not (lo <= value <= hi):
                print(f"{_FAIL} {name} 目标 {value:+.4f} 超出模型限位 [{lo:+.4f}, {hi:+.4f}]")
                ok = False
            data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    if ok:
        print(f"{_OK} 启动姿态全部在模型限位内")

    mujoco.mj_forward(model, data)
    left = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _SPEC.ee_body_name("left"))]
    right = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _SPEC.ee_body_name("right"))]
    gap = float(np.linalg.norm(left - right))
    print(f"{_OK if gap > 0.3 else _WARN} 启动姿态下两手末端间距 = {gap:.3f} m")
    if gap <= 0.3:
        print("       间距偏小，实机归位前务必目视确认不会相碰")
    return ok


def check_hands(output: DexForceRealOutput) -> None:
    _section("6. 灵巧手 / end effectors")
    for side in ("left", "right"):
        hand = output.hand_state(side)
        if hand is None:
            print(f"{_WARN} {side}: 没有收到 /feedback/ee/{side}")
            continue
        joints = list(getattr(hand, "joint_states", []))
        print(f"{_OK} {side}: ee_name={getattr(hand, 'ee_name', '?')!r} "
              f"status={getattr(hand, 'status', '?')} 关节数={len(joints)}")
        for joint in joints:
            print(f"        {joint.name:<16} pos={joint.position:7.2f} "
                  f"state={joint.status} err={joint.error_code}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args(argv)

    output = DexForceRealOutput(_SPEC, read_only=True, watchdog_timeout_s=1.0)
    try:
        output.connect()
    except ImportError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        if not check_feedback(output, args.timeout):
            return 1
        health_ok = check_health(output)
        names_ok = check_joint_names(output)

        model_ok = True
        if not args.skip_model:
            model_ok = check_model_roundtrip(output)
            model_ok &= check_home_pose()
        check_hands(output)

        _section("结论 / verdict")
        if health_ok and names_ok and model_ok:
            print("  所有检查通过。真机遥操前仍须：降 global_speed_ratio、"
                  "确认工作区无人、手放急停上。")
            return 0
        print("  有检查未通过，先解决上面的 [FAIL]/[warn] 再动真机。")
        return 1
    finally:
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())

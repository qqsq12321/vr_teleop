"""Teleoperate simulated dual-arm + dual-hand robots via VR hand tracking.

Input sources (--input-source):
  quest3         Meta Quest 3 via UDP (default)
  avp            Apple Vision Pro via avp_stream / Tracking Streamer
  pico4          Pico 4 via relay / direct TCP

Examples:
    python example/teleop_sim.py --port 9000
    python example/teleop_sim.py --robot galbot --hand inspire --port 9000
    python example/teleop_sim.py --robot realman_rm75b --input-source pico4
    python example/teleop_sim.py --input-source avp --avp-ip 192.168.1.100
    python example/teleop_sim.py --input-source pico4
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(_Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"))

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from utils.core.arm_controller import TeleopArmController
from utils.core.input_stream import TeleopInputStream
from utils.core.quaternion import (
    transform_vr_to_robot_pose,
)
from utils.robots.dexforce import DexForceSpec
from utils.robots.galbot import GalbotInspireSpec
from utils.robots.realman_rm75b import RealManRM75BInspireSpec

# ---------------------------------------------------------------------------
# Scene / config paths
# ---------------------------------------------------------------------------

_DEXFORCE_SPEC = DexForceSpec()
_GALBOT_SPEC = GalbotInspireSpec()
_REALMAN_RM75B_SPEC = RealManRM75BInspireSpec()


_DEFAULT_HANDS = {
    "dexforce": "linker_l20",
    "galbot": "inspire",
    "realman_rm75b": "inspire",
}
_HAND_ALIASES = {
    "inspire": "inspire",
    "inspire_rh56dfx": "inspire",
    "linker": "linker_l20",
    "linker_l20": "linker_l20",
}
_ROBOT_HAND_SPECS = {
    ("dexforce", "linker_l20"): _DEXFORCE_SPEC,
    ("galbot", "inspire"): _GALBOT_SPEC,
    ("realman_rm75b", "inspire"): _REALMAN_RM75B_SPEC,
}


def _normalize_hand(robot: str, hand: str | None) -> str:
    if hand is None:
        hand = _DEFAULT_HANDS[robot]
    normalized = _HAND_ALIASES.get(hand)
    if normalized is None:
        supported = ", ".join(sorted(set(_HAND_ALIASES)))
        raise ValueError(f"Unsupported hand '{hand}'. Supported hands: {supported}.")
    if (robot, normalized) not in _ROBOT_HAND_SPECS:
        supported = ", ".join(
            hand_name for robot_name, hand_name in sorted(_ROBOT_HAND_SPECS) if robot_name == robot
        )
        raise ValueError(f"Robot '{robot}' does not support hand '{normalized}'. Supported: {supported}.")
    return normalized


def _resolve_hand_config(args: argparse.Namespace, spec, side: str = "right") -> str:
    """Resolve the retarget config path for the given robot/input source/side."""
    if args.hand_config:
        return args.hand_config
    return str(spec.hand_config_path(args.input_source, side=side))


def _robot_label(args: argparse.Namespace) -> str:
    return f"{args.robot}+{args.hand}"


def _quest3_sim_pose_converter(wrist_pose):
    wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
    wrist_quaternion = (wrist_pose[3], wrist_pose[4], wrist_pose[5], wrist_pose[6])
    return transform_vr_to_robot_pose(wrist_position, wrist_quaternion)


# ---------------------------------------------------------------------------
# Initial pose helpers
# ---------------------------------------------------------------------------


def _write_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...], values) -> None:
    for name, value in zip(names, values):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint '{name}' not found.")
        data.qpos[model.jnt_qposadr[jid]] = float(value)


def _joint_id_map(model: mujoco.MjModel, names: tuple[str, ...] | list[str]) -> dict[str, int]:
    return {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names}


def _set_clamped_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: dict[str, int],
    joint_name: str,
    value: float,
) -> None:
    jid = joint_ids.get(joint_name, -1)
    if jid == -1:
        return
    lo, hi = model.jnt_range[jid]
    data.qpos[model.jnt_qposadr[jid]] = float(np.clip(value, lo, hi))


def _set_clamped_joint_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    joint_ids: dict[str, int],
    joint_name: str,
    value: float,
) -> None:
    jid = joint_ids.get(joint_name, -1)
    if jid == -1:
        return
    lo, hi = model.jnt_range[jid]
    value = float(np.clip(value, lo, hi))
    if args.control_mode == "ctrl":
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if aid != -1:
            ctrl_lo, ctrl_hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(value, ctrl_lo, ctrl_hi))
            return
    data.qpos[model.jnt_qposadr[jid]] = value


def _apply_pico4_head_torso(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    spec,
    torso_joints: dict[str, int],
    yaw: float,
    pitch: float,
    t: float,
) -> None:
    def _set(m, d, joints, name, value):
        _set_clamped_joint_target(m, d, args, joints, name, value)

    spec.apply_head_torso(model, data, _set, torso_joints, yaw, pitch, t)


def _apply_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, spec=None) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    if spec is not None:
        if spec.torso_joint_names:
            _write_joint_qpos(model, data, spec.torso_joint_names, spec.torso_home_qpos)
        if spec.head_joint_names:
            _write_joint_qpos(model, data, spec.head_joint_names, spec.head_home_qpos)
        _write_joint_qpos(model, data, spec.arm_joint_names("left"), spec.arm_home_qpos("left"))
        _write_joint_qpos(model, data, spec.arm_joint_names("right"), spec.arm_home_qpos("right"))
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Bimanual simulation loop
# ---------------------------------------------------------------------------


def _controller_kwargs(spec, side: str, robot_label: str) -> dict:
    return {
        "robot_label": robot_label,
        "home_qpos": spec.arm_home_qpos(side),
        "arm_joint_names": spec.arm_joint_names(side),
        "hand_joint_names": spec.hand_joint_names(side),
        "ee_body_name": spec.ee_body_name(side),
        "base_body_name": spec.base_body_name,
        "hand_qpos_mapping": spec.hand_qpos_mapping,
    }


def _run_bimanual(args: argparse.Namespace) -> None:
    spec = _ROBOT_HAND_SPECS[(args.robot, args.hand)]
    robot_label = _robot_label(args)
    if args.scene is None:
        xml_path = Path(spec.scene_path).expanduser().resolve()
    else:
        xml_path = Path(args.scene).expanduser().resolve()
    print(f"Loading {robot_label} scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose(model, data, spec)
    if model.nu:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1 and model.key_ctrl.shape[0] > key_id:
            data.ctrl[:] = model.key_ctrl[key_id]
        else:
            data.ctrl[: min(model.nu, model.nq)] = data.qpos[: min(model.nu, model.nq)]

    left_hand_config = _resolve_hand_config(args, spec, side="left")
    right_hand_config = _resolve_hand_config(args, spec, side="right")

    left_arm = TeleopArmController(
        model,
        data,
        ik_data,
        "left",
        args,
        hand_config=left_hand_config,
        hand_type=spec.hand_type,
        **_controller_kwargs(spec, "left", robot_label),
    )
    right_arm = TeleopArmController(
        model,
        data,
        ik_data,
        "right",
        args,
        hand_config=right_hand_config,
        hand_type=spec.hand_type,
        **_controller_kwargs(spec, "right", robot_label),
    )

    input_stream = TeleopInputStream(args)

    _torso_joints = _joint_id_map(model, spec.torso_joint_names + spec.head_joint_names)
    _head_zero_q = [None]  # 用列表包装，loop 内可赋值
    _head_zero_y = [None]  # HMD 初始高度（Pico Y 轴）
    _last_head_pose_time = [0.0]

    last_log_time = time.time()

    with viewer.launch_passive(model, data) as vis:
        vis.cam.azimuth = model.vis.global_.azimuth
        vis.cam.elevation = model.vis.global_.elevation
        vis.cam.distance = model.stat.extent * 1.5
        vis.cam.lookat[:] = model.stat.center
        physics_accumulator = 0.0
        last_physics_time = time.perf_counter()

        while vis.is_running():
            loop_start = time.time()

            input_stream.poll_arms(left_arm, right_arm, _quest3_sim_pose_converter)
            if input_stream.pico4_input is not None:
                # Head pose → 躯干/颈部
                head_raw = input_stream.get_pico4_head_pose()
                if head_raw is not None:
                    _last_head_pose_time[0] = time.monotonic()
                    qx, qy, qz, qw = head_raw[3], head_raw[4], head_raw[5], head_raw[6]
                    q_cur = np.array([qw, qx, qy, qz])  # [w,x,y,z]
                    head_y = float(head_raw[1])  # Pico Y 轴 = 高度（向上为正）
                    if _head_zero_q[0] is None:
                        _head_zero_q[0] = q_cur.copy()
                        _head_zero_y[0] = head_y
                    # q_cal = q_cur * q_zero^(-1)，得到相对初始姿态的旋转
                    q0 = _head_zero_q[0]
                    q_zero_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
                    def _qmul(a, b):
                        w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
                        x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
                        y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
                        z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
                        n = np.sqrt(w*w+x*x+y*y+z*z)
                        return np.array([w,x,y,z]) / n if n > 1e-8 else np.array([1.,0,0,0])
                    q = _qmul(q_cur, q_zero_inv)
                    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                    # Pico OpenXR: X右 Y上 Z后
                    # yaw:  绕 Y 轴，左转为正
                    yaw   =  np.arctan2(2*(qy*qw + qx*qz), 1 - 2*(qx*qx + qy*qy))
                    # pitch: 绕 X 轴，低头为负（Pico），取反使低头为正（和机器人一致）
                    pitch =  np.arctan2(2*(qx*qw - qy*qz), 1 - 2*(qx*qx + qz*qz))
                    # 高度：HMD Y 轴相对初始值的变化（站起为正，蹲下为负）
                    dy = float(np.clip((_head_zero_y[0] - head_y), -0.3, 0.3))
                    t = dy / 0.3  # -1~1，蹲下为正
                    _apply_pico4_head_torso(
                        model,
                        data,
                        args,
                        spec,
                        _torso_joints,
                        yaw,
                        pitch,
                        t,
                    )
                elif _last_head_pose_time[0] > 0.0 and time.monotonic() - _last_head_pose_time[0] > 0.5:
                    _head_zero_q[0] = None
                    _head_zero_y[0] = None
                    _last_head_pose_time[0] = 0.0

            q_left = left_arm.step_ik()
            q_right = right_arm.step_ik()
            if args.control_mode == "ctrl":
                now_perf = time.perf_counter()
                physics_accumulator += min(now_perf - last_physics_time, args.max_physics_dt)
                last_physics_time = now_perf

                left_arm.apply_ctrl(q_left)
                right_arm.apply_ctrl(q_right)
                step_count = min(
                    args.max_physics_steps,
                    int(physics_accumulator / model.opt.timestep),
                )
                if step_count > 0:
                    for _ in range(step_count):
                        mujoco.mj_step(model, data)
                    physics_accumulator -= step_count * model.opt.timestep
                else:
                    mujoco.mj_forward(model, data)
            else:
                left_arm.apply_qpos(q_left)
                right_arm.apply_qpos(q_right)
                mujoco.mj_forward(model, data)
            vis.sync()

            now = time.time()
            if now - last_log_time > 1.0:
                print(
                    f"L resid: {left_arm.latest_residual} "
                    f"R resid: {right_arm.latest_residual}"
                )
                last_log_time = now

            sleep_time = model.opt.timestep - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    input_stream.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate dual-arm teleoperation with VR hand tracking.",
    )
    parser.add_argument(
        "--robot",
        default="dexforce",
        choices=sorted(_DEFAULT_HANDS),
        help="Robot body to teleoperate: dexforce (default), galbot, or realman_rm75b.",
    )
    parser.add_argument(
        "--hand",
        default=None,
        choices=sorted(_HAND_ALIASES),
        help=(
            "Dexterous hand/end-effector: defaults to linker_l20 for dexforce "
            "and inspire for galbot/realman_rm75b."
        ),
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Override scene XML/URDF path. Defaults to the selected robot scene.",
    )
    parser.add_argument(
        "--control-mode",
        default="qpos",
        choices=["qpos", "ctrl"],
        help="qpos directly sets IK results; ctrl drives position actuators through MuJoCo physics.",
    )
    parser.add_argument(
        "--max-physics-steps",
        type=int,
        default=20,
        help="Maximum MuJoCo substeps per viewer loop in --control-mode ctrl.",
    )
    parser.add_argument(
        "--max-physics-dt",
        type=float,
        default=0.05,
        help="Maximum wall-clock time accumulated per loop for ctrl-mode physics catch-up.",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--hand-config",
        default=None,
        help="Path to the hand retargeter YAML config.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=3.0,
        help="Scale for wrist position residuals.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help="EMA smoothing factor for wrist residuals (0-1).",
    )
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=1.0,
        help="Weight for orientation error in IK.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1e-3,
        help="Damping factor for IK solver.",
    )
    parser.add_argument(
        "--ik-current-weight",
        type=float,
        default=0.1,
        help="Weight for penalizing deviation from current pose in IK.",
    )
    parser.add_argument(
        "--input-source",
        default="quest3",
        choices=["quest3", "avp", "pico4"],
        help="Input device: quest3 (UDP, default), avp (Vision Pro), or pico4.",
    )
    parser.add_argument(
        "--avp-ip",
        default="192.168.1.100",
        help="Apple Vision Pro IP address (used with --input-source avp).",
    )
    parser.add_argument(
        "--pico4-mode",
        default="relay",
        choices=["relay", "direct"],
        help="Pico 4 input mode: relay daemon (default) or direct TCP server.",
    )
    parser.add_argument(
        "--pico4-port",
        type=int,
        default=63901,
        help="Pico 4 direct-mode TCP listen port.",
    )
    parser.add_argument(
        "--pico4-relay-host",
        default="127.0.0.1",
        help="Pico 4 relay daemon host.",
    )
    parser.add_argument(
        "--pico4-relay-port",
        type=int,
        default=63902,
        help="Pico 4 relay daemon port.",
    )
    parser.add_argument(
        "--pico4-broadcast-port",
        type=int,
        default=29888,
        help="Pico 4 direct-mode UDP broadcast port.",
    )
    args = parser.parse_args()
    try:
        args.hand = _normalize_hand(args.robot, args.hand)
    except ValueError as exc:
        parser.error(str(exc))

    _run_bimanual(args)


if __name__ == "__main__":
    main()

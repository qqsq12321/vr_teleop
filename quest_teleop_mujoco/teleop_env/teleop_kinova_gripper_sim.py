from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from util.ik import solve_pose_ik
from util.quaternion import (
    matrix_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    transform_vr_to_robot_pose,
)
from util.udp_socket import (
    parse_right_landmarks,
    parse_right_wrist_pose,
    pinch_distance_from_landmarks,
)

# Gen3 home pose (7 arm joints, from gen3.xml keyframe "home")
# Gen3 official home pose (backup):
# [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633]
_HOME_QPOS = np.array(
    [0.0, 0.585, 3.14, -1.6, 0.0, -0.861, 1.57],
    dtype=np.float64,
)

# Robotiq 2F-85 gripper control range
_GRIPPER_MAX = 0.8  # ctrl=0 → open, ctrl=0.8 → closed

# Pinch distance normalization (meters)
_PINCH_MAX_DISTANCE = 0.1


def _default_scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene" / "scene_kinova_gen3.xml"


def _apply_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    n = min(model.nq, _HOME_QPOS.shape[0])
    data.qpos[:n] = _HOME_QPOS[:n]
    n_ctrl = min(model.nu, n)
    if n_ctrl:
        data.ctrl[:n_ctrl] = _HOME_QPOS[:n_ctrl]
    mujoco.mj_forward(model, data)


def _pinch_to_gripper(pinch_distance: float) -> float:
    """Map pinch distance to Robotiq gripper control value.

    Small distance (pinched) → large ctrl (closed).
    Large distance (open hand) → small ctrl (open).
    """
    scaled = pinch_distance / _PINCH_MAX_DISTANCE
    clamped = min(1.0, max(0.0, scaled))
    return _GRIPPER_MAX * (1.0 - clamped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop the Kinova Gen3 with wrist residuals.")
    parser.add_argument(
        "--scene",
        default=str(_default_scene_path()),
        help="Path to a MuJoCo XML scene file.",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--site",
        default="kinova_ee_site",
        help="End-effector site name.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.5,
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
    args = parser.parse_args()

    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    if site_id == -1:
        raise ValueError(f"Site '{args.site}' not found in model.")

    gripper_actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
    )

    mujoco.mj_forward(model, data)
    initial_site_pos = data.site_xpos[site_id].copy()
    initial_site_quat = matrix_to_quaternion(
        data.site_xmat[site_id].reshape(3, 3).copy()
    )
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_xmat = None
    if base_body_id != -1:
        base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)

    initial_wrist_position = None
    initial_wrist_quaternion = None
    target_position = initial_site_pos.copy()
    target_quaternion = np.array(initial_site_quat, dtype=np.float64)
    last_log_time = time.time()
    latest_residual = None
    latest_euler_residual = None
    smoothed_residual = None
    latest_gripper_cmd = None

    with viewer.launch_passive(model, data) as vis:
        vis.cam.azimuth = model.vis.global_.azimuth
        vis.cam.elevation = model.vis.global_.elevation
        vis.cam.distance = model.stat.extent * 1.5
        vis.cam.lookat[:] = model.stat.center
        while vis.is_running():
            try:
                packet, _ = sock.recvfrom(1024)
            except BlockingIOError:
                packet = None

            if packet is not None:
                message = packet.decode("utf-8", errors="ignore")
                landmarks = parse_right_landmarks(message)
                if landmarks is not None and gripper_actuator_id != -1:
                    pinch_distance = pinch_distance_from_landmarks(landmarks)
                    if pinch_distance is not None:
                        latest_gripper_cmd = _pinch_to_gripper(pinch_distance)
                wrist_pose = parse_right_wrist_pose(message)
                if wrist_pose is not None:
                    wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
                    wrist_quaternion = (
                        wrist_pose[3],
                        wrist_pose[4],
                        wrist_pose[5],
                        wrist_pose[6],
                    )
                    robot_position, robot_quaternion = transform_vr_to_robot_pose(
                        wrist_position, wrist_quaternion
                    )
                    if initial_wrist_position is None:
                        initial_wrist_position = robot_position
                        initial_wrist_quaternion = robot_quaternion
                    else:
                        residual = np.array(
                            [
                                robot_position[0] - initial_wrist_position[0],
                                robot_position[1] - initial_wrist_position[1],
                                robot_position[2] - initial_wrist_position[2],
                            ],
                            dtype=np.float64,
                        )
                        if base_xmat is not None:
                            residual = base_xmat @ residual
                        if smoothed_residual is None:
                            smoothed_residual = residual
                        else:
                            smoothed_residual = (
                                args.ema_alpha * residual
                                + (1.0 - args.ema_alpha) * smoothed_residual
                            )
                        target_position = (
                            initial_site_pos + args.position_scale * smoothed_residual
                        )

                        relative_quaternion = quaternion_multiply(
                            robot_quaternion,
                            quaternion_inverse(initial_wrist_quaternion),
                        )
                        # Negate X/Y components to fix rotation inversion for Gen3
                        relative_quaternion = (
                            -relative_quaternion[0],
                            -relative_quaternion[1],
                            relative_quaternion[2],
                            relative_quaternion[3],
                        )
                        target_quaternion = np.array(
                            quaternion_multiply(relative_quaternion, initial_site_quat),
                            dtype=np.float64,
                        )
                        norm = np.linalg.norm(target_quaternion)
                        if norm > 0.0:
                            target_quaternion /= norm
                        euler_residual = quaternion_to_euler_xyz(
                            relative_quaternion[0],
                            relative_quaternion[1],
                            relative_quaternion[2],
                            relative_quaternion[3],
                        )
                        latest_residual = smoothed_residual
                        latest_euler_residual = euler_residual

            now = time.time()
            if (
                latest_residual is not None
                and latest_euler_residual is not None
                and now - last_log_time > 1.0
            ):
                print(
                    f"Wrist residual (xyz): {latest_residual.tolist()} "
                    f"euler: {list(latest_euler_residual)}"
                )
                last_log_time = now

            q_sol = solve_pose_ik(
                model,
                ik_data,
                site_id,
                target_position,
                target_quaternion,
                data.qpos[: model.nq],
                rot_weight=args.rot_weight,
                damping=args.ik_damping,
                current_q_weight=args.ik_current_weight,
                home_qpos=_HOME_QPOS,
                skip_tail_joints=0,
            )
            if model.nu:
                ctrl = data.ctrl.copy()
                for act_id in range(model.nu):
                    # Only update joint-based actuators (skip tendon-based gripper)
                    if model.actuator_trntype[act_id] != 0:  # mjTRN_JOINT = 0
                        continue
                    joint_id = model.actuator_trnid[act_id, 0]
                    if joint_id < 0:
                        continue
                    qadr = model.jnt_qposadr[joint_id]
                    if qadr < q_sol.shape[0]:
                        ctrl[act_id] = q_sol[qadr]
                data.ctrl[:] = ctrl
            if latest_gripper_cmd is not None and gripper_actuator_id != -1:
                data.ctrl[gripper_actuator_id] = latest_gripper_cmd
            mujoco.mj_step(model, data)
            vis.sync()

            sleep_time = model.opt.timestep
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()

"""Inverse-kinematics helpers for MuJoCo models."""

import math

import mujoco
import numpy as np

from util.quaternion import quaternion_to_matrix


def solve_position_ik(
    model: mujoco.MjModel,
    workspace: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iters: int = 30,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> np.ndarray:
    """Levenberg-Marquardt IK for a site position."""
    q = np.asarray(q_init, dtype=np.float64).copy()
    for _ in range(max_iters):
        workspace.qpos[: model.nq] = q
        workspace.qvel[:] = 0.0
        mujoco.mj_forward(model, workspace)
        err = target_pos - workspace.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, workspace, jacp, None, site_id)
        JJ = jacp @ jacp.T + damping * np.eye(3)
        dq = jacp.T @ np.linalg.solve(JJ, err)
        q += dq
    return q


def solve_pose_ik(
    model: mujoco.MjModel,
    workspace: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iters: int = 30,
    tol: float = 1e-4,
    rot_weight: float = 1.0,
    home_qpos: np.ndarray | None = np.array(
        [0.0, 0.9, -0.9, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float64
    ),
    home_weight: float = 0.01,
    current_q_weight: float = 0.0,
    skip_tail_joints: int = 2,
    damping: float = 1e-3,
    dof_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Levenberg-Marquardt IK for a site pose (position + orientation)."""
    q = np.asarray(q_init, dtype=np.float64).copy()
    target_rot = np.asarray(quaternion_to_matrix(target_quat), dtype=np.float64)

    # Heuristic: assume home_qpos defines the active robot joints.
    # Everything else (e.g. free joints for objects) is ignored.
    n_robot = 0
    if home_qpos is not None:
        n_robot = len(home_qpos)

    # Sanity check: if n_robot > model.nq, something is wrong, clamp it.
    if n_robot > model.nq:
        n_robot = model.nq
    
    # If home_qpos is not provided or empty, default to full state (not recommended with free extra objects)
    if n_robot == 0:
        n_robot = model.nq

    for _ in range(max_iters):
        workspace.qpos[: model.nq] = q
        workspace.qvel[:] = 0.0
        mujoco.mj_forward(model, workspace)
        current_pos = workspace.site_xpos[site_id]
        current_rot = workspace.site_xmat[site_id].reshape(3, 3)

        err_pos = target_pos - current_pos
        rot_err = _rotation_error(target_rot, current_rot)
        err = np.hstack([err_pos, rot_weight * rot_err])
        if np.linalg.norm(err) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, workspace, jacp, jacr, site_id)
        jac = np.vstack([jacp, rot_weight * jacr])

        if dof_indices is not None:
             # Zero out columns NOT in dof_indices
            mask = np.ones(model.nv, dtype=bool)
            mask[dof_indices] = False
            jac[:, mask] = 0.0
        else:
            # Zero out non-robot columns (e.g. cube free joint)
            if n_robot < model.nv:
                jac[:, n_robot:] = 0.0

            if skip_tail_joints:
                # Skip the last few joints of the ROBOT (e.g. gripper fingers)
                # Ensure we don't go negative
                start_skip = max(0, n_robot - skip_tail_joints)
                jac[:, start_skip:n_robot] = 0.0

        if home_weight > 0.0 and home_qpos is not None:
             # Only penalize deviation for robot joints
            home = np.asarray(home_qpos, dtype=np.float64)
            scale = math.sqrt(home_weight)
            
            # Jacobian for home term
            # We extend jac to include n_robot rows (or dof_indices rows)
            
            if dof_indices is not None:
                # If dof_indices provided, home_qpos must match the size of dof_indices?
                # Or we assume home_qpos is full size and we only pick dof_indices?
                # For safety, let's assume if dof_indices is used, home_qpos should optionally be full
                # or aligned. But to be safe let's assume home_qpos might not match.
                # Ideally caller passes a relevant home_qpos slice.
                # Let's assume home_qpos CORRESPONDS to the active joints if dof_indices is None.
                # If dof_indices IS provided, handling home_qpos is tricky if lengths differ.
                # Let's ignore home_weight if dof_indices is set UNLESS implemented carefully.
                # Implementing: Assume home_qpos matches q[dof_indices]
                if len(home) == len(dof_indices):
                     err_home = scale * (home - q[dof_indices])
                     jac_home = np.zeros((len(dof_indices), model.nv))
                     for i, dof_idx in enumerate(dof_indices):
                         jac_home[i, dof_idx] = scale
                     
                     err = np.hstack([err, err_home])
                     jac = np.vstack([jac, jac_home])
            else:
                # Deviation only for the n_robot joints
                err_home = scale * (home - q[:n_robot])
                jac_home = np.zeros((n_robot, model.nv))
                np.fill_diagonal(jac_home[:n_robot, :n_robot], scale)
                
                if skip_tail_joints:
                    start_skip = max(0, n_robot - skip_tail_joints)
                    err_home[start_skip:] = 0.0
                    jac_home[start_skip:, :] = 0.0
                    jac_home[:, start_skip:n_robot] = 0.0

                err = np.hstack([err, err_home])
                jac = np.vstack([jac, jac_home])

        if current_q_weight > 0.0:
            scale = math.sqrt(current_q_weight)
            
            if dof_indices is not None:
                err_curr = scale * (q_init[dof_indices] - q[dof_indices])
                jac_curr = np.zeros((len(dof_indices), model.nv))
                for i, dof_idx in enumerate(dof_indices):
                    jac_curr[i, dof_idx] = scale
                
                err = np.hstack([err, err_curr])
                jac = np.vstack([jac, jac_curr])
            else:
                # Penalize deviation from the INITIAL q for this step (not the current iteration's q)
                # This requires us to use q_init which is passed in.
                # However, q_init might be full size, we only care about robot part.
                q_initial_robot = q_init[:n_robot]
                err_curr = scale * (q_initial_robot - q[:n_robot])
                
                jac_curr = np.zeros((n_robot, model.nv))
                np.fill_diagonal(jac_curr[:n_robot, :n_robot], scale)
                
                if skip_tail_joints:
                    start_skip = max(0, n_robot - skip_tail_joints)
                    err_curr[start_skip:] = 0.0
                    jac_curr[start_skip:, :] = 0.0
                    jac_curr[:, start_skip:n_robot] = 0.0

                err = np.hstack([err, err_curr])
                jac = np.vstack([jac, jac_curr])

        JJ = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(JJ, err)
        
        # Zero out non-robot dq again to be safe
        if dof_indices is not None:
             dq[mask] = 0.0
        elif n_robot < model.nv:
            dq[n_robot:] = 0.0
        
        # Update q using integratePos to handle free joints correctly (even if we don't move them)
        mujoco.mj_integratePos(model, q, dq, 1.0)
    return q


def _rotation_error(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    r_err = target_rot @ current_rot.T
    trace = np.trace(r_err)
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle

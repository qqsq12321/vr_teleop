#!/usr/bin/env python3
"""Watch capsule self- and floor-collision avoidance on the single-arm RM75-B, no VR needed.

The flange target sweeps from the home pose to ``--goal`` and back, on a loop.
The default goal is by the upper arm -- a fold that runs the forearm and hand
into the arm when nothing stops it. The collision capsules are drawn over the
robot, each coloured by its worst clearance to another capsule or the floor:

    green   clear of the influence distance
    orange  inside it: the IK is pushing these apart
    red     inside the hard margin: the real robot would hold this frame

Compare the two runs:

    python tools/view_capsule_avoidance.py --hand inspire
    python tools/view_capsule_avoidance.py --hand inspire --no-avoid

Push the hand into the floor instead:

    python tools/view_capsule_avoidance.py --hand inspire --goal 0.5 0 -0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.core.body.ik import solve_pose_ik  # noqa: E402
from utils.core.capsule_collision import CapsuleSet, draw_capsules  # noqa: E402
from utils.core.quaternion import matrix_to_quaternion  # noqa: E402
from utils.robots.realman_rm75b_single import HAND_SPECS  # noqa: E402

# Matches the real entrypoint, so this shows what the hardware would do.
INFLUENCE_M = 0.05
MARGIN_M = 0.005
JOINT_LIMIT_MARGIN_RAD = 0.05
ARM = np.arange(7)
SWEEP_FRAMES = 200  # One way; the sweep then returns.
FRAME_S = 0.02


class AvoidanceDemo:
    def __init__(self, hand, goal, avoid):
        self.spec = HAND_SPECS[hand]()
        self.model = mujoco.MjModel.from_xml_path(str(self.spec.scene_path))
        self.data = mujoco.MjData(self.model)
        self.workspace = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.capsules = CapsuleSet(
            self.model, self.spec.collision_chain, hand=self.spec.collision_hand,
            influence=INFLUENCE_M, margin=MARGIN_M, floor_z=self.spec.collision_floor_z,
        )
        self.avoid = avoid
        self.site = self.model.site("right_tcp").id
        self.start = self.data.site_xpos[self.site].copy()
        self.quat = matrix_to_quaternion(self.data.site_xmat[self.site].reshape(3, 3))
        self.goal = np.asarray(goal, dtype=np.float64)
        self.frame = 0

    def target(self):
        phase = self.frame % (2 * SWEEP_FRAMES)
        fraction = phase / SWEEP_FRAMES if phase < SWEEP_FRAMES else 2 - phase / SWEEP_FRAMES
        return self.start + (self.goal - self.start) * fraction

    def step(self):
        target = self.target()
        q = solve_pose_ik(
            self.model, self.workspace, self.site, target, self.quat, self.data.qpos.copy(),
            home_qpos=self.data.qpos[:7], skip_tail_joints=0, dof_indices=ARM,
            current_q_weight=0.3,
            collision=self.capsules if self.avoid else None,
            joint_limit_margin=JOINT_LIMIT_MARGIN_RAD if self.avoid else None,
        )
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.frame += 1
        return target

    def capsule_states(self):
        return self.capsules.clearance_per_capsule(self.data)

    def draw(self, scene, target):
        clearance = draw_capsules(scene, self.capsules, self.data)
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.015, 0, 0]), target, np.eye(3).flatten(),
                                np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float32))
            scene.ngeom += 1
        return clearance


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hand", choices=tuple(HAND_SPECS), default="inspire")
    parser.add_argument("--no-avoid", action="store_true",
                        help="Turn avoidance off to see the arm run into itself or the floor")
    parser.add_argument("--goal", type=float, nargs=3, default=(0.05, 0.0, 0.35),
                        metavar=("X", "Y", "Z"),
                        help="World point the flange is dragged toward (default: by the upper arm)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Run this many frames without a window and print the result")
    args = parser.parse_args(argv)
    demo = AvoidanceDemo(args.hand, args.goal, avoid=not args.no_avoid)
    label = "avoidance OFF" if args.no_avoid else "avoidance ON"

    if args.frames:
        worst = np.inf
        for _ in range(args.frames):
            demo.step()
            worst = min(worst, float(demo.capsule_states().min()))
        print(f"{label}: worst clearance over {args.frames} frames = {worst * 1000:+.0f} mm")
        return 0

    from mujoco import viewer

    print(f"{label}. Green clear, orange avoiding, red would be held on the robot.")
    last_print = 0.0
    with viewer.launch_passive(demo.model, demo.data) as vis:
        vis.cam.distance = demo.model.stat.extent * 1.5
        vis.cam.lookat[:] = demo.model.stat.center
        while vis.is_running():
            start = time.time()
            target = demo.step()
            with vis.lock():
                clearance = demo.draw(vis.user_scn, target)
            vis.sync()
            if start - last_print > 1.0:
                print(f"  min clearance {clearance * 1000:+6.0f} mm", flush=True)
                last_print = start
            remaining = FRAME_S - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Drive the real single-arm RM75-B with its collision capsules shown in a window.

Runs ``example/teleop_rm75b_single_real.py`` unchanged -- same arguments, IK,
avoidance and hard backstop -- and mirrors every frame the backstop checks
into a MuJoCo viewer, capsules coloured by their worst clearance to another
capsule or the floor:

    green   clear of the influence distance
    orange  inside it: the IK is pushing these apart
    red     inside the hard margin: the arm is holding, this pose was NOT sent

The window shows the pose sent each frame -- the IK solution, cut short at the
margin when it would cross it -- or, when the arm holds, the refused pose.
Closing the window does not stop the arm; Ctrl+C does.

    python example/pico4_daemon.py          # in another terminal
    python tests/teleop_real_capsules.py --enable-motion --hand inspire

There is deliberately no --no-avoid here: on hardware that would let the arm
hit itself or the floor. Use tests/teleop_sim_capsules.py --no-avoid for that.

Not a pytest module (no ``test_`` prefix): it moves the real arm.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.core.capsule_collision import draw_capsules  # noqa: E402

PRINT_PERIOD_S = 1.0


def _load_entry():
    spec = importlib.util.spec_from_file_location(
        "teleop_rm75b_single_real", ROOT / "example" / "teleop_rm75b_single_real.py"
    )
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    return entry


def _with_viewer(backstop_cls):
    import mujoco
    from mujoco import viewer

    class _ViewedBackstop(backstop_cls):
        """The real backstop, unchanged, plus a window showing what it checked."""

        def __init__(self, model, spec, **kwargs):
            super().__init__(model, spec, **kwargs)
            self._view_data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, self._view_data, 0)
            mujoco.mj_forward(model, self._view_data)
            # Opened before the arm is connected, so a missing display fails
            # the run while the arm is still idle.
            self._vis = viewer.launch_passive(model, self._view_data)
            self._vis.cam.distance = model.stat.extent * 1.5
            self._vis.cam.lookat[:] = model.stat.center
            self._last_print = 0.0

        def limit(self, q_from, q_to):
            sent = super().limit(q_from, q_to)
            if self._vis.is_running():
                with self._vis.lock():
                    self._view_data.qpos[:] = q_to if sent is None else sent
                    mujoco.mj_kinematics(self.model, self._view_data)
                    clearance = draw_capsules(self._vis.user_scn, self.capsules, self._view_data)
                self._vis.sync()
                now = time.monotonic()
                if now - self._last_print > PRINT_PERIOD_S:
                    _distance, pair = self.capsules.min_clearance(self._view_data)
                    where = f" ({pair[0]} / {pair[1]})" if pair else ""
                    print(f"  min clearance {clearance * 1000:+6.0f} mm{where}", flush=True)
                    self._last_print = now
            return sent

    return _ViewedBackstop


def main(argv=None):
    entry = _load_entry()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--enable-motion" in argv:
        entry._CollisionBackstop = _with_viewer(entry._CollisionBackstop)
        print("Capsules: green clear, orange avoiding, red holding (not sent).", flush=True)
    else:
        print("Capsules are only shown with --enable-motion.", flush=True)
    return entry.main(argv)


if __name__ == "__main__":
    sys.exit(main())

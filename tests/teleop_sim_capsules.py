#!/usr/bin/env python3
"""Teleoperate the single-arm RM75-B in sim with its collision capsules drawn.

Runs ``example/teleop_sim.py`` unchanged -- same input, IK and avoidance --
and overlays the very capsules the IK avoids with on every viewer frame, each
coloured by its worst clearance to another capsule or the floor:

    green   clear of the influence distance
    orange  inside it: the IK is pushing these apart
    red     inside the hard margin: the real robot would hold this frame

Takes every teleop_sim argument; ``--robot`` defaults to realman_rm75b_single:

    python example/pico4_daemon.py          # in another terminal
    python tests/teleop_sim_capsules.py --hand inspire --input-source pico4

Add ``--no-avoid`` to keep drawing the capsules but take them out of the IK,
to see the arm run into itself or the floor.

Not a pytest module (no ``test_`` prefix): it opens a window and needs VR input.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from example import teleop_sim  # noqa: E402
from utils.core.capsule_collision import draw_capsules  # noqa: E402

PRINT_PERIOD_S = 1.0
NO_AVOID = "--no-avoid"


class _CapsuleViewer:
    """Stands in for ``mujoco.viewer``; draws capsules before each sync."""

    def __init__(self, real_viewer, capsule_sets):
        self._real = real_viewer
        self._capsule_sets = capsule_sets

    def launch_passive(self, model, data, *args, **kwargs):
        return _CapsuleHandle(self._real.launch_passive(model, data, *args, **kwargs),
                              data, self._capsule_sets)


class _CapsuleHandle:
    def __init__(self, handle, data, capsule_sets):
        self._handle = handle
        self._data = data
        self._capsule_sets = capsule_sets
        self._last_print = 0.0

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def sync(self):
        if self._capsule_sets:
            # One arm per controller, but they share the same capsule set.
            capsules = self._capsule_sets[0]
            with self._handle.lock():
                clearance = draw_capsules(self._handle.user_scn, capsules, self._data)
            now = time.monotonic()
            if now - self._last_print > PRINT_PERIOD_S:
                _distance, pair = capsules.min_clearance(self._data)
                where = f" ({pair[0]} / {pair[1]})" if pair else ""
                print(f"  min clearance {clearance * 1000:+6.0f} mm{where}", flush=True)
                self._last_print = now
        self._handle.sync()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--robot" or a.startswith("--robot=") for a in argv):
        argv = ["--robot", "realman_rm75b_single", *argv]

    # Ours, not teleop_sim's: strip it before teleop_sim parses the rest.
    avoid = NO_AVOID not in argv
    argv = [a for a in argv if a != NO_AVOID]

    capsule_sets = []
    build = teleop_sim._collision_kwargs

    def _capture(model, spec):
        kwargs = build(model, spec)
        if "collision" in kwargs and not capsule_sets:
            capsule_sets.append(kwargs["collision"])
        # Without avoidance the capsules are still drawn, but the IK never
        # sees them (nor the joint-limit margin), as on robots that opt out.
        return kwargs if avoid else {}

    teleop_sim._collision_kwargs = _capture
    teleop_sim.viewer = _CapsuleViewer(teleop_sim.viewer, capsule_sets)
    sys.argv = [sys.argv[0], *argv]
    print("avoidance ON" if avoid else "avoidance OFF (capsules drawn only)")
    print("Capsules: green clear, orange avoiding, red would be held on the robot.")
    teleop_sim.main()
    if not capsule_sets:
        print("This robot declares no collision capsules; nothing was drawn.")


if __name__ == "__main__":
    main()

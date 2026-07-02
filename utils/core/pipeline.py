"""Teleoperation pipeline shell.

The current entry scripts still own their control loops. This module defines
the stable core boundary for the next extraction step: normalized input frames
enter here, normalized commands leave here.
"""

from __future__ import annotations

from utils.core.types import SideCommand, TeleopCommand, TeleopFrame


class TeleopPipeline:
    def step(self, frame: TeleopFrame | None, now: float) -> TeleopCommand:
        if frame is not None and frame.stop_requested:
            return TeleopCommand(left=SideCommand(), right=SideCommand(), stop_requested=True)
        raise NotImplementedError("Move the existing entry-script control loop here in the next refactor step.")

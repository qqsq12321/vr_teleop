"""Common output backend interface."""

from __future__ import annotations

from typing import Protocol

from utils.core.types import TeleopCommand


class TeleopOutput(Protocol):
    def apply(self, command: TeleopCommand) -> None:
        """Apply a normalized teleop command to the backend."""

    def step(self) -> None:
        """Advance or flush the backend."""

    def close(self) -> None:
        """Release backend resources."""

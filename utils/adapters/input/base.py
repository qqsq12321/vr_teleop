"""Common input-device interface."""

from __future__ import annotations

from typing import Protocol

from utils.core.types import TeleopFrame


class InputDevice(Protocol):
    def poll(self) -> TeleopFrame | None:
        """Return the latest normalized teleop frame, or None when no data arrived."""

    def close(self) -> None:
        """Release network/device resources."""

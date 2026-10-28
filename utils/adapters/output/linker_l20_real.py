"""RealMan companion adapter for one or two Linker L20 hands.

The RM75-B arm is controlled through :mod:`realman_real`; this module owns only
Linker L20 RS485 traffic.  A pair can share one RS485 adapter because the two
hands use different Modbus slave IDs (right ``0x29``, left ``0x2a``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from third_party.AnyDexRetarget.example.output.real.drivers_linker_l20 import (
    LinkerL20Output,
    LinkerL20RS485,
)

RIGHT_SLAVE_ID = 0x29
LEFT_SLAVE_ID = 0x2A
SIDES = ("left", "right")


class LinkerL20RealOutput:
    """Send AnyDex qpos targets to Linker L20 hand(s) over RS485.

    ``port`` is the normal single-adapter setup.  ``left_port`` and
    ``right_port`` may be supplied when the two hands use separate USB/RS485
    adapters.  If both requested sides use the same port, one serial handle is
    shared and the slave ID is selected before every Modbus request.
    """

    def __init__(
        self,
        *,
        sides: Iterable[str] = ("right",),
        port: str = "/dev/ttyUSB0",
        left_port: str | None = None,
        right_port: str | None = None,
        baudrate: int = 460800,
        left_slave_id: int = LEFT_SLAVE_ID,
        right_slave_id: int = RIGHT_SLAVE_ID,
        speed: int = 200,
        current_limit: int = 200,
        clear_faults: bool = False,
        open_on_exit: bool = False,
        print_registers: bool = False,
        dry_run: bool = False,
    ) -> None:
        selected = tuple(dict.fromkeys(str(side).lower() for side in sides))
        if not selected or any(side not in SIDES for side in selected):
            raise ValueError(f"Linker L20 sides must be left/right, got {selected}")
        self.sides = selected
        self._drivers: dict[str, LinkerL20Output] = {}
        self._shared_transport: LinkerL20RS485 | None = None
        self._closed = False

        ports = {
            "left": left_port or port,
            "right": right_port or port,
        }
        slave_ids = {"left": int(left_slave_id), "right": int(right_slave_id)}
        common_port = len(selected) > 1 and len({ports[side] for side in selected}) == 1

        try:
            if common_port and not dry_run:
                common = ports[selected[0]]
                self._shared_transport = LinkerL20RS485(
                    port=common, baudrate=baudrate, slave_id=slave_ids[selected[0]]
                )

            for side in selected:
                kwargs = dict(
                    port=ports[side],
                    baudrate=baudrate,
                    slave_id=slave_ids[side],
                    speed=speed,
                    current_limit=current_limit,
                    clear_faults=clear_faults,
                    open_on_exit=open_on_exit,
                    print_registers=print_registers,
                    dry_run=dry_run,
                )
                if self._shared_transport is not None:
                    kwargs["transport"] = self._shared_transport
                self._drivers[side] = LinkerL20Output(**kwargs)
        except BaseException:
            self.close()
            raise

    def open(self, side: str) -> None:
        self._driver(side).open_palm()

    def send(self, side: str, qpos: Sequence[float] | np.ndarray, joint_names: Sequence[str]) -> None:
        self._driver(side).send(qpos, joint_names)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for driver in self._drivers.values():
                driver.close()
        finally:
            self._drivers.clear()
            # Drivers using a shared transport deliberately do not close it.
            if self._shared_transport is not None:
                self._shared_transport.close()
                self._shared_transport = None

    def _driver(self, side: str) -> LinkerL20Output:
        side = str(side).lower()
        try:
            return self._drivers[side]
        except KeyError as exc:
            raise ValueError(f"Linker L20 side {side!r} was not selected") from exc


__all__ = ["LEFT_SLAVE_ID", "RIGHT_SLAVE_ID", "LinkerL20RealOutput"]

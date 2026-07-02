"""Inspire RH56DFX real-hand output backend."""

from __future__ import annotations

import time

import numpy as np

from utils.robots.galbot import GalbotInspireSpec


_RESPONSE_TIMEOUT_S = 0.01
_SPEC = GalbotInspireSpec()


def retarget_qpos_to_channels(retarget_output: np.ndarray) -> list[int]:
    """Map AnyDexRetarget's 12-D Inspire qpos output to 6 hardware channels."""
    output = np.asarray(retarget_output, dtype=np.float64)
    if output.shape[0] < len(_SPEC.hand_joint_suffixes):
        raise ValueError(f"Expected at least 12 Inspire joints, got shape {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError("Invalid Inspire retarget output: contains NaN/Inf")

    result: list[int] = []
    for idx, max_rad, invert in zip(
        _SPEC.inspire_channel_indices,
        _SPEC.inspire_channel_max_rad,
        _SPEC.inspire_channel_invert,
    ):
        value = float(np.clip(output[idx] / max_rad, 0.0, 1.0))
        if invert:
            value = 1.0 - value
        result.append(int(value * 2000))
    return result


def encode_inspire_channels(channels: list[int]) -> list[int]:
    """Encode 0..2000 channel values into the Inspire 0..1000 serial range."""
    if len(channels) != 6:
        raise ValueError(f"Inspire hand expects 6 channels, got {len(channels)}")
    return [int(np.clip(round(ch / 2.0), 0, 1000)) for ch in channels]


class InspireSerialOutput:
    """Direct serial controller for an Inspire RH56DFX hand."""

    def __init__(self, port_name: str, baudrate: int = 115200, hand_id: int = 1):
        try:
            import serial
        except ImportError as exc:
            raise ImportError("pyserial is required for Inspire hand control.") from exc

        self._port = serial.Serial(port_name, baudrate, timeout=_RESPONSE_TIMEOUT_S)
        self._hand_id = int(hand_id)
        self._port_name = port_name
        self._baudrate = int(baudrate)

    def send_hand_qpos(self, qpos: np.ndarray) -> None:
        channels = retarget_qpos_to_channels(qpos)
        encoded = encode_inspire_channels(channels)
        packet = bytearray([0xEB, 0x90, self._hand_id, 0x0F, 0x12, 0xCE, 0x05])
        for angle in encoded:
            packet.append(angle & 0xFF)
            packet.append((angle >> 8) & 0xFF)
        checksum = sum(packet[2 : 2 + 0x0F + 3])
        packet.append(checksum & 0xFF)
        self._port.write(packet)
        self._read_response()

    def send(self, qpos: np.ndarray, joint_names: list[str] | None = None) -> None:
        self.send_hand_qpos(qpos)

    def close(self) -> None:
        if self._port.is_open:
            self._port.close()

    def _read_response(self) -> bytes:
        deadline = time.time() + _RESPONSE_TIMEOUT_S
        input_bytes = bytearray()
        while time.time() < deadline:
            chunk = self._port.read(self._port.in_waiting or 1)
            if not chunk:
                break
            input_bytes += chunk
        return bytes(input_bytes)

import math
import socket
from typing import Optional, Sequence

import numpy as np


def _parse_pose(message: str, prefix: str) -> Optional[Sequence[float]]:
    prefix = prefix.lower()
    for line in message.splitlines():
        if not line.strip().lower().startswith(prefix):
            continue
        _, _, rest = line.partition(":")
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                break
        if len(values) == 7:
            return values
    return None


def _parse_landmarks(message: str, prefix: str) -> Optional[Sequence[float]]:
    prefix = prefix.lower()
    for line in message.splitlines():
        if not line.strip().lower().startswith(prefix):
            continue
        _, _, rest = line.partition(":")
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                break
        if len(values) == 63:
            return values
    return None


def parse_right_wrist_pose(message: str) -> Optional[Sequence[float]]:
    return _parse_pose(message, "right wrist")


def parse_left_wrist_pose(message: str) -> Optional[Sequence[float]]:
    return _parse_pose(message, "left wrist")


def parse_right_landmarks(message: str) -> Optional[Sequence[float]]:
    return _parse_landmarks(message, "right landmarks")


def parse_left_landmarks(message: str) -> Optional[Sequence[float]]:
    return _parse_landmarks(message, "left landmarks")


def pinch_distance_from_landmarks(
    landmarks: Sequence[float], thumb_tip_index: int = 4, index_tip_index: int = 8
) -> Optional[float]:
    if len(landmarks) < 63:
        return None
    thumb_offset = thumb_tip_index * 3
    index_offset = index_tip_index * 3
    if index_offset + 2 >= len(landmarks):
        return None
    thumb = landmarks[thumb_offset : thumb_offset + 3]
    index_tip = landmarks[index_offset : index_offset + 3]
    dx = thumb[0] - index_tip[0]
    dy = thumb[1] - index_tip[1]
    dz = thumb[2] - index_tip[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def clamp_pinch_ratio(pinch_distance: float, max_distance: float = 0.1) -> float:
    """Normalize pinch distance to 0-1 ratio (0 = fully pinched)."""
    if max_distance <= 0:
        return 0.0
    return float(np.clip(pinch_distance / max_distance, 0.0, 1.0))


def make_socket(port: int) -> socket.socket:
    """Create a non-blocking UDP socket bound to the given port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    return sock


def recv_latest_packet(sock: socket.socket) -> bytes | None:
    """Drain the socket buffer and return only the latest packet."""
    latest = None
    while True:
        try:
            latest, _ = sock.recvfrom(65536)
        except BlockingIOError:
            break
    return latest

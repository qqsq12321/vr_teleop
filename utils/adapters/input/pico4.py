"""Pico 4 input device for teleoperation.

Two operating modes:

  relay (default):
    Connects to pico_relay_daemon's local relay port (127.0.0.1:63902).
    Frame format: [4B device_id_len LE][device_id][4B json_len LE][JSON]
    Use this when pico_relay_daemon is already running (the normal case).

  direct:
    PC acts as TCP server on port 63901. Pico 4 discovers PC via UDP broadcast
    and connects directly. Use when relay daemon is NOT running.
    Frame format: [0x3F][cmd][4B payload_len LE][payload][8B timestamp_ns][0xA5]

Usage:
    # relay mode (default)
    python teleop_sim.py --input pico4

    # direct mode
    python teleop_sim.py --input pico4 --pico4-mode direct --pico4-port 63901
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from utils.core.quaternion import matrix_to_quaternion, quaternion_to_matrix

logger = logging.getLogger(__name__)

# ── Protocol constants ───────────────────────────────────────────────────────
_CLIENT_HEAD = 0x3F
_SERVER_HEAD = 0xCF
_TAIL = 0xA5
_CMD_CONNECT = 0x19
_CMD_HEARTBEAT = 0x23
_CMD_DEVICE_STATE_JSON = 0x6D
_CMD_BATTERY = 0x1A
_CMD_SENSOR = 0x1B
_CMD_UDP_TCP_IP = 0x7E

_HEADER_SIZE = 6
_TAIL_SIZE = 9
_MIN_FRAME = _HEADER_SIZE + _TAIL_SIZE

DEFAULT_RELAY_HOST = '127.0.0.1'
DEFAULT_RELAY_PORT = 63902
DEFAULT_DIRECT_PORT = 63901
DEFAULT_UDP_BROADCAST_PORT = 29888
_BROADCAST_INTERVAL_S = 5.0
_HEARTBEAT_TIMEOUT_S = 20.0
_RELAY_RECONNECT_S = 2.0

# 26 SDK joints → 21 MediaPipe joints
# Removes: Palm(0), Index_metacarpal(6), Middle_metacarpal(11),
#          Ring_metacarpal(16), Little_metacarpal(21)
JOINT_21_INDICES = [
    1,  2,  3,  4,  5,
    7,  8,  9,  10,
    12, 13, 14, 15,
    17, 18, 19, 20,
    22, 23, 24, 25,
]

# Pico/OpenXR-like frame: X right, Y up, Z backward. Robot frame: X forward,
# Y left, Z up.
_PICO_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_PICO_TO_ROBOT_T = _PICO_TO_ROBOT.T


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_csv_floats(s: str) -> list[float]:
    out = []
    for part in s.split(','):
        part = part.strip()
        if part:
            try:
                out.append(float(part))
            except ValueError:
                out.append(0.0)
    return out


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def transform_pico4_to_robot_pose(
    pico_pos: np.ndarray,
    pico_quat: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert Pico hand pose to the robot frame used by teleop."""
    robot_pos = _PICO_TO_ROBOT @ pico_pos
    pico_rot = np.asarray(quaternion_to_matrix(_quat_normalize(pico_quat)), dtype=np.float64)
    robot_rot = _PICO_TO_ROBOT @ pico_rot @ _PICO_TO_ROBOT_T
    robot_quat = matrix_to_quaternion(
        tuple(tuple(float(v) for v in row) for row in robot_rot)
    )
    return (
        (float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])),
        robot_quat,
    )


def _parse_hand_state(
    hand_j: dict,
    wrist_joint_index: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(hand_j, dict):
        return None, None, None
    is_active = hand_j.get('isActive', 0)
    try:
        is_active = int(float(str(is_active)))
    except (ValueError, TypeError):
        return None, None, None
    if is_active != 1:
        return None, None, None

    joints = hand_j.get('HandJointLocations')
    if not isinstance(joints, list) or len(joints) != 26:
        return None, None, None

    all_joints = np.zeros((26, 7), dtype=np.float32)
    for i, joint in enumerate(joints):
        if not isinstance(joint, dict):
            continue
        p = joint.get('p')
        if not isinstance(p, str):
            continue
        vals = _parse_csv_floats(p)
        if len(vals) >= 7:
            all_joints[i] = vals[:7]
        elif len(vals) >= 3:
            all_joints[i, :3] = vals[:3]
            all_joints[i, 6] = 1.0

    result = np.zeros((21, 3), dtype=np.float32)
    for dst_i, src_i in enumerate(JOINT_21_INDICES):
        result[dst_i] = all_joints[src_i, :3]

    result -= result[0].copy()
    wrist_index = int(np.clip(wrist_joint_index, 0, all_joints.shape[0] - 1))
    wrist_pose = all_joints[wrist_index].copy()
    return result, wrist_pose, all_joints


def _parse_hand_joints(hand_j: dict) -> Optional[np.ndarray]:
    landmarks, _wrist_pose, _all_joints = _parse_hand_state(hand_j, wrist_joint_index=0)
    return landmarks


def _parse_tracking_json(
    payload: bytes,
    wrist_joint_index: int = 0,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],  # head pose [x,y,z,qx,qy,qz,qw]
]:
    try:
        outer = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None, None, None, None, None, None

    if isinstance(outer, dict) and 'value' in outer and isinstance(outer['value'], str):
        inner_str = outer['value'].replace('\\', '')
        try:
            data = json.loads(inner_str)
        except json.JSONDecodeError:
            return None, None, None, None, None, None, None
    else:
        data = outer

    if not isinstance(data, dict):
        return None, None, None, None, None, None, None

    hand_section = data.get('Hand', {})
    if not isinstance(hand_section, dict):
        return None, None, None, None, None, None, None

    left, left_wrist, left_joints = _parse_hand_state(
        hand_section.get('leftHand', {}),
        wrist_joint_index=wrist_joint_index,
    )
    right, right_wrist, right_joints = _parse_hand_state(
        hand_section.get('rightHand', {}),
        wrist_joint_index=wrist_joint_index,
    )

    # Parse head pose
    head_pose = None
    head_section = data.get('Head', {})
    if isinstance(head_section, dict):
        pose_str = head_section.get('pose', '')
        if isinstance(pose_str, str):
            vals = _parse_csv_floats(pose_str)
            if len(vals) >= 7:
                head_pose = np.array(vals[:7], dtype=np.float64)

    return left, right, left_wrist, right_wrist, left_joints, right_joints, head_pose


# ── Relay mode frame parser ───────────────────────────────────────────────────
# Frame: [4B device_id_len LE][device_id bytes][4B json_len LE][JSON bytes]

class _RelayFrameParser:
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def try_parse(self) -> Optional[bytes]:
        """Return next JSON payload bytes, or None if incomplete."""
        while len(self._buf) >= 8:
            id_len = struct.unpack_from('<I', self._buf, 0)[0]
            # sanity: device_id should be a short ASCII string
            if id_len > 256 or len(self._buf) < 4 + id_len + 4:
                # corrupt, try shifting
                del self._buf[0]
                continue
            json_len = struct.unpack_from('<I', self._buf, 4 + id_len)[0]
            if json_len > 10_000_000:
                del self._buf[0]
                continue
            total = 4 + id_len + 4 + json_len
            if len(self._buf) < total:
                return None
            payload = bytes(self._buf[4 + id_len + 4: total])
            del self._buf[:total]
            return payload
        return None


# ── Direct mode frame parser ──────────────────────────────────────────────────

class _DirectFrameParser:
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def try_parse(self) -> Optional[dict]:
        while self._buf:
            try:
                idx = self._buf.index(_CLIENT_HEAD)
            except ValueError:
                self._buf.clear()
                return None
            if idx:
                del self._buf[:idx]
            if len(self._buf) < _MIN_FRAME:
                return None
            cmd = self._buf[1]
            plen = struct.unpack_from('<I', self._buf, 2)[0]
            if plen > 10_000_000:
                del self._buf[0]
                continue
            total = _HEADER_SIZE + plen + _TAIL_SIZE
            if len(self._buf) < total:
                return None
            if self._buf[total - 1] != _TAIL:
                del self._buf[0]
                continue
            payload = bytes(self._buf[_HEADER_SIZE: _HEADER_SIZE + plen])
            del self._buf[:total]
            return {'cmd': cmd, 'payload': payload}
        return None


# ── UDP broadcaster (direct mode only) ───────────────────────────────────────

def _build_broadcast_packet(ip: str) -> bytes:
    ip_bytes = ip.encode('utf-8')
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF
    pkt = bytearray()
    pkt.append(_SERVER_HEAD)
    pkt.append(_CMD_UDP_TCP_IP)
    pkt.extend(struct.pack('<I', len(ip_bytes)))
    pkt.extend(ip_bytes)
    pkt.extend(struct.pack('<Q', ts_ms))
    pkt.append(_TAIL)
    return bytes(pkt)


def _get_local_ips() -> list[str]:
    ips = []
    try:
        import fcntl

        for _index, name in socket.if_nameindex():
            if name == 'lo' or not Path(f'/sys/class/net/{name}/device').exists():
                continue
            request = struct.pack('256s', name[:15].encode('utf-8'))
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as iface_sock:
                try:
                    flags = struct.unpack(
                        'H', fcntl.ioctl(iface_sock.fileno(), 0x8913, request)[16:18]
                    )[0]
                    if not flags & 0x1:
                        continue
                    response = fcntl.ioctl(iface_sock.fileno(), 0x8915, request)
                except OSError:
                    continue
            ips.append(socket.inet_ntoa(response[20:24]))
    except (ImportError, OSError):
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            if info[0] == socket.AF_INET:
                ip = info[4][0]
                if not ip.startswith('127.'):
                    ips.append(ip)
    except Exception:
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ips.append(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return list(dict.fromkeys(ips))


# ── Main class ────────────────────────────────────────────────────────────────

class Pico4:
    """Pico 4 hand tracking input device.

    Args:
        mode: 'relay' (default) or 'direct'.
        relay_host: relay daemon host (relay mode only, default '127.0.0.1').
        relay_port: relay daemon port (relay mode only, default 63902).
        port: TCP listen port (direct mode only, default 63901).
        broadcast_port: UDP broadcast port (direct mode only, default 29888).
    """

    def __init__(
        self,
        mode: str = 'relay',
        relay_host: str = DEFAULT_RELAY_HOST,
        relay_port: int = DEFAULT_RELAY_PORT,
        port: int = DEFAULT_DIRECT_PORT,
        broadcast_port: int = DEFAULT_UDP_BROADCAST_PORT,
        wrist_joint_index: int = 0,
    ) -> None:
        self._mode = mode.lower()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._wrist_joint_index = wrist_joint_index

        self._left_hand: Optional[np.ndarray] = None
        self._right_hand: Optional[np.ndarray] = None
        self._left_wrist_pose: Optional[np.ndarray] = None
        self._right_wrist_pose: Optional[np.ndarray] = None
        self._left_joints: Optional[np.ndarray] = None
        self._right_joints: Optional[np.ndarray] = None
        self._head_pose: Optional[np.ndarray] = None
        self._last_update: float = 0.0

        if self._mode == 'relay':
            self._relay_host = relay_host
            self._relay_port = relay_port
            self._thread = threading.Thread(target=self._run_relay, daemon=True)
            logger.info('Pico4 relay mode: connecting to %s:%d', relay_host, relay_port)
        else:
            self._direct_port = port
            self._broadcast_port = broadcast_port
            self._thread = threading.Thread(target=self._run_direct_server, daemon=True)
            logger.info('Pico4 direct mode: listening on TCP %d', port)

        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────

    def get_fingers_data(self) -> dict:
        empty = np.zeros((21, 3), dtype=np.float32)
        with self._lock:
            left = self._left_hand.copy() if self._left_hand is not None else empty.copy()
            right = self._right_hand.copy() if self._right_hand is not None else empty.copy()
        return {'left_fingers': left, 'right_fingers': right}

    def poll(self) -> bool:
        """Return True if any Pico frame has been received."""
        with self._lock:
            return self._last_update > 0.0

    def get_landmarks_mediapipe(self, side: str = "right") -> Optional[np.ndarray]:
        """Return Pico hand landmarks as wrist-relative MediaPipe (21, 3)."""
        with self._lock:
            hand = self._right_hand if side == "right" else self._left_hand
            return hand.copy() if hand is not None else None

    def get_wrist_pose(
        self,
        side: str = "right",
    ) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
        """Return wrist pose converted to the robot frame."""
        with self._lock:
            pose = self._right_wrist_pose if side == "right" else self._left_wrist_pose
            if pose is None:
                return None
            pose = pose.copy()
        if pose.shape[0] < 7 or np.allclose(pose[:3], 0.0):
            return None
        pico_pos = np.asarray(pose[:3], dtype=np.float64)
        pico_quat = np.asarray(pose[3:7], dtype=np.float64)
        return transform_pico4_to_robot_pose(pico_pos, pico_quat)

    def get_pinch_distance(self, side: str = "right") -> Optional[float]:
        """Return thumb-index pinch distance in meters."""
        with self._lock:
            joints = self._right_joints if side == "right" else self._left_joints
            if joints is None:
                return None
            joints = joints.copy()
        # SDK joint 5 is thumb tip and 10 is index tip for the 26-joint Pico skeleton.
        thumb_tip = joints[5, :3]
        index_tip = joints[10, :3]
        if np.allclose(thumb_tip, 0.0) or np.allclose(index_tip, 0.0):
            return None
        return float(np.linalg.norm(thumb_tip - index_tip))

    def get_head_pose(self) -> Optional[np.ndarray]:
        """Return HMD pose as [x, y, z, qx, qy, qz, qw] in Pico frame, or None."""
        with self._lock:
            pose = self._head_pose.copy() if self._head_pose is not None else None
        if pose is None or pose.shape[0] < 7:
            return None
        if np.allclose(pose[:3], 0.0):
            return None
        quat = np.asarray(pose[3:7], dtype=np.float64)
        if float(np.linalg.norm(quat)) < 1e-6:
            return None
        return pose

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── relay mode ────────────────────────────────────────────────────────

    def _run_relay(self) -> None:
        """Connect to relay daemon on 127.0.0.1:63902 and read JSON frames."""
        while not self._stop.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self._relay_host, self._relay_port))
                sock.settimeout(1.0)
                logger.info('Pico4 connected to relay %s:%d', self._relay_host, self._relay_port)
                self._relay_loop(sock)
            except (ConnectionRefusedError, OSError) as e:
                logger.warning('Pico4 relay connect failed: %s, retry in %.1fs', e, _RELAY_RECONNECT_S)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
            if not self._stop.is_set():
                self._stop.wait(_RELAY_RECONNECT_S)

    def _relay_loop(self, sock: socket.socket) -> None:
        parser = _RelayFrameParser()
        while not self._stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            parser.feed(data)
            while True:
                payload = parser.try_parse()
                if payload is None:
                    break
                left, right, left_wrist, right_wrist, left_joints, right_joints, head_pose = _parse_tracking_json(
                    payload,
                    wrist_joint_index=self._wrist_joint_index,
                )
                with self._lock:
                    if left is not None:
                        self._left_hand = left
                        self._left_wrist_pose = left_wrist
                        self._left_joints = left_joints
                    if right is not None:
                        self._right_hand = right
                        self._right_wrist_pose = right_wrist
                        self._right_joints = right_joints
                    if head_pose is not None:
                        self._head_pose = head_pose
                    if left is not None or right is not None:
                        self._last_update = time.monotonic()
        logger.info('Pico4 relay connection closed')

    # ── direct mode ───────────────────────────────────────────────────────

    def _run_direct_server(self) -> None:
        # Start UDP broadcaster
        broadcaster = threading.Thread(target=self._broadcast_loop, daemon=True)
        broadcaster.start()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(('0.0.0.0', self._direct_port))
        except OSError as e:
            logger.error('Pico4 direct mode: cannot bind port %d: %s', self._direct_port, e)
            return
        srv.listen(1)
        srv.settimeout(1.0)
        logger.info('Pico4 direct mode: TCP server on port %d', self._direct_port)
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                logger.info('Pico4 direct: connected from %s', addr)
                t = threading.Thread(
                    target=self._direct_client_loop, args=(conn,), daemon=True
                )
                t.start()
        finally:
            srv.close()

    def _direct_client_loop(self, conn: socket.socket) -> None:
        parser = _DirectFrameParser()
        conn.settimeout(1.0)
        last_hb = time.monotonic()
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_hb > _HEARTBEAT_TIMEOUT_S:
                    break
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                parser.feed(data)
                while True:
                    frame = parser.try_parse()
                    if frame is None:
                        break
                    if frame['cmd'] in (_CMD_HEARTBEAT, _CMD_CONNECT,
                                        _CMD_BATTERY, _CMD_SENSOR):
                        last_hb = time.monotonic()
                    if frame['cmd'] == _CMD_DEVICE_STATE_JSON:
                        left, right, left_wrist, right_wrist, left_joints, right_joints, head_pose = _parse_tracking_json(
                            frame['payload'],
                            wrist_joint_index=self._wrist_joint_index,
                        )
                        with self._lock:
                            if left is not None:
                                self._left_hand = left
                                self._left_wrist_pose = left_wrist
                                self._left_joints = left_joints
                            if right is not None:
                                self._right_hand = right
                                self._right_wrist_pose = right_wrist
                                self._right_joints = right_joints
                            if head_pose is not None:
                                self._head_pose = head_pose
                            if left is not None or right is not None:
                                self._last_update = time.monotonic()
        finally:
            conn.close()
            logger.info('Pico4 direct: client disconnected')

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                for ip in _get_local_ips():
                    parts = ip.split('.')
                    if len(parts) == 4:
                        bcast = '.'.join(parts[:3]) + '.255'
                        pkt = _build_broadcast_packet(ip)
                        try:
                            sock.sendto(pkt, (bcast, self._broadcast_port))
                        except OSError:
                            pass
                self._stop.wait(_BROADCAST_INTERVAL_S)
        finally:
            sock.close()

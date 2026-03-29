import math
import socket
import sys
from typing import Optional, Sequence

from util.quaternion import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    transform_vr_to_robot_pose,
)


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


def pinch_to_gripper(pinch_distance: float, max_distance: float = 0.1) -> float:
    if max_distance <= 0:
        return 0.0
    scaled = pinch_distance / max_distance
    clamped = min(1.0, max(0.0, scaled))
    return 0.035 * clamped


def create_udp_listener(port=9000):
    """Create a UDP socket that listens on the specified port and parses wrist data."""
    try:
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to the port
        sock.bind(('0.0.0.0', port))
        
        print(f"UDP listener started on port {port}")
        print("Waiting for messages... (Press Ctrl+C to stop)")
        
        initial_wrist_position = None
        initial_wrist_quaternion = None

        # Listen for incoming messages
        while True:
            data, addr = sock.recvfrom(1024)
            message = data.decode("utf-8", errors="ignore")
            wrist_pose = parse_right_wrist_pose(message)
            if wrist_pose is None:
                continue

            wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
            wrist_quaternion = (wrist_pose[3], wrist_pose[4], wrist_pose[5], wrist_pose[6])
            robot_position, robot_quaternion = transform_vr_to_robot_pose(
                wrist_position, wrist_quaternion
            )

            if initial_wrist_position is None:
                initial_wrist_position = robot_position
                initial_wrist_quaternion = robot_quaternion
                print(f"Initial wrist pose: {initial_wrist_position} {initial_wrist_quaternion}")
                continue

            residual = [
                robot_position[0] - initial_wrist_position[0],
                robot_position[1] - initial_wrist_position[1],
                robot_position[2] - initial_wrist_position[2],
            ]
            relative_quaternion = quaternion_multiply(
                robot_quaternion, quaternion_inverse(initial_wrist_quaternion)
            )
            euler_residual = quaternion_to_euler_xyz(
                relative_quaternion[0],
                relative_quaternion[1],
                relative_quaternion[2],
                relative_quaternion[3],
            )
            landmarks = parse_right_landmarks(message)
            pinch_distance = None
            gripper = None
            if landmarks is not None:
                pinch_distance = pinch_distance_from_landmarks(landmarks)
                if pinch_distance is not None:
                    gripper = pinch_to_gripper(pinch_distance)
            print(
                # f"Wrist residual (xyz): {residual} euler: {euler_residual} "
                f"pinch: {pinch_distance} gripper: {gripper}"
            )
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        sock.close()

if __name__ == "__main__":
    create_udp_listener(9000)

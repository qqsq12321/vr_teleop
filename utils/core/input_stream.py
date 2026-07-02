"""Shared VR input polling for teleoperation entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from utils.adapters.input.quest3 import (
    make_socket,
    recv_latest_packet,
    parse_left_landmarks,
    parse_left_wrist_pose,
    parse_right_landmarks,
    parse_right_wrist_pose,
)


@dataclass
class InputPollResult:
    saw_valid_data: bool = False
    stop_requested: bool = False


class TeleopInputStream:
    """Own the selected VR input source and update arm controllers from it."""

    def __init__(self, args):
        self.args = args
        self.sock = None
        self.avp_input = None
        self.pico4_input = None

        if args.input_source == "avp":
            from utils.adapters.input.avp import AVPInput

            self.avp_input = AVPInput(ip=args.avp_ip)
            print(f"  Input: Apple Vision Pro ({args.avp_ip})")
        elif args.input_source == "pico4":
            from utils.adapters.input.pico4 import Pico4

            self.pico4_input = Pico4(
                mode=args.pico4_mode,
                relay_host=args.pico4_relay_host,
                relay_port=args.pico4_relay_port,
                port=args.pico4_port,
                broadcast_port=args.pico4_broadcast_port,
            )
            print(f"  Input: Pico 4 ({args.pico4_mode})")
        else:
            self.sock = make_socket(args.port)
            print(f"  Input: Quest 3 (UDP port {args.port})")

    def poll_arms(self, left_arm, right_arm, quest3_pose_converter, *, allow_stop_gesture: bool = False) -> InputPollResult:
        result = InputPollResult()

        if self.avp_input is not None:
            if not self.avp_input.poll():
                return result
            if allow_stop_gesture and self.avp_input.check_stop_gesture():
                result.stop_requested = True
                return result
            for arm, side in ((left_arm, "left"), (right_arm, "right")):
                if arm.update_hand_from_mediapipe(self.avp_input.get_landmarks_mediapipe(side)):
                    result.saw_valid_data = True
                wrist = self.avp_input.get_wrist_pose(side)
                if wrist is not None:
                    arm.update_from_pose(*wrist)
                    result.saw_valid_data = True
            return result

        if self.pico4_input is not None:
            for arm, side in ((left_arm, "left"), (right_arm, "right")):
                if arm.update_hand_from_mediapipe(self.pico4_input.get_landmarks_mediapipe(side)):
                    result.saw_valid_data = True
                wrist = self.pico4_input.get_wrist_pose(side)
                if wrist is not None:
                    arm.update_from_pose(*wrist)
                    result.saw_valid_data = True
            return result

        packet = recv_latest_packet(self.sock)
        if packet is None:
            return result
        message = packet.decode("utf-8", errors="ignore")
        for arm, side in ((left_arm, "left"), (right_arm, "right")):
            landmarks = parse_left_landmarks(message) if side == "left" else parse_right_landmarks(message)
            if arm.update_hand_from_raw_landmarks(landmarks):
                result.saw_valid_data = True
            wrist_pose = parse_left_wrist_pose(message) if side == "left" else parse_right_wrist_pose(message)
            if wrist_pose is not None:
                robot_position, robot_quaternion = quest3_pose_converter(wrist_pose)
                arm.update_from_pose(robot_position, robot_quaternion)
                result.saw_valid_data = True
        return result

    def get_pico4_head_pose(self):
        if self.pico4_input is None:
            return None
        return self.pico4_input.get_head_pose()

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        if self.pico4_input is not None:
            self.pico4_input.stop()

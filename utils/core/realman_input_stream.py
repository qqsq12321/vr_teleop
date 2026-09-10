"""RealMan-only input policy. Never replay the simulation's cached samples."""
from dataclasses import dataclass, field

from utils.adapters.input.pico4_tracked import TrackedPico4
from utils.core.input_stream import InputPollResult as SimInputPollResult


@dataclass
class InputPollResult(SimInputPollResult):
    wrist_received_at: dict[str, float] = field(default_factory=dict)
    hand_received_at: dict[str, float] = field(default_factory=dict)
    invalid_wrist_sides: set[str] = field(default_factory=set)
    invalid_hand_sides: set[str] = field(default_factory=set)


class RealManInputStream:
    def __init__(self, args):
        if args.input_source != 'pico4':
            raise ValueError('RealMan requires timestamped Pico tracking')
        self._pico_sequence = 0
        self.pico4_input = TrackedPico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
        print(f'  Input: Pico 4 ({args.pico4_mode}; RealMan fresh-frame policy)')

    def poll_arms(self, left_arm, right_arm, _converter):
        result = InputPollResult()
        frame = self.pico4_input.latest_frame(self._pico_sequence)
        if frame is None:
            return result
        self._pico_sequence = frame.sequence
        for arm, side in ((left_arm, 'left'), (right_arm, 'right')):
            if arm.update_hand_from_mediapipe(frame.hands[side]):
                result.saw_valid_data = True
                result.hand_received_at[side] = frame.received_at
            else:
                result.invalid_hand_sides.add(side)
            wrist = frame.wrist_pose(side)
            if wrist is not None:
                arm.update_from_pose(*wrist)
                result.saw_valid_data = True
                result.wrist_received_at[side] = frame.received_at
            else:
                result.invalid_wrist_sides.add(side)
        return result

    def close(self):
        self.pico4_input.stop()

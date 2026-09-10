import threading
from types import SimpleNamespace

import numpy as np
import pytest

from utils.adapters.input.pico4_tracked import TrackedPico4 as Pico4, _parse_hand_state
from utils.core.realman_input_stream import RealManInputStream as TeleopInputStream


def pico_without_thread():
    pico = Pico4.__new__(Pico4)
    pico._lock = threading.Lock()
    pico._sequence = 0
    pico._last_update = 0
    return pico


def publish(pico, *, left=True, right=True):
    hand = np.ones((21,3))
    wrist = np.array([0.2, 1.2, -0.3, 0, 0, 0, 1])
    pico._publish_tracking(hand if left else None, hand if right else None,
                           wrist if left else None, wrist if right else None,
                           None, None, None)


def test_atomic_snapshot_copied_once_and_inactive_side_cleared():
    pico = pico_without_thread()
    assert pico.latest_frame() is None
    publish(pico)
    frame = pico.latest_frame()
    assert frame.sequence == 1
    frame.hands['left'][:] = 123
    assert np.all(pico.latest_frame().hands['left'] == 1)
    assert pico.latest_frame(frame.sequence) is None
    publish(pico, left=False)
    frame2 = pico.latest_frame(frame.sequence)
    assert frame2.sequence == 2
    assert frame2.hands['left'] is None
    assert frame2.wrist_pose('left') is None
    assert frame2.wrist_pose('right') is not None
    assert pico.get_landmarks_mediapipe('left') is None


class FakeController:
    def __init__(self):
        self.poses = []
    def update_hand_from_mediapipe(self, pts):
        return pts is not None
    def update_from_pose(self, *pose):
        self.poses.append(pose)


def test_input_stream_never_processes_same_pico_frame_twice():
    stream = TeleopInputStream.__new__(TeleopInputStream)
    stream.avp_input = None
    stream.pico4_input = pico_without_thread()
    stream._pico_sequence = 0
    left, right = FakeController(), FakeController()
    publish(stream.pico4_input)
    first = stream.poll_arms(left, right, None)
    second = stream.poll_arms(left, right, None)
    assert first.saw_valid_data
    assert set(first.wrist_received_at) == {'left', 'right'}
    assert not second.saw_valid_data
    assert not second.wrist_received_at
    assert len(left.poses) == len(right.poses) == 1
    publish(stream.pico4_input, left=False)
    third = stream.poll_arms(left, right, None)
    assert third.invalid_wrist_sides == {'left'}
    assert set(third.wrist_received_at) == {'right'}
    assert len(left.poses) == 1
    assert len(right.poses) == 2


@pytest.mark.parametrize('bad', ['nan,0,0,0,0,0,1', '0,0,0,0,0,0,0', 'bad,0,0,0,0,0,1', '', None])
def test_parser_drops_corrupt_hand(bad):
    hand = dict(isActive=1, HandJointLocations=[{'p':'1,2,3,0,0,0,1'} for _ in range(26)])
    hand['HandJointLocations'][0]['p'] = bad
    assert _parse_hand_state(hand, 0) == (None, None, None)

@pytest.mark.parametrize('split', range(1, 40))
def test_relay_preserves_partial_header_and_payload(split):
    import struct
    from utils.adapters.input.pico4 import _RelayFrameParser
    device = b'pico-test-device'
    payload = b'{"Hand":{"leftHand":{"isActive":0}}}'
    frame = struct.pack('<I', len(device)) + device + struct.pack('<I', len(payload)) + payload
    parser = _RelayFrameParser()
    parser.feed(frame[:split])
    assert parser.try_parse() is None
    assert parser.try_parse() is None
    parser.feed(frame[split:] + frame)
    assert parser.try_parse() == payload
    assert parser.try_parse() == payload
    assert parser.try_parse() is None

"""AVP wrist-pose validity gating. No headset, no network: the stream is faked."""

import numpy as np
import pytest

from utils.adapters.input.avp import AVPInput


class _FakeHand:
    def __init__(self, wrist):
        self.wrist = wrist


class _FakeFrame:
    def __init__(self, wrist):
        self.left = _FakeHand(wrist)
        self.right = _FakeHand(wrist)


def _input_with(wrist) -> AVPInput:
    """An AVPInput holding one frame, without constructing a real streamer."""
    avp = AVPInput.__new__(AVPInput)
    avp._latest = None if wrist is None else _FakeFrame(wrist)
    return avp


def _matrix(position=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = position
    if rotation is not None:
        matrix[:3, :3] = rotation
    return matrix


@pytest.mark.parametrize("side", ["left", "right"])
def test_placeholder_identity_wrist_is_rejected(side: str) -> None:
    """avp_stream fills an identity matrix for a hand it has no data for.

    ``streamer.py`` does ``data.get("left_wrist", np.eye(4))``, which is not
    None and so passes every None check. Accepting it pins the WristTracker
    baseline to the headset origin, and the first genuinely tracked frame then
    arrives as a residual the size of the hand's absolute coordinates. With
    position_scale applied that is metres of commanded motion, which on
    hardware means the arms snap across the workspace.
    """
    assert _input_with(_matrix()).get_wrist_pose(side) is None


@pytest.mark.parametrize("side", ["left", "right"])
def test_tracked_wrist_is_accepted(side: str) -> None:
    pose = _input_with(_matrix((0.10, 0.30, 1.20))).get_wrist_pose(side)
    assert pose is not None
    position, quaternion = pose
    assert np.all(np.isfinite(position))
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    # Rejection must key on position: the origin is what breaks the baseline.
    assert np.linalg.norm(position) == pytest.approx(np.linalg.norm((0.10, 0.30, 1.20)))


def test_rotated_placeholder_at_origin_is_still_rejected() -> None:
    """A non-identity rotation does not make an origin position trackable."""
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _input_with(_matrix(rotation=rotation)).get_wrist_pose("left") is None


def test_wrist_just_off_the_origin_is_not_filtered() -> None:
    """The gate must not swallow a real hand that happens to sit near zero."""
    assert _input_with(_matrix((0.001, 0.0, 0.0))).get_wrist_pose("left") is not None


def test_missing_frame_and_missing_hand_return_none() -> None:
    assert _input_with(None).get_wrist_pose("left") is None

    avp = _input_with(_matrix((0.1, 0.2, 1.0)))
    avp._latest.left = None
    assert avp.get_wrist_pose("left") is None
    assert avp.get_wrist_pose("right") is not None

    avp = _input_with(_matrix((0.1, 0.2, 1.0)))
    avp._latest.left = _FakeHand(None)
    assert avp.get_wrist_pose("left") is None

"""Hand retargeting: Quest 3 landmarks -> dexterous hand joint angles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Unity LH (x right, y up, z forward) -> RH (x front, y left, z up)
_UNITY_TO_RH = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


def landmarks_to_mediapipe(raw_landmarks: list[float]) -> np.ndarray:
    """Convert 63 raw floats (Unity LH) to (21, 3) array in RH frame."""
    arr = np.array(raw_landmarks, dtype=np.float64).reshape(21, 3)
    return (_UNITY_TO_RH @ arr.T).T


def default_hand_config_path() -> Path:
    """Return default path to quest3_wuji_hand.yaml retarget config."""
    return (
        Path(__file__).resolve().parent.parent
        / "third_party"
        / "AnyDexRetarget"
        / "example"
        / "config"
        / "adaptive"
        / "quest3"
        / "quest3_wuji_hand.yaml"
    )


def default_inspire_config_path() -> Path:
    """Return default path to avp_inspire_hand.yaml retarget config."""
    return (
        Path(__file__).resolve().parent.parent
        / "third_party"
        / "AnyDexRetarget"
        / "example"
        / "config"
        / "adaptive"
        / "avp"
        / "avp_inspire_hand.yaml"
    )


def default_pico4_config_path(hand_type: str = "wuji") -> Path:
    """Return default Pico4 retarget config for supported dexterous hands."""
    config_name = {
        "wuji": "pico4_wuji_hand.yaml",
        "inspire": "pico4_inspire_hand.yaml",
    }.get(hand_type)
    if config_name is None:
        raise ValueError(f"No default Pico4 retarget config for hand type: {hand_type}")
    return (
        Path(__file__).resolve().parent.parent
        / "config"
        / "pico4"
        / config_name
    )


class HandRetargeter:
    """Wraps AnyDexRetarget: Quest 3 landmarks -> joint angles."""

    def __init__(self, config_path: str | Path | None = None, side: str = "right"):
        from anydexretarget import Retargeter

        if config_path is None:
            default_cfg = default_hand_config_path()
            if default_cfg.exists():
                config_path = str(default_cfg)
            else:
                print(f"Warning: default hand config not found at {default_cfg}")
                print("Hand retargeting will be disabled. Use --hand-config to specify.")
                self._retargeter = None
                return
        self._retargeter = Retargeter.from_yaml(str(config_path), side)
        print(f"Retargeter loaded from {config_path}")

    @property
    def available(self) -> bool:
        return self._retargeter is not None

    def retarget(self, raw_landmarks: list[float]) -> np.ndarray | None:
        """63 floats (Unity LH) -> joint angles, or None if invalid."""
        if self._retargeter is None:
            return None
        mediapipe_pts = landmarks_to_mediapipe(raw_landmarks)
        return self.retarget_mediapipe(mediapipe_pts)

    def retarget_mediapipe(self, mediapipe_pts: np.ndarray) -> np.ndarray | None:
        """(21, 3) MediaPipe landmarks -> joint angles, or None if invalid."""
        if self._retargeter is None:
            return None
        if np.allclose(mediapipe_pts, 0):
            return None
        return self._retargeter.retarget(mediapipe_pts)

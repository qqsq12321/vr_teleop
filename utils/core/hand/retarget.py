"""Hand retargeting: VR landmarks -> linker_l20 dexterous hand joint angles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Unity LH (x right, y up, z forward) -> RH (x front, y left, z up)
_UNITY_TO_RH = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)

_ANYDEX_CONFIG_DIR = Path(__file__).resolve().parents[3] / "third_party" / "AnyDexRetarget" / "example" / "config" / "adaptive"


def _anydex_hand_config_path(input_source: str, hand_type: str, side: str = "right") -> Path:
    source = input_source if input_source in {"quest3", "avp", "pico4"} else "quest3"
    if hand_type == "linker_l20" and source == "pico4" and side == "left":
        return _ANYDEX_CONFIG_DIR / "pico4" / "pico4_linker_l20_left.yaml"
    return _ANYDEX_CONFIG_DIR / source / f"{source}_{hand_type}.yaml"


def _anydex_linker_l20_config_path(input_source: str, side: str = "right") -> Path:
    return _anydex_hand_config_path(input_source, "linker_l20", side=side)


def landmarks_to_mediapipe(raw_landmarks: list[float]) -> np.ndarray:
    """Convert 63 raw floats (Unity LH) to (21, 3) array in RH frame."""
    arr = np.array(raw_landmarks, dtype=np.float64).reshape(21, 3)
    return (_UNITY_TO_RH @ arr.T).T


def default_pico4_config_path(side: str = "right") -> Path:
    """Return default Pico4 linker_l20 retarget config."""
    return _anydex_linker_l20_config_path("pico4", side=side)


def default_linker_l20_config_path(input_source: str = "quest3", side: str = "right") -> Path:
    """Return default linker_l20 retarget config for a given input source."""
    return _anydex_linker_l20_config_path(input_source, side=side)


def default_inspire_hand_config_path(input_source: str = "quest3", side: str = "right") -> Path:
    """Return default inspire_hand retarget config for a given input source."""
    return _anydex_hand_config_path(input_source, "inspire_hand", side=side)


class HandRetargeter:
    """Wraps AnyDexRetarget: VR landmarks -> linker_l20 joint angles."""

    def __init__(self, config_path: str | Path | None = None, side: str = "right"):
        from anydexretarget import Retargeter

        if config_path is None:
            config_path = default_linker_l20_config_path("quest3", side=side)
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: hand config not found at {config_path}")
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

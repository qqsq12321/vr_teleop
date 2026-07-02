"""Output backends for simulation and real hardware."""

from utils.adapters.output.inspire_real import (
    InspireSerialOutput,
    encode_inspire_channels,
    retarget_qpos_to_channels,
)
from utils.adapters.output.galbot_real import GalbotRealRobotOutput

__all__ = [
    "GalbotRealRobotOutput",
    "InspireSerialOutput",
    "encode_inspire_channels",
    "retarget_qpos_to_channels",
]

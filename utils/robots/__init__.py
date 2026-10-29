"""Robot specifications."""

from utils.robots.cvte_carm_d3 import CVTECArmD3Spec
from utils.robots.dexforce import DexForceSpec
from utils.robots.galbot import GalbotInspireSpec
from utils.robots.realman_rm75b import RealManRM75BInspireSpec
from utils.robots.realman_rm75b_l20 import RealManRM75BL20Spec
from utils.robots.realman_rm75b_single import RealManRM75BSingleSpec

__all__ = [
    "CVTECArmD3Spec",
    "DexForceSpec",
    "GalbotInspireSpec",
    "RealManRM75BInspireSpec",
    "RealManRM75BL20Spec",
    "RealManRM75BSingleSpec",
]

"""Fusion-based GHZ state generation for hybrid GHZ-BSM entanglement routing.

This module implements the GHZ state generation approach described in Chen et al.,
"Routing Entanglement in Complex Quantum Networks Using GHZ States", arXiv:2604.03155 (2026),
using the fusion ideas of Bartolucci et al., Nature Communications 14, 912 (2023).
"""

from .ghz_generation import (
    GHZGenerationA,
    GHZEntanglementGenerationA,
    GHZNode,
    GHZMessage,
    GHZMsgType,
    MAX_DEGREE,
    DEFAULT_SUCCESS_BASE,
)
from .ghz_rules import install_ghz_eg_rules

__all__ = [
    "GHZGenerationA",
    "GHZEntanglementGenerationA",
    "GHZNode",
    "GHZMessage",
    "GHZMsgType",
    "MAX_DEGREE",
    "DEFAULT_SUCCESS_BASE",
    "install_ghz_eg_rules",
]
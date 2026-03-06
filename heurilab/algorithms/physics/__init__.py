"""
Physics-based Algorithms
────────────────────────
GSA  — Gravitational Search Algorithm
MVO  — Multi-Verse Optimizer
SCA  — Sine Cosine Algorithm
AOA  — Arithmetic Optimization Algorithm
"""

from heurilab.algorithms.physics.gsa import GSA
from heurilab.algorithms.physics.mvo import MVO
from heurilab.algorithms.physics.sca import SCA
from heurilab.algorithms.physics.aoa import AOA

__all__ = ["GSA", "MVO", "SCA", "AOA"]

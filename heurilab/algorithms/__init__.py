"""
Built-in metaheuristic algorithms — 20 algorithms in 5 categories.

Swarm Intelligence:   PSO, GWO, WOA, MFO, SSA, HHO, MPA
Evolutionary:         GA, DE, ES, EP
Physics-based:        GSA, MVO, SCA, AOA
Human/Social:         TLBO, JA
Bio-inspired:         ABC, FA, SOS
"""

from heurilab.algorithms.base import _Base

from heurilab.algorithms.swarm import PSO, GWO, WOA, MFO, SSA, HHO, MPA
from heurilab.algorithms.evolutionary import GA, DE, ES, EP
from heurilab.algorithms.physics import GSA, MVO, SCA, AOA
from heurilab.algorithms.human import TLBO, JA
from heurilab.algorithms.bio import ABC, FA, SOS

__all__ = [
    "_Base",
    # Swarm Intelligence
    "PSO", "GWO", "WOA", "MFO", "SSA", "HHO", "MPA",
    # Evolutionary
    "GA", "DE", "ES", "EP",
    # Physics-based
    "GSA", "MVO", "SCA", "AOA",
    # Human/Social
    "TLBO", "JA",
    # Bio-inspired
    "ABC", "FA", "SOS",
]

# Convenience: all algorithms as (name, class) tuples grouped by category
SWARM_ALGORITHMS = [
    ("PSO", PSO), ("GWO", GWO), ("WOA", WOA), ("MFO", MFO),
    ("SSA", SSA), ("HHO", HHO), ("MPA", MPA),
]

EVOLUTIONARY_ALGORITHMS = [
    ("GA", GA), ("DE", DE), ("ES", ES), ("EP", EP),
]

PHYSICS_ALGORITHMS = [
    ("GSA", GSA), ("MVO", MVO), ("SCA", SCA), ("AOA", AOA),
]

HUMAN_ALGORITHMS = [
    ("TLBO", TLBO), ("JA", JA),
]

BIO_ALGORITHMS = [
    ("ABC", ABC), ("FA", FA), ("SOS", SOS),
]

ALL_ALGORITHMS = (
    SWARM_ALGORITHMS + EVOLUTIONARY_ALGORITHMS +
    PHYSICS_ALGORITHMS + HUMAN_ALGORITHMS + BIO_ALGORITHMS
)

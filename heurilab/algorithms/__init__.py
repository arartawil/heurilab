"""
Built-in metaheuristic algorithms — 40 algorithms in 5 categories.

Swarm Intelligence:   PSO, GWO, WOA, MFO, SSA, HHO, MPA, BA, CS, FPA, DA, GOA
Evolutionary:         GA, DE, ES, EP, CMA, BBO, SHADE
Physics-based:        GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO
Human/Social:         TLBO, JA, HS, ICA, CA, BSO
Bio-inspired:         ABC, FA, SOS, BFO, CSA, BOA, TSA
"""

from heurilab.algorithms.base import _Base

from heurilab.algorithms.swarm import PSO, GWO, WOA, MFO, SSA, HHO, MPA, BA, CS, FPA, DA, GOA
from heurilab.algorithms.evolutionary import GA, DE, ES, EP, CMA, BBO, SHADE
from heurilab.algorithms.physics import GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO
from heurilab.algorithms.human import TLBO, JA, HS, ICA, CA, BSO
from heurilab.algorithms.bio import ABC, FA, SOS, BFO, CSA, BOA, TSA

__all__ = [
    "_Base",
    # Swarm Intelligence
    "PSO", "GWO", "WOA", "MFO", "SSA", "HHO", "MPA",
    "BA", "CS", "FPA", "DA", "GOA",
    # Evolutionary
    "GA", "DE", "ES", "EP", "CMA", "BBO", "SHADE",
    # Physics-based
    "GSA", "MVO", "SCA", "AOA", "SA", "EO", "WDO", "HGSO",
    # Human/Social
    "TLBO", "JA", "HS", "ICA", "CA", "BSO",
    # Bio-inspired
    "ABC", "FA", "SOS", "BFO", "CSA", "BOA", "TSA",
]

# Convenience: all algorithms as (name, class) tuples grouped by category
SWARM_ALGORITHMS = [
    ("PSO", PSO), ("GWO", GWO), ("WOA", WOA), ("MFO", MFO),
    ("SSA", SSA), ("HHO", HHO), ("MPA", MPA),
    ("BA", BA), ("CS", CS), ("FPA", FPA), ("DA", DA), ("GOA", GOA),
]

EVOLUTIONARY_ALGORITHMS = [
    ("GA", GA), ("DE", DE), ("ES", ES), ("EP", EP),
    ("CMA", CMA), ("BBO", BBO), ("SHADE", SHADE),
]

PHYSICS_ALGORITHMS = [
    ("GSA", GSA), ("MVO", MVO), ("SCA", SCA), ("AOA", AOA),
    ("SA", SA), ("EO", EO), ("WDO", WDO), ("HGSO", HGSO),
]

HUMAN_ALGORITHMS = [
    ("TLBO", TLBO), ("JA", JA),
    ("HS", HS), ("ICA", ICA), ("CA", CA), ("BSO", BSO),
]

BIO_ALGORITHMS = [
    ("ABC", ABC), ("FA", FA), ("SOS", SOS),
    ("BFO", BFO), ("CSA", CSA), ("BOA", BOA), ("TSA", TSA),
]

ALL_ALGORITHMS = (
    SWARM_ALGORITHMS + EVOLUTIONARY_ALGORITHMS +
    PHYSICS_ALGORITHMS + HUMAN_ALGORITHMS + BIO_ALGORITHMS
)

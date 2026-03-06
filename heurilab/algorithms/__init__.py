"""
Built-in metaheuristic algorithms — 60 algorithms in 5 categories.

Swarm Intelligence:   PSO, GWO, WOA, MFO, SSA, HHO, MPA, BA, CS, FPA, DA, GOA,
                      ALO, SHO, DO, EHO
Evolutionary:         GA, DE, ES, EP, CMA, BBO, SHADE, TLGO, CoDE, SaDE, OXDE
Physics-based:        GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO, CSS, CFO, TWO, ASO
Human/Social:         TLBO, JA, HS, ICA, CA, BSO, SOS_H, QLA, INFO, HBO
Bio-inspired:         ABC, FA, SOS, BFO, CSA, BOA, TSA, WHO, SBO, MBO, EPO
"""

from heurilab.algorithms.base import _Base

from heurilab.algorithms.swarm import (PSO, GWO, WOA, MFO, SSA, HHO, MPA,
                                       BA, CS, FPA, DA, GOA,
                                       ALO, SHO, DO, EHO)
from heurilab.algorithms.evolutionary import (GA, DE, ES, EP, CMA, BBO, SHADE,
                                              TLGO, CoDE, SaDE, OXDE)
from heurilab.algorithms.physics import (GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO,
                                         CSS, CFO, TWO, ASO)
from heurilab.algorithms.human import (TLBO, JA, HS, ICA, CA, BSO,
                                       SOS_H, QLA, INFO, HBO)
from heurilab.algorithms.bio import (ABC, FA, SOS, BFO, CSA, BOA, TSA,
                                     WHO, SBO, MBO, EPO)

__all__ = [
    "_Base",
    # Swarm Intelligence
    "PSO", "GWO", "WOA", "MFO", "SSA", "HHO", "MPA",
    "BA", "CS", "FPA", "DA", "GOA",
    "ALO", "SHO", "DO", "EHO",
    # Evolutionary
    "GA", "DE", "ES", "EP", "CMA", "BBO", "SHADE",
    "TLGO", "CoDE", "SaDE", "OXDE",
    # Physics-based
    "GSA", "MVO", "SCA", "AOA", "SA", "EO", "WDO", "HGSO",
    "CSS", "CFO", "TWO", "ASO",
    # Human/Social
    "TLBO", "JA", "HS", "ICA", "CA", "BSO",
    "SOS_H", "QLA", "INFO", "HBO",
    # Bio-inspired
    "ABC", "FA", "SOS", "BFO", "CSA", "BOA", "TSA",
    "WHO", "SBO", "MBO", "EPO",
]

# Convenience: all algorithms as (name, class) tuples grouped by category
SWARM_ALGORITHMS = [
    ("PSO", PSO), ("GWO", GWO), ("WOA", WOA), ("MFO", MFO),
    ("SSA", SSA), ("HHO", HHO), ("MPA", MPA),
    ("BA", BA), ("CS", CS), ("FPA", FPA), ("DA", DA), ("GOA", GOA),
    ("ALO", ALO), ("SHO", SHO), ("DO", DO), ("EHO", EHO),
]

EVOLUTIONARY_ALGORITHMS = [
    ("GA", GA), ("DE", DE), ("ES", ES), ("EP", EP),
    ("CMA", CMA), ("BBO", BBO), ("SHADE", SHADE),
    ("TLGO", TLGO), ("CoDE", CoDE), ("SaDE", SaDE), ("OXDE", OXDE),
]

PHYSICS_ALGORITHMS = [
    ("GSA", GSA), ("MVO", MVO), ("SCA", SCA), ("AOA", AOA),
    ("SA", SA), ("EO", EO), ("WDO", WDO), ("HGSO", HGSO),
    ("CSS", CSS), ("CFO", CFO), ("TWO", TWO), ("ASO", ASO),
]

HUMAN_ALGORITHMS = [
    ("TLBO", TLBO), ("JA", JA),
    ("HS", HS), ("ICA", ICA), ("CA", CA), ("BSO", BSO),
    ("SOS_H", SOS_H), ("QLA", QLA), ("INFO", INFO), ("HBO", HBO),
]

BIO_ALGORITHMS = [
    ("ABC", ABC), ("FA", FA), ("SOS", SOS),
    ("BFO", BFO), ("CSA", CSA), ("BOA", BOA), ("TSA", TSA),
    ("WHO", WHO), ("SBO", SBO), ("MBO", MBO), ("EPO", EPO),
]

ALL_ALGORITHMS = (
    SWARM_ALGORITHMS + EVOLUTIONARY_ALGORITHMS +
    PHYSICS_ALGORITHMS + HUMAN_ALGORITHMS + BIO_ALGORITHMS
)

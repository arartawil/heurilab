"""
Built-in metaheuristic algorithms — 100 algorithms in 6 categories.

Swarm Intelligence:   PSO, GWO, WOA, MFO, SSA, HHO, MPA, BA, CS, FPA, DA, GOA,
                      ALO, SHO, DO, EHO, AO, HGS, GTO, RUN
Evolutionary:         GA, DE, ES, EP, CMA, BBO, SHADE, TLGO, CoDE, SaDE, OXDE,
                      AGDE, LSHADE, EBOwithCMAR, IMODE
Physics-based:        GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO, CSS, CFO, TWO, ASO,
                      RIME, AEO, GBO, TSO
Human/Social:         TLBO, JA, HS, ICA, CA, BSO, SOS_H, QLA, INFO, HBO,
                      AOArch, CHIO, SSOA, POA
Bio-inspired:         ABC, FA, SOS, BFO, CSA, BOA, TSA, WHO, SBO, MBO, EPO,
                      SMA, HBA, RSA, GJO
Modern (2022–2025):   AVOA, DMO, MGO, DBO, COA, OOA, NOA, SAO, FLA, EVO,
                      EDO, MOA, CPO, PO, FO, HO, KOA, SBOA, GMO, FFO
"""

from heurilab.algorithms.base import _Base

from heurilab.algorithms.swarm import (PSO, GWO, WOA, MFO, SSA, HHO, MPA,
                                       BA, CS, FPA, DA, GOA,
                                       ALO, SHO, DO, EHO,
                                       AO, HGS, GTO, RUN)
from heurilab.algorithms.evolutionary import (GA, DE, ES, EP, CMA, BBO, SHADE,
                                              TLGO, CoDE, SaDE, OXDE,
                                              AGDE, LSHADE, EBOwithCMAR, IMODE)
from heurilab.algorithms.physics import (GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO,
                                         CSS, CFO, TWO, ASO,
                                         RIME, AEO, GBO, TSO)
from heurilab.algorithms.human import (TLBO, JA, HS, ICA, CA, BSO,
                                       SOS_H, QLA, INFO, HBO,
                                       AOArch, CHIO, SSOA, POA)
from heurilab.algorithms.bio import (ABC, FA, SOS, BFO, CSA, BOA, TSA,
                                     WHO, SBO, MBO, EPO,
                                     SMA, HBA, RSA, GJO)
from heurilab.algorithms.modern import (AVOA, DMO, MGO,
                                        DBO, COA, OOA, NOA, SAO, FLA, EVO, EDO, MOA,
                                        CPO, PO, FO, HO, KOA, SBOA, GMO, FFO)

__all__ = [
    "_Base",
    # Swarm Intelligence
    "PSO", "GWO", "WOA", "MFO", "SSA", "HHO", "MPA",
    "BA", "CS", "FPA", "DA", "GOA",
    "ALO", "SHO", "DO", "EHO",
    "AO", "HGS", "GTO", "RUN",
    # Evolutionary
    "GA", "DE", "ES", "EP", "CMA", "BBO", "SHADE",
    "TLGO", "CoDE", "SaDE", "OXDE",
    "AGDE", "LSHADE", "EBOwithCMAR", "IMODE",
    # Physics-based
    "GSA", "MVO", "SCA", "AOA", "SA", "EO", "WDO", "HGSO",
    "CSS", "CFO", "TWO", "ASO",
    "RIME", "AEO", "GBO", "TSO",
    # Human/Social
    "TLBO", "JA", "HS", "ICA", "CA", "BSO",
    "SOS_H", "QLA", "INFO", "HBO",
    "AOArch", "CHIO", "SSOA", "POA",
    # Bio-inspired
    "ABC", "FA", "SOS", "BFO", "CSA", "BOA", "TSA",
    "WHO", "SBO", "MBO", "EPO",
    "SMA", "HBA", "RSA", "GJO",
    # Modern (2022–2025)
    "AVOA", "DMO", "MGO",
    "DBO", "COA", "OOA", "NOA", "SAO", "FLA", "EVO", "EDO", "MOA",
    "CPO", "PO", "FO", "HO", "KOA", "SBOA", "GMO", "FFO",
]

# Convenience: all algorithms as (name, class) tuples grouped by category
SWARM_ALGORITHMS = [
    ("PSO", PSO), ("GWO", GWO), ("WOA", WOA), ("MFO", MFO),
    ("SSA", SSA), ("HHO", HHO), ("MPA", MPA),
    ("BA", BA), ("CS", CS), ("FPA", FPA), ("DA", DA), ("GOA", GOA),
    ("ALO", ALO), ("SHO", SHO), ("DO", DO), ("EHO", EHO),
    ("AO", AO), ("HGS", HGS), ("GTO", GTO), ("RUN", RUN),
]

EVOLUTIONARY_ALGORITHMS = [
    ("GA", GA), ("DE", DE), ("ES", ES), ("EP", EP),
    ("CMA", CMA), ("BBO", BBO), ("SHADE", SHADE),
    ("TLGO", TLGO), ("CoDE", CoDE), ("SaDE", SaDE), ("OXDE", OXDE),
    ("AGDE", AGDE), ("LSHADE", LSHADE), ("EBOwithCMAR", EBOwithCMAR), ("IMODE", IMODE),
]

PHYSICS_ALGORITHMS = [
    ("GSA", GSA), ("MVO", MVO), ("SCA", SCA), ("AOA", AOA),
    ("SA", SA), ("EO", EO), ("WDO", WDO), ("HGSO", HGSO),
    ("CSS", CSS), ("CFO", CFO), ("TWO", TWO), ("ASO", ASO),
    ("RIME", RIME), ("AEO", AEO), ("GBO", GBO), ("TSO", TSO),
]

HUMAN_ALGORITHMS = [
    ("TLBO", TLBO), ("JA", JA),
    ("HS", HS), ("ICA", ICA), ("CA", CA), ("BSO", BSO),
    ("SOS_H", SOS_H), ("QLA", QLA), ("INFO", INFO), ("HBO", HBO),
    ("AOArch", AOArch), ("CHIO", CHIO), ("SSOA", SSOA), ("POA", POA),
]

BIO_ALGORITHMS = [
    ("ABC", ABC), ("FA", FA), ("SOS", SOS),
    ("BFO", BFO), ("CSA", CSA), ("BOA", BOA), ("TSA", TSA),
    ("WHO", WHO), ("SBO", SBO), ("MBO", MBO), ("EPO", EPO),
    ("SMA", SMA), ("HBA", HBA), ("RSA", RSA), ("GJO", GJO),
]

MODERN_ALGORITHMS = [
    ("AVOA", AVOA), ("DMO", DMO), ("MGO", MGO),
    ("DBO", DBO), ("COA", COA), ("OOA", OOA), ("NOA", NOA),
    ("SAO", SAO), ("FLA", FLA), ("EVO", EVO), ("EDO", EDO), ("MOA", MOA),
    ("CPO", CPO), ("PO", PO), ("FO", FO), ("HO", HO),
    ("KOA", KOA), ("SBOA", SBOA), ("GMO", GMO), ("FFO", FFO),
]

ALL_ALGORITHMS = (
    SWARM_ALGORITHMS + EVOLUTIONARY_ALGORITHMS +
    PHYSICS_ALGORITHMS + HUMAN_ALGORITHMS + BIO_ALGORITHMS +
    MODERN_ALGORITHMS
)

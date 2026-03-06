"""
Physics-based Algorithms
────────────────────────
GSA  — Gravitational Search Algorithm
MVO  — Multi-Verse Optimizer
SCA  — Sine Cosine Algorithm
AOA  — Arithmetic Optimization Algorithm
SA   — Simulated Annealing
EO   — Equilibrium Optimizer
WDO  — Wind Driven Optimization
HGSO — Henry Gas Solubility Optimization
CSS  — Charged System Search
CFO  — Central Force Optimization
TWO  — Tug of War Optimization
ASO  — Atom Search Optimization
"""

from heurilab.algorithms.physics.gsa import GSA
from heurilab.algorithms.physics.mvo import MVO
from heurilab.algorithms.physics.sca import SCA
from heurilab.algorithms.physics.aoa import AOA
from heurilab.algorithms.physics.sa import SA
from heurilab.algorithms.physics.eo import EO
from heurilab.algorithms.physics.wdo import WDO
from heurilab.algorithms.physics.hgso import HGSO
from heurilab.algorithms.physics.css import CSS
from heurilab.algorithms.physics.cfo import CFO
from heurilab.algorithms.physics.two import TWO
from heurilab.algorithms.physics.aso import ASO

__all__ = ["GSA", "MVO", "SCA", "AOA", "SA", "EO", "WDO", "HGSO",
           "CSS", "CFO", "TWO", "ASO"]

"""
Evolutionary Algorithms
───────────────────────
GA    — Genetic Algorithm
DE    — Differential Evolution
ES    — Evolution Strategy (mu,lambda)
EP    — Evolutionary Programming
CMA   — Covariance Matrix Adaptation ES (CMA-ES)
BBO   — Biogeography-Based Optimization
SHADE — Success-History based Adaptive DE
TLGO  — Teaching-Learning-based Genetic Optimization
CoDE  — Composite Differential Evolution
SaDE  — Self-adaptive Differential Evolution
OXDE  — Opposition-based Learning DE
"""

from heurilab.algorithms.evolutionary.ga import GA
from heurilab.algorithms.evolutionary.de import DE
from heurilab.algorithms.evolutionary.es import ES
from heurilab.algorithms.evolutionary.ep import EP
from heurilab.algorithms.evolutionary.cma import CMA
from heurilab.algorithms.evolutionary.bbo import BBO
from heurilab.algorithms.evolutionary.shade import SHADE
from heurilab.algorithms.evolutionary.tlgo import TLGO
from heurilab.algorithms.evolutionary.code import CoDE
from heurilab.algorithms.evolutionary.sade import SaDE
from heurilab.algorithms.evolutionary.oxde import OXDE

__all__ = ["GA", "DE", "ES", "EP", "CMA", "BBO", "SHADE",
           "TLGO", "CoDE", "SaDE", "OXDE"]

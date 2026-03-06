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
"""

from heurilab.algorithms.evolutionary.ga import GA
from heurilab.algorithms.evolutionary.de import DE
from heurilab.algorithms.evolutionary.es import ES
from heurilab.algorithms.evolutionary.ep import EP
from heurilab.algorithms.evolutionary.cma import CMA
from heurilab.algorithms.evolutionary.bbo import BBO
from heurilab.algorithms.evolutionary.shade import SHADE

__all__ = ["GA", "DE", "ES", "EP", "CMA", "BBO", "SHADE"]

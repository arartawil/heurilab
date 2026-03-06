"""
Evolutionary Algorithms
───────────────────────
GA  — Genetic Algorithm
DE  — Differential Evolution
ES  — Evolution Strategy (mu,lambda)
EP  — Evolutionary Programming
"""

from heurilab.algorithms.evolutionary.ga import GA
from heurilab.algorithms.evolutionary.de import DE
from heurilab.algorithms.evolutionary.es import ES
from heurilab.algorithms.evolutionary.ep import EP

__all__ = ["GA", "DE", "ES", "EP"]

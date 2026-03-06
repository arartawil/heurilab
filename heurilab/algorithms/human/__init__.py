"""
Human / Social-based Algorithms
────────────────────────────────
TLBO — Teaching-Learning-Based Optimization
JA   — Jaya Algorithm
HS   — Harmony Search
ICA  — Imperialist Competitive Algorithm
CA   — Cultural Algorithm
BSO  — Brain Storm Optimization
"""

from heurilab.algorithms.human.tlbo import TLBO
from heurilab.algorithms.human.ja import JA
from heurilab.algorithms.human.hs import HS
from heurilab.algorithms.human.ica import ICA
from heurilab.algorithms.human.ca import CA
from heurilab.algorithms.human.bso import BSO

__all__ = ["TLBO", "JA", "HS", "ICA", "CA", "BSO"]

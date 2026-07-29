"""
Human / Social-based Algorithms
────────────────────────────────
TLBO — Teaching-Learning-Based Optimization
JA   — Jaya Algorithm
HS   — Harmony Search
ICA  — Imperialist Competitive Algorithm
CA    — Cultural Algorithm
BSO   — Brain Storm Optimization
INFO   — Weighted Mean of Vectors
HBO    — Heap-Based Optimizer
AOArch — Archimedes Optimization Algorithm
CHIO   — Coronavirus Herd Immunity Optimizer
SSOA   — Sparrow Search Optimization Algorithm
POA    — Political Optimizer Algorithm
ED     — Enterprise Development-inspired metaheuristic
"""

from heurilab.algorithms.human.tlbo import TLBO
from heurilab.algorithms.human.ja import JA
from heurilab.algorithms.human.hs import HS
from heurilab.algorithms.human.ica import ICA
from heurilab.algorithms.human.ca import CA
from heurilab.algorithms.human.bso import BSO
from heurilab.algorithms.human.info import INFO
from heurilab.algorithms.human.hbo import HBO
from heurilab.algorithms.human.aoarch import AOArch
from heurilab.algorithms.human.chio import CHIO
from heurilab.algorithms.human.ssoa import SSOA
from heurilab.algorithms.human.poa import POA
from heurilab.algorithms.human.ed import ED

__all__ = ["TLBO", "JA", "HS", "ICA", "CA", "BSO",
           "INFO", "HBO",
           "AOArch", "CHIO", "SSOA", "POA", "ED"]

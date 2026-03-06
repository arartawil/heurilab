"""
Human / Social-based Algorithms
────────────────────────────────
TLBO — Teaching-Learning-Based Optimization
JA   — Jaya Algorithm
HS   — Harmony Search
ICA  — Imperialist Competitive Algorithm
CA    — Cultural Algorithm
BSO   — Brain Storm Optimization
SOS_H — Social Optimization Search
QLA   — Q-Learning-based Algorithm
INFO   — Weighted Mean of Vectors
HBO    — Heap-Based Optimizer
AOArch — Archimedes Optimization Algorithm
CHIO   — Coronavirus Herd Immunity Optimizer
SSOA   — Sparrow Search Optimization Algorithm
POA    — Political Optimizer Algorithm
"""

from heurilab.algorithms.human.tlbo import TLBO
from heurilab.algorithms.human.ja import JA
from heurilab.algorithms.human.hs import HS
from heurilab.algorithms.human.ica import ICA
from heurilab.algorithms.human.ca import CA
from heurilab.algorithms.human.bso import BSO
from heurilab.algorithms.human.sos_h import SOS_H
from heurilab.algorithms.human.qla import QLA
from heurilab.algorithms.human.info import INFO
from heurilab.algorithms.human.hbo import HBO
from heurilab.algorithms.human.aoarch import AOArch
from heurilab.algorithms.human.chio import CHIO
from heurilab.algorithms.human.ssoa import SSOA
from heurilab.algorithms.human.poa import POA

__all__ = ["TLBO", "JA", "HS", "ICA", "CA", "BSO",
           "SOS_H", "QLA", "INFO", "HBO",
           "AOArch", "CHIO", "SSOA", "POA"]

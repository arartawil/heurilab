"""
Bio-inspired Algorithms
───────────────────────
ABC — Artificial Bee Colony
FA  — Firefly Algorithm
SOS — Symbiotic Organisms Search
BFO — Bacterial Foraging Optimization
CSA — Crow Search Algorithm
BOA — Butterfly Optimization Algorithm
TSA — Tree Seed Algorithm
WHO — Wildebeest Herd Optimization
SBO — Satin Bowerbird Optimizer
MBO — Monarch Butterfly Optimization
EPO — Emperor Penguin Optimizer
SMA — Slime Mould Algorithm
HBA — Honey Badger Algorithm
RSA — Reptile Search Algorithm
GJO — Golden Jackal Optimization
"""

from heurilab.algorithms.bio.abc import ABC
from heurilab.algorithms.bio.fa import FA
from heurilab.algorithms.bio.sos import SOS
from heurilab.algorithms.bio.bfo import BFO
from heurilab.algorithms.bio.csa import CSA
from heurilab.algorithms.bio.boa import BOA
from heurilab.algorithms.bio.tsa import TSA
from heurilab.algorithms.bio.who import WHO
from heurilab.algorithms.bio.sbo import SBO
from heurilab.algorithms.bio.mbo import MBO
from heurilab.algorithms.bio.epo import EPO
from heurilab.algorithms.bio.sma import SMA
from heurilab.algorithms.bio.hba import HBA
from heurilab.algorithms.bio.rsa import RSA
from heurilab.algorithms.bio.gjo import GJO

__all__ = ["ABC", "FA", "SOS", "BFO", "CSA", "BOA", "TSA",
           "WHO", "SBO", "MBO", "EPO",
           "SMA", "HBA", "RSA", "GJO"]

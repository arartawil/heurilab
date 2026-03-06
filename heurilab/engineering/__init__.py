"""Engineering design problems subpackage."""

from heurilab.engineering.problems import (
    pressure_vessel,
    welded_beam,
    tension_compression_spring,
    ENGINEERING_PROBLEMS,
)
from heurilab.engineering.runner import run_engineering_problems

__all__ = [
    "pressure_vessel",
    "welded_beam",
    "tension_compression_spring",
    "ENGINEERING_PROBLEMS",
    "run_engineering_problems",
]

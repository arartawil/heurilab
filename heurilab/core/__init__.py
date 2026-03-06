"""
Core — experiment runner, benchmark configuration, and classic test functions.
"""

from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite
from heurilab.core.runner import run_experiment
from heurilab.core.functions import (
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
    F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
    F21, F22, F23,
    CLASSICAL_FUNCTIONS,
    get_classical_suite, get_unimodal_suite,
    get_multimodal_suite, get_fixeddim_suite,
)

__all__ = [
    "BenchmarkConfig", "BenchmarkSuite", "run_experiment",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7",
    "F8", "F9", "F10", "F11", "F12", "F13",
    "F14", "F15", "F16", "F17", "F18", "F19", "F20",
    "F21", "F22", "F23",
    "CLASSICAL_FUNCTIONS",
    "get_classical_suite", "get_unimodal_suite",
    "get_multimodal_suite", "get_fixeddim_suite",
]

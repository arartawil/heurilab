"""
HeuriLab — Metaheuristic Experiment Infrastructure
====================================================
Automated runner, CSV outputs, convergence/box plots, and statistical Excel analysis
for metaheuristic optimization research.
"""

from heurilab.core.runner import run_experiment
from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite
from heurilab.core.functions import (
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
    F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
    F21, F22, F23,
    CLASSICAL_FUNCTIONS,
    get_classical_suite, get_unimodal_suite,
    get_multimodal_suite, get_fixeddim_suite,
)
from heurilab.core.cec2017 import (
    CEC17_F1, CEC17_F3,
    CEC17_F4, CEC17_F5, CEC17_F6, CEC17_F7, CEC17_F8, CEC17_F9, CEC17_F10,
    CEC17_F11, CEC17_F12, CEC17_F13, CEC17_F14, CEC17_F15,
    CEC17_F16, CEC17_F17, CEC17_F18, CEC17_F19, CEC17_F20,
    CEC17_F21, CEC17_F22, CEC17_F23, CEC17_F24, CEC17_F25,
    CEC17_F26, CEC17_F27, CEC17_F28, CEC17_F29, CEC17_F30,
    CEC2017_FUNCTIONS,
    get_cec2017_suite, get_cec2017_unimodal_suite,
    get_cec2017_multimodal_suite, get_cec2017_hybrid_suite,
    get_cec2017_composition_suite,
)
from heurilab.analyzer import enhance
from heurilab.analyzer import cec2017_benchmarks

__version__ = "2.0.1"
__all__ = [
    "run_experiment", "BenchmarkConfig", "BenchmarkSuite",
    # Classical F1–F23
    "F1", "F2", "F3", "F4", "F5", "F6", "F7",
    "F8", "F9", "F10", "F11", "F12", "F13",
    "F14", "F15", "F16", "F17", "F18", "F19", "F20",
    "F21", "F22", "F23",
    "CLASSICAL_FUNCTIONS",
    "get_classical_suite", "get_unimodal_suite",
    "get_multimodal_suite", "get_fixeddim_suite",
    # CEC 2017
    "CEC17_F1", "CEC17_F3",
    "CEC17_F4", "CEC17_F5", "CEC17_F6", "CEC17_F7", "CEC17_F8", "CEC17_F9", "CEC17_F10",
    "CEC17_F11", "CEC17_F12", "CEC17_F13", "CEC17_F14", "CEC17_F15",
    "CEC17_F16", "CEC17_F17", "CEC17_F18", "CEC17_F19", "CEC17_F20",
    "CEC17_F21", "CEC17_F22", "CEC17_F23", "CEC17_F24", "CEC17_F25",
    "CEC17_F26", "CEC17_F27", "CEC17_F28", "CEC17_F29", "CEC17_F30",
    "CEC2017_FUNCTIONS",
    "get_cec2017_suite", "get_cec2017_unimodal_suite",
    "get_cec2017_multimodal_suite", "get_cec2017_hybrid_suite",
    "get_cec2017_composition_suite",
    # Analyzer
    "enhance",
]

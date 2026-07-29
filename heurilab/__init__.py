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
from heurilab.core.cec2020 import (
    CEC20_F1, CEC20_F2, CEC20_F3, CEC20_F4, CEC20_F5,
    CEC20_F6, CEC20_F7, CEC20_F8, CEC20_F9, CEC20_F10,
    CEC2020_FUNCTIONS,
    get_cec2020_suite, get_cec2020_unimodal_suite,
    get_cec2020_multimodal_suite, get_cec2020_hybrid_suite,
    get_cec2020_composition_suite,
)
from heurilab.core.budget import (
    budget_report, calibrate_all, calibrate_iterations,
    measure_evals_per_iteration,
)
from heurilab.core.opfunu_suites import (
    OPFUNU_YEARS, get_opfunu_suite, list_opfunu_functions, get_opfunu_optimum,
    get_cec2014_opfunu_suite, get_cec2017_opfunu_suite, get_cec2020_opfunu_suite,
    get_cec2021_opfunu_suite, get_cec2022_opfunu_suite,
)
from heurilab.engineering.problems import (
    EngineeringProblem, PROBLEMS as ENGINEERING_PROBLEM_SET,
    get_engineering_problems,
)
from heurilab.analyzer import enhance
from heurilab.analyzer import cec2017_benchmarks

__version__ = "2.3.0"
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
    # CEC 2020
    "CEC20_F1", "CEC20_F2", "CEC20_F3", "CEC20_F4", "CEC20_F5",
    "CEC20_F6", "CEC20_F7", "CEC20_F8", "CEC20_F9", "CEC20_F10",
    "CEC2020_FUNCTIONS",
    "get_cec2020_suite", "get_cec2020_unimodal_suite",
    "get_cec2020_multimodal_suite", "get_cec2020_hybrid_suite",
    "get_cec2020_composition_suite",
    # Function-evaluation budgets
    "calibrate_iterations", "calibrate_all", "budget_report",
    "measure_evals_per_iteration",
    # CEC via opfunu (optional dependency)
    "OPFUNU_YEARS", "get_opfunu_suite", "list_opfunu_functions", "get_opfunu_optimum",
    "get_cec2014_opfunu_suite", "get_cec2017_opfunu_suite", "get_cec2020_opfunu_suite",
    "get_cec2021_opfunu_suite", "get_cec2022_opfunu_suite",
    # Engineering design problems
    "EngineeringProblem", "ENGINEERING_PROBLEM_SET", "get_engineering_problems",
    # Analyzer
    "enhance",
]

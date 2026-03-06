"""
HeuriLab — Metaheuristic Experiment Infrastructure
====================================================
Automated runner, CSV outputs, convergence/box plots, and statistical Excel analysis
for metaheuristic optimization research.
"""

from heurilab.core.runner import run_experiment
from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite

__version__ = "1.0.0"
__all__ = ["run_experiment", "BenchmarkConfig", "BenchmarkSuite"]

"""
Example usage of the heurilab package with all 20 built-in algorithms.

The first algorithm in the list is treated as the "proposed" algorithm.
Swap it with your own custom class if needed.
"""

import numpy as np
from heurilab import run_experiment, BenchmarkSuite
from heurilab.algorithms import (
    # Swarm Intelligence
    PSO, GWO, WOA, MFO, SSA, HHO, MPA,
    # Evolutionary
    GA, DE, ES, EP,
    # Physics-based
    GSA, MVO, SCA, AOA,
    # Human/Social
    TLBO, JA,
    # Bio-inspired
    ABC, FA, SOS,
)


# ── Benchmark Functions ──────────────────────────────────────────────

def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# ── Define Suites ────────────────────────────────────────────────────

classical = BenchmarkSuite("Classical")
classical.add("Sphere", sphere, lb=-100, ub=100, dim=30)
classical.add("Rastrigin", rastrigin, lb=-5.12, ub=5.12, dim=30)
classical.add("Ackley", ackley, lb=-32, ub=32, dim=30)
classical.add("Rosenbrock", rosenbrock, lb=-30, ub=30, dim=30)


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    algorithms = [
        # ─── Proposed (first = highlighted) ───
        ("PSO", PSO),
        # ─── Swarm Intelligence ───
        ("GWO", GWO),
        ("WOA", WOA),
        ("MFO", MFO),
        ("SSA", SSA),
        ("HHO", HHO),
        ("MPA", MPA),
        # ─── Evolutionary ───
        ("GA", GA),
        ("DE", DE),
        ("ES", ES),
        ("EP", EP),
        # ─── Physics-based ───
        ("GSA", GSA),
        ("MVO", MVO),
        ("SCA", SCA),
        ("AOA", AOA),
        # ─── Human/Social ───
        ("TLBO", TLBO),
        ("JA", JA),
        # ─── Bio-inspired ───
        ("ABC", ABC),
        ("FA", FA),
        ("SOS", SOS),
    ]

    run_experiment(
        algorithms=algorithms,
        benchmark_suites=[classical],
        output_dir="output",
        pop_size=30,
        max_iter=100,
        dim=30,
        n_runs=5,            # Use 30 for real experiments
        run_engineering=True,
        engineering_n_runs=5, # Use 30 for real experiments
    )

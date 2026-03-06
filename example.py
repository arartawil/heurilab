"""
Example usage of the heurilab package with all 80 built-in algorithms.

The first algorithm in the list is treated as the "proposed" algorithm.
Swap it with your own custom class if needed.
"""

import numpy as np
from heurilab import run_experiment, get_classical_suite, get_unimodal_suite, get_multimodal_suite, get_cec2017_suite
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


# ── Built-in Benchmark Suites ───────────────────────────────────────
# Use all 23 classical functions (F1–F23):
#   classical = get_classical_suite()
#
# Or split by type:
#   unimodal   = get_unimodal_suite()     # F1–F7
#   multimodal = get_multimodal_suite()    # F8–F13
#   fixeddim   = get_fixeddim_suite()      # F14–F23
#
# CEC 2017 (29 functions):
#   cec2017 = get_cec2017_suite()          # All 29 (F1,F3–F30)

classical = get_classical_suite()
cec2017   = get_cec2017_suite()


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
        benchmark_suites=[classical, cec2017],
        output_dir="output",
        pop_size=30,
        max_iter=100,
        dim=30,
        n_runs=5,            # Use 30 for real experiments
        run_engineering=True,
        engineering_n_runs=5, # Use 30 for real experiments
    )

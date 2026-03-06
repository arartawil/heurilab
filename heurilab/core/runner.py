"""
Main experiment runner — orchestrates all outputs.
"""

import os
import time
import numpy as np
from typing import List, Tuple, Type, Dict, Optional

from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite
from heurilab.exporters.csv_export import (
    init_csv_files, append_raw_run, append_results, append_convergence, _pad_or_trim
)
from heurilab.exporters.plots import plot_convergence, plot_boxplot
from heurilab.exporters.excel_export import (
    generate_results_excel, generate_wilcoxon_excel, generate_friedman_excel
)
from heurilab.engineering.runner import run_engineering_problems


def run_experiment(
    algorithms: List[Tuple[str, Type]],
    benchmark_suites: List[BenchmarkSuite],
    output_dir: str = "output",
    pop_size: int = 50,
    max_iter: int = 300,
    dim: int = 30,
    n_runs: int = 30,
    run_engineering: bool = True,
    engineering_pop_size: int = 50,
    engineering_max_iter: int = 500,
    engineering_n_runs: int = 30,
):
    """
    Run the full experiment pipeline.

    Parameters
    ----------
    algorithms : list of (name, AlgorithmClass) tuples
        First entry is the 'proposed' algorithm.
    benchmark_suites : list of BenchmarkSuite
        Each suite is a category with benchmark configs.
    output_dir : str
        Root output directory.
    pop_size, max_iter, dim, n_runs : int
        Experiment settings.
    run_engineering : bool
        Whether to run engineering design problems.
    engineering_pop_size, engineering_max_iter, engineering_n_runs : int
        Settings for engineering problems.
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_names = [name for name, _ in algorithms]
    proposed_name = algo_names[0]

    print(f"{'='*60}")
    print(f"  Metaheuristic Experiment")
    print(f"  Algorithms: {', '.join(algo_names)}")
    print(f"  Proposed: {proposed_name}")
    print(f"  Pop={pop_size}, MaxIter={max_iter}, Dim={dim}, Runs={n_runs}")
    print(f"{'='*60}\n")

    # ── Phase 1: Run experiments & produce CSVs ──────────────────────
    csv_paths = init_csv_files(output_dir, max_iter)

    # Data collectors for Excel/plots
    results_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    raw_data: Dict[str, Dict[str, List[float]]] = {}
    convergence_data: Dict[str, Dict[str, list]] = {}
    suites_map: Dict[str, List[str]] = {}

    total_combos = sum(len(s.benchmarks) for s in benchmark_suites) * len(algorithms)
    combo_count = 0

    for suite in benchmark_suites:
        category = suite.category
        func_names_in_suite = []
        suites_map[category] = func_names_in_suite

        for bench in suite.benchmarks:
            func_name = bench.name
            func_names_in_suite.append(func_name)

            if func_name not in raw_data:
                raw_data[func_name] = {}
                results_data[func_name] = {}
                convergence_data[func_name] = {}

            for algo_name, algo_class in algorithms:
                combo_count += 1
                print(f"[{combo_count}/{total_combos}] {func_name} × {algo_name}")

                run_fitnesses = []
                run_convergences = []

                for run in range(n_runs):
                    bench_dim = bench.dim if bench.dim else dim
                    algo = algo_class(
                        pop_size=pop_size,
                        dim=bench_dim,
                        lb=bench.lb,
                        ub=bench.ub,
                        max_iter=max_iter,
                        obj_func=bench.obj_func,
                    )

                    t0 = time.time()
                    best_sol, best_fit, conv = algo.optimize()
                    elapsed = time.time() - t0

                    conv = list(conv)
                    run_fitnesses.append(best_fit)
                    run_convergences.append(conv)

                    append_raw_run(csv_paths["raw_runs"], func_name, algo_name,
                                   run, best_fit, elapsed, conv, max_iter)

                # Flush summary rows
                append_results(csv_paths["results"], func_name, algo_name, run_fitnesses)
                append_convergence(csv_paths["convergence"], func_name, algo_name,
                                   run_convergences, max_iter)

                # Store for Excel/plots
                arr = np.array(run_fitnesses)
                results_data[func_name][algo_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "best": float(np.min(arr)),
                    "worst": float(np.max(arr)),
                    "median": float(np.median(arr)),
                }
                raw_data[func_name][algo_name] = run_fitnesses

                mean_conv = np.mean(
                    [_pad_or_trim(c, max_iter + 1) for c in run_convergences],
                    axis=0
                )
                convergence_data[func_name][algo_name] = list(mean_conv)

                print(f"       Mean={np.mean(arr):.4e}  Std={np.std(arr):.4e}")

    print(f"\nCSV files written to: {os.path.join(output_dir, 'CSV Data')}")

    # ── Phase 2: Convergence & Box plots ─────────────────────────────
    print("\nGenerating plots...")
    all_funcs = []
    for suite in benchmark_suites:
        for bench in suite.benchmarks:
            all_funcs.append(bench.name)

    for func_name in all_funcs:
        plot_convergence(func_name, algo_names,
                         convergence_data.get(func_name, {}),
                         output_dir, max_iter)
        plot_boxplot(func_name, algo_names,
                     raw_data.get(func_name, {}),
                     output_dir)

    print(f"Plots saved to: {output_dir}")

    # ── Phase 3: Excel outputs ───────────────────────────────────────
    print("\nGenerating Excel files...")
    generate_results_excel(results_data, algo_names, suites_map, output_dir)
    generate_wilcoxon_excel(raw_data, algo_names, suites_map, output_dir)
    generate_friedman_excel(raw_data, algo_names, suites_map, output_dir)
    print(f"Excel files saved to: {os.path.join(output_dir, 'Excel Files')}")

    # ── Phase 4: Engineering problems ────────────────────────────────
    if run_engineering:
        print("\nRunning engineering design problems...")
        run_engineering_problems(
            proposed_name, algorithms[0][1], output_dir,
            pop_size=engineering_pop_size,
            max_iter=engineering_max_iter,
            n_runs=engineering_n_runs,
        )

    print(f"\n{'='*60}")
    print("  Experiment complete!")
    print(f"  All outputs in: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

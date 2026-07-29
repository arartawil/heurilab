"""
Main experiment runner — orchestrates all outputs.
"""

import os
import time
import hashlib
import numpy as np
from typing import List, Tuple, Type, Dict, Optional
from tqdm import tqdm

from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite
from heurilab.exporters.csv_export import (
    init_csv_files, append_raw_run, append_results, append_convergence, _pad_or_trim
)
from heurilab.exporters.plots import plot_convergence, plot_boxplot
from heurilab.exporters.excel_export import (
    generate_results_excel, generate_wilcoxon_excel, generate_friedman_excel
)
from heurilab.engineering.runner import run_engineering_problems


def _execute_run(algo_class, spec, seed):
    """Run one algorithm once. Module-level so it can be sent to a worker process.

    Returns ``(best_fitness, convergence, elapsed_seconds)``. The solution vector
    is deliberately not returned: for a 100-dimensional, 30-run, 60-function
    campaign it would dominate the data shipped back from workers, and nothing
    downstream consumes it.
    """
    if seed is not None:
        # Stochastic objectives (F7 Quartic and any user objective using the
        # legacy global stream) must be pinned inside the worker, not the parent.
        np.random.seed(seed % (2 ** 32))
    algo = algo_class(seed=seed, **spec)
    t0 = time.time()
    _, best_fit, conv = algo.optimize()
    return float(best_fit), list(conv), time.time() - t0, int(getattr(algo, "n_fes", 0))


def _make_executor(n_jobs: int):
    """Return a callable mapping ``_execute_run`` over a list of (spec, seed) jobs.

    ``n_jobs == 1`` yields a lazy generator so that serial runs keep writing to
    raw_runs.csv after every single run, which is what makes a long campaign
    crash-resistant. Parallel runs necessarily flush per (function, algorithm)
    block instead.
    """
    if n_jobs == 1:
        def _serial(algo_class, spec, seeds):
            return (_execute_run(algo_class, spec, s) for s in seeds)
        return _serial

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:
        raise ImportError(
            "n_jobs != 1 requires the optional 'joblib' package.\n"
            "    pip install joblib\n"
            "or  pip install heurilab[parallel]"
        ) from exc

    def _parallel(algo_class, spec, seeds):
        return Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_execute_run)(algo_class, spec, s) for s in seeds)
    return _parallel


def run_experiment(
    algorithms: List[Tuple[str, Type]],
    benchmark_suites: List[BenchmarkSuite],
    output_dir: str = "output",
    pop_size: int = 50,
    max_iter: int = 300,
    dim: int = 30,
    n_runs: int = 30,
    seed: Optional[int] = None,
    max_fes: Optional[int] = None,
    n_jobs: int = 1,
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
    max_fes : int or None
        Objective-evaluation budget per run, shared by every algorithm. This is
        the fair basis for comparison: algorithms differ several-fold in
        evaluations consumed per iteration, so equal ``max_iter`` gives some of
        them far more search than others. When set, each algorithm's
        ``max_iter`` is calibrated so its annealing schedules complete exactly
        as the budget is spent, and a hard cap stops any run that would exceed
        it. ``None`` (default) keeps the historical equal-iteration behaviour.
    n_jobs : int
        Independent runs to execute in parallel (``-1`` uses every core).
        ``1`` (default) runs serially and writes ``raw_runs.csv`` after every
        single run. Any other value requires ``joblib`` and flushes per
        (function, algorithm) block instead. Results are identical either way
        when ``seed`` is set: each run's seed is derived up front, so execution
        order cannot change the outcome.
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
    print(f"  Seed: {seed if seed is not None else 'none (not reproducible)'}")
    print(f"  Parallel: {'serial' if n_jobs == 1 else f'{n_jobs} workers'}")
    if max_fes is not None:
        print(f"  Budget: {max_fes:,} function evaluations per run "
              f"(max_iter calibrated per algorithm)")
    print(f"{'='*60}\n")

    # ── Phase 1: Run experiments & produce CSVs ──────────────────────
    execute = _make_executor(n_jobs)

    # With an evaluation budget, every algorithm needs its own iteration count
    # so that its schedules finish exactly as the budget runs out.
    calibrated = {}
    if max_fes is not None:
        from heurilab.core.budget import calibrate_iterations
        for algo_name, algo_class in algorithms:
            iters, per_iter = calibrate_iterations(
                algo_class, max_fes, pop_size=pop_size, dim=dim)
            calibrated[algo_name] = iters
            print(f"    {algo_name:12} {per_iter:6.1f} evals/iter -> max_iter={iters}")
        max_iter = max(calibrated.values())      # CSV column width

    csv_paths = init_csv_files(output_dir, max_iter)

    # Data collectors for Excel/plots
    results_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    raw_data: Dict[str, Dict[str, List[float]]] = {}
    convergence_data: Dict[str, Dict[str, list]] = {}
    suites_map: Dict[str, List[str]] = {}

    total_combos = sum(len(s.benchmarks) for s in benchmark_suites) * len(algorithms)

    # ── Main progress bar ────────────────────────────────────────────
    pbar = tqdm(total=total_combos, desc="Experiment",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="green", dynamic_ncols=True)

    def _run_seed(func_name: str, algo_name: str, run: int):
        """Deterministic per-run seed, stable under reordering of the matrix."""
        if seed is None:
            return None
        key = [seed,
               int(hashlib.sha256(func_name.encode()).hexdigest()[:8], 16),
               int(hashlib.sha256(algo_name.encode()).hexdigest()[:8], 16),
               run]
        return int(np.random.SeedSequence(key).generate_state(1)[0])

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
                pbar.set_postfix_str(f"{func_name} × {algo_name}", refresh=True)

                run_fitnesses = []
                run_convergences = []

                bench_dim = bench.dim if bench.dim else dim
                run_seeds = [_run_seed(func_name, algo_name, r) for r in range(n_runs)]
                spec = dict(pop_size=pop_size, dim=bench_dim,
                            lb=bench.lb, ub=bench.ub,
                            max_iter=calibrated.get(algo_name, max_iter),
                            obj_func=bench.obj_func)
                if max_fes is not None:
                    spec["max_fes"] = max_fes

                for run, (best_fit, conv, elapsed, n_fes) in enumerate(
                        execute(algo_class, spec, run_seeds)):
                    run_fitnesses.append(best_fit)
                    run_convergences.append(conv)

                    # Serial: written after every single run. Parallel: after
                    # each (function, algorithm) block.
                    append_raw_run(csv_paths["raw_runs"], func_name, algo_name,
                                   run, best_fit, elapsed, conv, max_iter,
                                   seed=run_seeds[run], n_fes=n_fes)

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

                pbar.update(1)

    pbar.close()
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
            seed=seed,
            n_jobs=n_jobs,
        )

    print(f"\n{'='*60}")
    print("  Experiment complete!")
    print(f"  All outputs in: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

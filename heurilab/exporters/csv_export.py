"""
CSV output generation: results.csv, raw_runs.csv, convergence.csv
"""

import csv
import os
import numpy as np
from typing import Dict, List, Tuple


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def init_csv_files(output_dir: str, max_iter: int):
    """Create CSV files with headers. Returns dict of file paths."""
    csv_dir = os.path.join(output_dir, "CSV Data")
    os.makedirs(csv_dir, exist_ok=True)

    paths = {
        "results": os.path.join(csv_dir, "results.csv"),
        "raw_runs": os.path.join(csv_dir, "raw_runs.csv"),
        "convergence": os.path.join(csv_dir, "convergence.csv"),
    }

    # results.csv header
    with open(paths["results"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Algorithm", "Mean", "Std", "Best", "Worst", "Median"])

    # raw_runs.csv header
    with open(paths["raw_runs"], "w", newline="") as f:
        writer = csv.writer(f)
        conv_headers = [f"Conv_{i}" for i in range(max_iter + 1)]
        writer.writerow(["Benchmark", "Algorithm", "Run", "BestFitness", "Time_s"] + conv_headers)

    # convergence.csv header
    with open(paths["convergence"], "w", newline="") as f:
        writer = csv.writer(f)
        iter_headers = [f"Iter_{i}" for i in range(max_iter + 1)]
        writer.writerow(["Benchmark", "Algorithm"] + iter_headers)

    return paths


def _pad_or_trim(conv: list, length: int) -> list:
    """Pad or trim convergence list to exactly `length` entries."""
    conv = list(conv)
    if len(conv) >= length:
        return conv[:length]
    if len(conv) == 0:
        return [0.0] * length
    return conv + [conv[-1]] * (length - len(conv))


def append_raw_run(csv_path: str, benchmark_name: str, algo_name: str,
                   run_idx: int, best_fitness: float, elapsed: float,
                   convergence: list, max_iter: int):
    """Append one run to raw_runs.csv."""
    conv = _pad_or_trim(convergence, max_iter + 1)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([benchmark_name, algo_name, run_idx + 1, best_fitness, f"{elapsed:.4f}"] + conv)


def append_results(csv_path: str, benchmark_name: str, algo_name: str,
                   fitness_values: List[float]):
    """Append summary statistics for one algorithm-function combo to results.csv."""
    arr = np.array(fitness_values)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            benchmark_name, algo_name,
            f"{np.mean(arr):.6e}", f"{np.std(arr):.6e}",
            f"{np.min(arr):.6e}", f"{np.max(arr):.6e}",
            f"{np.median(arr):.6e}"
        ])


def append_convergence(csv_path: str, benchmark_name: str, algo_name: str,
                       all_convergences: List[list], max_iter: int):
    """Append mean convergence for one algorithm-function combo to convergence.csv."""
    length = max_iter + 1
    padded = [_pad_or_trim(c, length) for c in all_convergences]
    mean_conv = np.mean(padded, axis=0)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([benchmark_name, algo_name] + [f"{v:.6e}" for v in mean_conv])

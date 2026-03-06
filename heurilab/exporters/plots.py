"""
Convergence curve plots and box plots.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


# Distinct colors and markers for algorithms
_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#000000", "#ffe119", "#ffd8b1",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h",
            "d", ">", "<", "8", "H", "+", "x", "1", "2", "3"]


def _get_style(idx: int, n_algos: int):
    color = _COLORS[idx % len(_COLORS)]
    marker = _MARKERS[idx % len(_MARKERS)]
    return color, marker


def plot_convergence(benchmark_name: str, algo_names: List[str],
                     mean_convergences: Dict[str, list],
                     output_dir: str, max_iter: int):
    """
    Plot convergence curves for all algorithms on one figure.
    First algorithm in algo_names is treated as 'proposed'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(max_iter + 1)
    marker_every = max(1, max_iter // 10)
    n = len(algo_names)

    # Plot competitors first, proposed last
    order = list(range(1, n)) + [0]
    for plot_order, idx in enumerate(order):
        name = algo_names[idx]
        conv = mean_convergences.get(name, [])
        if len(conv) == 0:
            continue
        conv = np.array(conv[:max_iter + 1])
        color, marker = _get_style(idx, n)

        is_proposed = (idx == 0)
        lw = 2.5 if is_proposed else 1.5
        zord = n + 1 if is_proposed else plot_order + 1

        ax.plot(x[:len(conv)], conv, color=color, marker=marker,
                markevery=marker_every, markersize=5, linewidth=lw,
                label=name, zorder=zord)

    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Fitness (log scale)", fontsize=11)
    ax.set_title(benchmark_name, fontsize=13)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(output_dir, "Convergence Curves")
    os.makedirs(save_dir, exist_ok=True)
    safe_name = benchmark_name.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(benchmark_name: str, algo_names: List[str],
                 fitness_data: Dict[str, List[float]],
                 output_dir: str):
    """
    Box plot of BestFitness across runs for all algorithms side-by-side.
    First algorithm is 'proposed' and highlighted.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    data = []
    colors = []
    for idx, name in enumerate(algo_names):
        vals = fitness_data.get(name, [])
        data.append(vals)
        c, _ = _get_style(idx, len(algo_names))
        colors.append(c)

    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                  markeredgecolor="black", markersize=5),
                    labels=algo_names)

    for idx, (patch, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        patch.set_facecolor(colors[idx])
        if idx == 0:  # proposed
            patch.set_alpha(0.9)
            patch.set_edgecolor("#FF0000")
            patch.set_linewidth(2.0)
        else:
            patch.set_alpha(0.6)

    # symlog if range is large
    all_vals = [v for sublist in data for v in sublist if v != 0]
    if len(all_vals) > 0:
        vmin, vmax = min(abs(v) for v in all_vals if v != 0) if any(v != 0 for v in all_vals) else 1, max(abs(v) for v in all_vals) if all_vals else 1
        if vmax / max(vmin, 1e-300) > 1000:
            ax.set_yscale("symlog", linthresh=1e-10)

    ax.set_title(benchmark_name, fontsize=13)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=8)

    save_dir = os.path.join(output_dir, "Box Plots")
    os.makedirs(save_dir, exist_ok=True)
    safe_name = benchmark_name.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

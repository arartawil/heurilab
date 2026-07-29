"""
Qualitative behaviour analysis.

Produces the six-panel figure that accompanies most metaheuristic proposals:
the objective landscape, the points the algorithm actually sampled, the
population's average fitness, the trajectory of a single decision variable, the
convergence curve, and the exploration-versus-exploitation balance.

Every panel is measured from one instrumented run rather than supplied by the
algorithm, so the figure can be produced for any optimizer conforming to the
``_Base`` interface - including one the user has just written.

The analysis is performed in two dimensions by convention. Search history and
variable trajectories are only interpretable when the search space can be drawn,
and the landscape panel requires it outright.

Exploration and exploitation percentages follow the dimension-wise diversity
measure of Hussain et al. (2019):

.. math::

    Div_j = \\frac{1}{n}\\sum_{i=1}^{n} | \\mathrm{median}(x_j) - x_{ij} |,
    \\qquad Div = \\frac{1}{D}\\sum_{j=1}^{D} Div_j

with :math:`XPL\\% = (Div / Div_{max}) \\times 100` and
:math:`XPT\\% = (|Div - Div_{max}| / Div_{max}) \\times 100`, where
:math:`Div_{max}` is the largest diversity observed during the run.

References
----------
Hussain, K., Salleh, M. N. M., Cheng, S., Shi, Y. (2019). On the exploration
and exploitation in popular swarm-based metaheuristic algorithms. *Neural
Computing and Applications*, 31(11), 7665-7683.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Type

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d projection)


@dataclass
class QualitativeAnalysis:
    """Measured behaviour of one algorithm on one benchmark function."""
    algorithm: str
    function: str
    lb: float
    ub: float
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    search_history: np.ndarray        # (n_evaluations, 2)
    average_fitness: np.ndarray       # (T,)
    trajectory: np.ndarray            # (T,) first variable of the first agent
    convergence: np.ndarray           # (T,) best-so-far
    exploration: np.ndarray           # (T,) percentage, NaN if not population-based
    exploitation: np.ndarray          # (T,) percentage, NaN if not population-based
    evals_per_iteration: float
    population_based: bool = True     # diversity is undefined for single-solution methods

    @property
    def mean_exploration(self) -> float:
        return float(np.nanmean(self.exploration)) if self.population_based else float("nan")

    @property
    def crossover_iteration(self) -> Optional[int]:
        """First iteration at which exploitation overtakes exploration.

        A well-balanced algorithm crosses over once, early-to-middle in the run.
        A method that never crosses has not converged; one that crosses in the
        opening iterations has collapsed prematurely.
        """
        if not self.population_based:
            return None
        over = np.where(self.exploitation > self.exploration)[0]
        return int(over[0]) + 1 if over.size else None

    def summary(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "function": self.function,
            "iterations": int(len(self.convergence)),
            "evals_per_iteration": float(self.evals_per_iteration),
            "population_based": bool(self.population_based),
            "mean_exploration_pct": self.mean_exploration,
            "mean_exploitation_pct": (float(np.nanmean(self.exploitation))
                                      if self.population_based else float("nan")),
            "crossover_iteration": self.crossover_iteration,
            "final_best": float(self.convergence[-1]),
        }

    def __repr__(self):
        return (f"QualitativeAnalysis({self.algorithm} on {self.function}, "
                f"XPL={self.mean_exploration:.1f}%, "
                f"crossover={self.crossover_iteration})")


class _Recorder:
    """Objective wrapper that records every evaluated point in order."""

    def __init__(self, objective):
        self.objective = objective
        self.points = []
        self.values = []

    def __call__(self, x):
        arr = np.asarray(x, dtype=float).ravel()
        value = float(self.objective(arr))
        self.points.append(arr.copy())
        self.values.append(value)
        return value


def _diversity(population: np.ndarray) -> float:
    """Dimension-wise median diversity of one population."""
    med = np.median(population, axis=0)
    return float(np.mean(np.mean(np.abs(med - population), axis=0)))


def _landscape(objective, lb, ub, resolution: int = 90):
    """Evaluate the objective on a 2-D grid for the surface panel."""
    xs = np.linspace(lb, ub, resolution)
    ys = np.linspace(lb, ub, resolution)
    gx, gy = np.meshgrid(xs, ys)
    gz = np.empty_like(gx)
    for i in range(resolution):
        for j in range(resolution):
            gz[i, j] = float(objective(np.array([gx[i, j], gy[i, j]])))
    return gx, gy, gz


def qualitative_analysis(algorithm,
                         benchmark,
                         pop_size: int = 30,
                         max_iter: int = 500,
                         seed: int = 20260728,
                         resolution: int = 90) -> QualitativeAnalysis:
    """
    Run one algorithm on one benchmark and measure its qualitative behaviour.

    Parameters
    ----------
    algorithm : (str, type)
        ``(name, AlgorithmClass)``, where the class subclasses ``_Base``.
    benchmark : BenchmarkConfig or (str, callable, lb, ub)
        The function to analyse. Its dimensionality is overridden to two.
    pop_size, max_iter, seed : int
        Run settings.
    resolution : int
        Grid resolution of the landscape panel. Lower it for costly objectives.

    Returns
    -------
    QualitativeAnalysis
    """
    algo_name, algo_class = algorithm

    if hasattr(benchmark, "obj_func"):
        fn_name, objective = benchmark.name, benchmark.obj_func
        lb = float(np.min(benchmark.lb)); ub = float(np.max(benchmark.ub))
    else:
        fn_name, objective, lb, ub = benchmark
        lb, ub = float(lb), float(ub)

    rec = _Recorder(objective)
    algo = algo_class(pop_size=pop_size, dim=2, lb=lb, ub=ub,
                      max_iter=max_iter, obj_func=rec, seed=seed)
    _, _, conv = algo.optimize()

    pts = np.asarray(rec.points, dtype=float)
    vals = np.asarray(rec.values, dtype=float)
    n_iter = max(len(list(conv)) - 1, 1)
    per_iter = len(pts) / n_iter
    block = max(int(round(per_iter)), 1)
    n_blocks = len(pts) // block
    if n_blocks < 3:
        raise ValueError(
            f"{algo_name}: only {n_blocks} usable iteration blocks. "
            "Increase max_iter or pop_size.")

    pop = pts[:n_blocks * block].reshape(n_blocks, block, 2)
    fit = vals[:n_blocks * block].reshape(n_blocks, block)

    average_fitness = fit.mean(axis=1)
    trajectory = pop[:, 0, 0]
    convergence = np.minimum.accumulate(fit.min(axis=1))

    # Population diversity is only meaningful when there is a population to
    # measure. Single-solution methods (Harmony Search, Simulated Annealing)
    # evaluate one or two candidates per iteration, and reporting 0% exploration
    # for them would be a measurement artifact rather than a finding.
    population_based = block >= 3
    if population_based:
        div = np.array([_diversity(pop[t]) for t in range(n_blocks)])
        div_max = float(div.max()) or 1.0
        exploration = div / div_max * 100.0
        exploitation = np.abs(div - div_max) / div_max * 100.0
    else:
        exploration = np.full(n_blocks, np.nan)
        exploitation = np.full(n_blocks, np.nan)

    gx, gy, gz = _landscape(objective, lb, ub, resolution)

    return QualitativeAnalysis(
        algorithm=algo_name, function=fn_name, lb=lb, ub=ub,
        grid_x=gx, grid_y=gy, grid_z=gz,
        search_history=pts, average_fitness=average_fitness,
        trajectory=trajectory, convergence=convergence,
        exploration=exploration, exploitation=exploitation,
        evals_per_iteration=per_iter, population_based=population_based,
    )


def plot_qualitative(analysis: QualitativeAnalysis,
                     output_dir: str = "qualitative",
                     filename: Optional[str] = None,
                     layout: str = "row",
                     log_convergence: bool = True,
                     dpi: int = 150,
                     font_scale: float = 1.0) -> str:
    """
    Draw the six-panel qualitative figure.

    Parameters
    ----------
    analysis : QualitativeAnalysis
    output_dir, filename : str
    layout : {"row", "grid"}
        ``"row"`` places all six panels side by side, as in the published
        convention; ``"grid"`` uses two rows of three, which reproduces better
        in a two-column manuscript.
    log_convergence : bool
        Logarithmic ordinate on the convergence panel, which is usual when
        fitness spans several orders of magnitude.
    font_scale : float
        Multiplier on every label, tick and title size. The defaults suit
        on-screen viewing; a figure reduced to one text column in a manuscript
        needs roughly ``font_scale=1.8`` for its axes to stay legible.

    Returns
    -------
    str
        Path to the written figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    a = analysis
    t = np.arange(1, len(a.convergence) + 1)

    if layout == "row":
        fig = plt.figure(figsize=(26, 3.9))
        shape = (1, 6)
    else:
        fig = plt.figure(figsize=(15, 8.5))
        shape = (2, 3)

    # ── 1. landscape ─────────────────────────────────────────────────
    ax = fig.add_subplot(*shape, 1, projection="3d")
    ax.plot_surface(a.grid_x, a.grid_y, a.grid_z, cmap="viridis",
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.set_title(a.function, fontsize=9 * font_scale)
    ax.set_xlabel("x1", fontsize=7 * font_scale); ax.set_ylabel("x2", fontsize=7 * font_scale)
    ax.tick_params(labelsize=5 * font_scale)

    # ── 2. search history ────────────────────────────────────────────
    ax = fig.add_subplot(*shape, 2)
    ax.scatter(a.search_history[:, 0], a.search_history[:, 1],
               s=4, alpha=0.55, color="tab:blue", edgecolors="none")
    ax.set_title("Search history", fontsize=9 * font_scale)
    ax.set_xlabel("x1", fontsize=8 * font_scale); ax.set_ylabel("x2", fontsize=8 * font_scale)
    ax.set_xlim(a.lb, a.ub); ax.set_ylim(a.lb, a.ub)
    ax.tick_params(labelsize=7 * font_scale)

    # ── 3. average fitness ───────────────────────────────────────────
    ax = fig.add_subplot(*shape, 3)
    ax.plot(t, a.average_fitness, lw=0.9, color="tab:blue")
    ax.set_title("Average fitness", fontsize=9 * font_scale)
    ax.set_xlabel("Iteration", fontsize=8 * font_scale); ax.set_ylabel("Fitness", fontsize=8 * font_scale)
    if np.all(a.average_fitness > 0) and a.average_fitness.max() / max(a.average_fitness.min(), 1e-30) > 1e3:
        ax.set_yscale("log")
    ax.tick_params(labelsize=7 * font_scale)

    # ── 4. trajectory of the first variable ──────────────────────────
    ax = fig.add_subplot(*shape, 4)
    ax.plot(t, a.trajectory, lw=0.9, color="tab:blue")
    ax.set_title("Trajectory of 1st dimension", fontsize=9 * font_scale)
    ax.set_xlabel("Iteration", fontsize=8 * font_scale); ax.set_ylabel("Value", fontsize=8 * font_scale)
    ax.tick_params(labelsize=7 * font_scale)

    # ── 5. convergence ───────────────────────────────────────────────
    ax = fig.add_subplot(*shape, 5)
    ax.plot(t, a.convergence, lw=1.1, color="tab:blue")
    if log_convergence and np.all(a.convergence > 0):
        ax.set_yscale("log")
    ax.set_title("Convergence curve", fontsize=9 * font_scale)
    ax.set_xlabel("Iteration", fontsize=8 * font_scale); ax.set_ylabel("Best score", fontsize=8 * font_scale)
    ax.tick_params(labelsize=7 * font_scale)

    # ── 6. exploration versus exploitation ───────────────────────────
    ax = fig.add_subplot(*shape, 6)
    if not a.population_based:
        ax.text(0.5, 0.5,
                "Exploration/exploitation\nundefined:\nsingle-solution method",
                ha="center", va="center", fontsize=8 * font_scale, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Exploration vs. exploitation", fontsize=9 * font_scale)
        fig.tight_layout()
        path = os.path.join(
            output_dir, filename or f"qualitative_{a.algorithm}_{a.function}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    ax.plot(t, a.exploration, lw=1.0, color="tab:red", label="Exploration")
    ax.plot(t, a.exploitation, lw=1.0, color="tab:blue", label="Exploitation")
    ax.fill_between(t, a.exploration, alpha=0.18, color="tab:red")
    if a.crossover_iteration:
        ax.axvline(a.crossover_iteration, color="0.4", ls=":", lw=1.0)
        ax.plot(a.crossover_iteration, 50, "o", ms=5, color="tab:blue")
    ax.set_title("Exploration vs. exploitation", fontsize=9 * font_scale)
    ax.set_xlabel("Iteration", fontsize=8 * font_scale); ax.set_ylabel("Percentage (%)", fontsize=8 * font_scale)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6 * font_scale, loc="center right")
    ax.tick_params(labelsize=7 * font_scale)

    fig.tight_layout()
    path = os.path.join(
        output_dir, filename or f"qualitative_{a.algorithm}_{a.function}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def qualitative_report(algorithm,
                       benchmarks: Sequence,
                       output_dir: str = "qualitative",
                       pop_size: int = 30,
                       max_iter: int = 500,
                       seed: int = 20260728,
                       layout: str = "row",
                       resolution: int = 90,
                       verbose: bool = True):
    """
    Produce the six-panel figure for one algorithm across several functions.

    Returns
    -------
    (list of QualitativeAnalysis, list of str)
        The analyses and the paths of the figures written.

    Examples
    --------
    >>> from heurilab import get_opfunu_suite
    >>> from heurilab.algorithms import GWO
    >>> from heurilab.analyzer import qualitative_report
    >>> suite = get_opfunu_suite("2017", ndim=2, functions=[1, 22])   # doctest: +SKIP
    >>> qualitative_report(("GWO", GWO), suite.benchmarks)            # doctest: +SKIP
    """
    analyses, paths = [], []
    for bench in benchmarks:
        name = bench.name if hasattr(bench, "name") else bench[0]
        if verbose:
            print(f"  qualitative: {algorithm[0]} on {name} ...", end=" ", flush=True)
        a = qualitative_analysis(algorithm, bench, pop_size=pop_size,
                                 max_iter=max_iter, seed=seed, resolution=resolution)
        p = plot_qualitative(a, output_dir, layout=layout)
        analyses.append(a); paths.append(p)
        if verbose:
            print(f"XPL={a.mean_exploration:.1f}%  crossover={a.crossover_iteration}")
    return analyses, paths


def balance_table(analyses: Sequence[QualitativeAnalysis]) -> str:
    """
    Markdown table of the exploration-exploitation balance across functions.

    Useful as a companion to the figures: the crossover iteration summarises in
    one number what the sixth panel shows, and makes premature convergence
    visible across a whole suite at a glance.
    """
    lines = ["| function | mean XPL % | mean XPT % | crossover iteration | final best |",
             "|---|---|---|---|---|"]
    for a in analyses:
        if not a.population_based:
            lines.append(f"| {a.function} | n/a | n/a | n/a | "
                         f"{a.convergence[-1]:.4e} |")
            continue
        cross = a.crossover_iteration if a.crossover_iteration else "never"
        lines.append(f"| {a.function} | {a.mean_exploration:.1f} | "
                     f"{np.nanmean(a.exploitation):.1f} | {cross} | "
                     f"{a.convergence[-1]:.4e} |")
    return "\n".join(lines)

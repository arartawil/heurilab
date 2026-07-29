"""
Exploration-coefficient traces.

Metaheuristic papers routinely include a figure showing how the algorithm's
control coefficient behaves over the run - GWO's ``a`` falling from 2 to 0, or a
term such as :math:`\\Phi = (1 - t/T)\\,\\mathcal{N}(0,1)` - to argue that the
method explores early and exploits late.

Those figures are normally drawn from the formula. This module measures the
coefficient **empirically instead**, from the algorithm's actual behaviour, so
the same plot can be produced for any optimizer including one whose control law
is not written down anywhere.

Method: the algorithm is run against an instrumented objective that records
every point it evaluates. Evaluations are reshaped into per-iteration blocks,
each agent's step ``x_i(t) - x_i(t-1)`` is projected onto a fixed random unit
direction, and the result is scaled so the early steps have unit dispersion.
What comes out is the effective exploration coefficient the algorithm applied -
the same quantity the analytical figure shows, but observed rather than assumed.

A gap between the two is itself informative: if the paper claims linear decay and
the measured trace does not contract, the implementation does not do what the
formula says.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Type

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ExplorationTrace:
    """Measured exploration coefficient of one algorithm."""
    name: str
    iterations: np.ndarray          # (T,)
    coefficients: np.ndarray        # (n_agents, T)
    envelope: np.ndarray            # (T,) empirical +/- envelope
    evals_per_iteration: float
    contraction: float              # late dispersion / early dispersion

    @property
    def contracts(self) -> bool:
        """True when late-run steps are materially smaller than early ones."""
        return self.contraction < 0.5

    def summary(self) -> dict:
        return {
            "name": self.name,
            "iterations": int(len(self.iterations)),
            "agents_traced": int(self.coefficients.shape[0]),
            "evals_per_iteration": float(self.evals_per_iteration),
            "contraction_ratio": float(self.contraction),
            "contracts": bool(self.contracts),
        }

    def __repr__(self):
        return (f"ExplorationTrace({self.name}, T={len(self.iterations)}, "
                f"contraction={self.contraction:.3f})")


class _TrajectoryRecorder:
    """Objective that records every evaluated point in order.

    Rastrigin by default rather than a sphere: on a sphere most population
    algorithms collapse within a few dozen iterations, so the measured trace
    flatlines and says nothing about the control law. A multimodal landscape
    keeps the population alive long enough for the decay to be visible.
    """

    def __init__(self, objective=None):
        self.points = []
        self.objective = objective

    def __call__(self, x):
        arr = np.asarray(x, dtype=float).ravel()
        self.points.append(arr.copy())
        if self.objective is not None:
            return float(self.objective(arr))
        return float(np.sum(arr ** 2 - 10.0 * np.cos(2.0 * np.pi * arr) + 10.0))


def exploration_trace(name: str,
                      algo_class: Type,
                      dim: int = 30,
                      pop_size: int = 30,
                      max_iter: int = 500,
                      lb: float = -5.12,
                      ub: float = 5.12,
                      seed: int = 20260728,
                      n_agents: int = 2,
                      obj_func=None) -> ExplorationTrace:
    """
    Measure an algorithm's effective exploration coefficient over a run.

    Parameters
    ----------
    name, algo_class
        The algorithm to trace.
    dim, pop_size, max_iter, lb, ub, seed
        Run settings. ``max_iter`` sets the x-axis of the resulting figure.
    n_agents : int
        How many individual agents to follow. Two is the convention in the
        published figures.
    obj_func : callable, optional
        Landscape to trace on. Defaults to Rastrigin, which keeps the population
        diverse long enough for the decay to show.

    Returns
    -------
    ExplorationTrace
    """
    rec = _TrajectoryRecorder(obj_func)
    algo = algo_class(pop_size=pop_size, dim=dim, lb=lb, ub=ub,
                      max_iter=max_iter, obj_func=rec, seed=seed)
    _, _, conv = algo.optimize()

    pts = np.asarray(rec.points, dtype=float)
    n_iter = max(len(list(conv)) - 1, 1)
    per_iter = len(pts) / n_iter
    block = max(int(round(per_iter)), 1)

    # Reshape the evaluation stream into per-iteration blocks. Algorithms with a
    # ragged evaluation pattern are trimmed to whole blocks.
    n_blocks = len(pts) // block
    if n_blocks < 3:
        raise ValueError(
            f"{name}: only {n_blocks} usable iteration blocks "
            f"({len(pts)} evaluations, {block} per iteration). "
            "Increase max_iter or pop_size.")
    traj = pts[:n_blocks * block].reshape(n_blocks, block, dim)

    traced = min(n_agents, block)
    steps = np.diff(traj[:, :traced, :], axis=0)             # (T-1, traced, dim)

    # Project onto one fixed random direction so the trace is signed and
    # oscillates, as in the analytical figures.
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    proj = steps @ direction                                  # (T-1, traced)

    # Scale so the opening steps have unit dispersion, making the vertical axis
    # comparable across algorithms with different natural step sizes.
    early = proj[: max(len(proj) // 10, 1)]
    scale = float(np.std(early)) or 1.0
    coefficients = (proj / scale).T                           # (traced, T-1)

    window = max(len(proj) // 40, 3)
    envelope = np.array([
        float(np.std(proj[max(0, t - window): t + window + 1]) / scale)
        for t in range(len(proj))
    ])

    tail = proj[-max(len(proj) // 10, 1):]
    contraction = float(np.std(tail) / (np.std(early) or 1.0))

    return ExplorationTrace(
        name=name, iterations=np.arange(1, len(proj) + 1),
        coefficients=coefficients, envelope=envelope,
        evals_per_iteration=per_iter, contraction=contraction,
    )


def plot_exploration_coefficient(trace: ExplorationTrace,
                                 output_dir: str = "coefficient_plots",
                                 filename: Optional[str] = None,
                                 title: Optional[str] = None,
                                 reference_envelope: Optional[Sequence[float]] = None,
                                 ylim: float = 2.0,
                                 colors: Sequence[str] = ("blue", "red")) -> str:
    """
    Draw the measured exploration coefficient in the conventional style.

    Two agent traces, dash-dot lines at +/-1, and a dotted envelope showing how
    the perturbation magnitude actually contracts.

    Parameters
    ----------
    trace : ExplorationTrace
    output_dir, filename : str
    title : str, optional
        Defaults to the algorithm name. Pass the control law in LaTeX to match
        the analytical figures, e.g. ``r"$\\Phi = (1 - t/T)\\,\\mathcal{N}(0,1)$"``.
    reference_envelope : sequence, optional
        A theoretical envelope to overlay for comparison, e.g.
        ``1 - t/T``. Drawn as a dashed line so any divergence from the measured
        envelope is visible.
    ylim : float
        Symmetric y-axis limit.
    colors : sequence of str
        One colour per traced agent.

    Returns
    -------
    str
        Path to the written figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    t = trace.iterations
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, series in enumerate(trace.coefficients):
        ax.plot(t, series, lw=0.8, color=colors[i % len(colors)], alpha=0.9,
                label=f"agent {i + 1}")

    ax.axhline(1.0, color="blue", ls="-.", lw=1.3)
    ax.axhline(-1.0, color="blue", ls="-.", lw=1.3)

    ax.plot(t, trace.envelope, color="black", ls=":", lw=1.4,
            label="measured envelope")
    ax.plot(t, -trace.envelope, color="black", ls=":", lw=1.4)

    if reference_envelope is not None:
        ref = np.asarray(reference_envelope, dtype=float)[:len(t)]
        ax.plot(t[:len(ref)], ref, color="green", ls="--", lw=1.2,
                label="theoretical envelope")
        ax.plot(t[:len(ref)], -ref, color="green", ls="--", lw=1.2)

    ax.set_xlabel("iteration")
    ax.set_ylabel("Exploration coefficient")
    ax.set_title(title or f"{trace.name} - measured exploration coefficient")
    ax.set_xlim(0, int(t[-1]))
    ax.set_ylim(-ylim, ylim)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    path = os.path.join(output_dir, filename or f"exploration_{trace.name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def linear_decay_envelope(n_iter: int) -> np.ndarray:
    """The ``1 - t/T`` envelope used by most published control laws."""
    t = np.arange(1, n_iter + 1)
    return 1.0 - t / n_iter


def coefficient_plot(name: str, algo_class: Type,
                     output_dir: str = "coefficient_plots",
                     compare_linear_decay: bool = True,
                     **kw) -> str:
    """
    One-call convenience: trace an algorithm and plot it.

    >>> from heurilab.analyzer.coefficients import coefficient_plot
    >>> from heurilab.algorithms import GWO
    >>> coefficient_plot("GWO", GWO, max_iter=500)

    With ``compare_linear_decay`` the canonical ``1 - t/T`` envelope is overlaid,
    so a claimed linear decay can be checked against what the code actually does.
    """
    plot_kw = {k: kw.pop(k) for k in ("title", "ylim", "colors", "filename")
               if k in kw}
    trace = exploration_trace(name, algo_class, **kw)
    ref = (linear_decay_envelope(len(trace.iterations))
           if compare_linear_decay else None)
    return plot_exploration_coefficient(trace, output_dir,
                                        reference_envelope=ref, **plot_kw)


# ═════════════════════════════════════════════════════════════════════
#  Analytic control-law figure
# ═════════════════════════════════════════════════════════════════════

def plot_control_law(law,
                     max_iter: int = 500,
                     output_dir: str = "coefficient_plots",
                     filename: str = "control_law.png",
                     title: Optional[str] = None,
                     n_traces: int = 2,
                     seed: int = 20260728,
                     envelope=None,
                     ylim: float = 2.0,
                     colors: Sequence[str] = ("blue", "red")) -> str:
    """
    Plot a control law you can write down, in the conventional published style.

    Use this when the coefficient has a closed form and you want the figure for
    a paper. Use :func:`coefficient_plot` when you instead want to measure what
    an implementation actually does.

    Parameters
    ----------
    law : callable
        ``law(t, T, rng) -> float``, the coefficient at iteration ``t`` of
        ``T``. For a coefficient of the form ``(1 - t/T) * N(0,1)``::

            law = lambda t, T, rng: (1 - t / T) * rng.normal()

    max_iter : int
        Number of iterations on the x-axis.
    title : str, optional
        LaTeX is supported.
    n_traces : int
        Independent realisations to draw, two by convention.
    envelope : callable, optional
        ``envelope(t, T) -> float`` for the dotted decay guide. Defaults to
        ``1 - t/T``.

    Returns
    -------
    str
        Path to the written figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(1, max_iter + 1)
    if envelope is None:
        def envelope(tt, tot):
            return 1.0 - tt / tot
    env = np.array([float(envelope(int(tt), max_iter)) for tt in t])

    fig, ax = plt.subplots(figsize=(11, 6))
    for i in range(n_traces):
        rng = np.random.default_rng(seed + i)
        series = np.array([float(law(int(tt), max_iter, rng)) for tt in t])
        ax.plot(t, series, lw=0.8, color=colors[i % len(colors)], alpha=0.95)

    ax.axhline(1.0, color="blue", ls="-.", lw=1.3)
    ax.axhline(-1.0, color="blue", ls="-.", lw=1.3)
    ax.plot(t, env, color="black", ls=":", lw=1.4)
    ax.plot(t, -env, color="black", ls=":", lw=1.4)

    ax.set_xlabel("iteration")
    ax.set_ylabel("Exploration coefficient")
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlim(0, max_iter)
    ax.set_ylim(-ylim, ylim)

    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

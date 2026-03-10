"""
Algorithm Enhancement Advisor
==============================
Runs diagnostic benchmarks on any algorithm and provides:
  - Performance scores (exploration, exploitation, stability, etc.)
  - Detected weaknesses
  - Ranked enhancement suggestions with code snippets
  - Convergence & diversity plots
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Type, Tuple, List, Dict

from heurilab.core.functions import F1, F5, F9, F10, F11


# ═════════════════════════════════════════════════════════════════════
#  Diagnostic Test Suite
# ═════════════════════════════════════════════════════════════════════

_DIAG_TESTS = {
    "unimodal": {
        "name": "Unimodal (F1 Sphere)",
        "func": F1, "lb": -100, "ub": 100, "dim": 30, "optimum": 0.0,
    },
    "rosenbrock": {
        "name": "Valley (F5 Rosenbrock)",
        "func": F5, "lb": -30, "ub": 30, "dim": 30, "optimum": 0.0,
    },
    "multimodal": {
        "name": "Multimodal (F9 Rastrigin)",
        "func": F9, "lb": -5.12, "ub": 5.12, "dim": 30, "optimum": 0.0,
    },
    "ackley": {
        "name": "Multimodal (F10 Ackley)",
        "func": F10, "lb": -32, "ub": 32, "dim": 30, "optimum": 0.0,
    },
    "griewank": {
        "name": "Multimodal (F11 Griewank)",
        "func": F11, "lb": -600, "ub": 600, "dim": 30, "optimum": 0.0,
    },
    "highdim": {
        "name": "High-Dim (F1 Sphere, d=100)",
        "func": F1, "lb": -100, "ub": 100, "dim": 100, "optimum": 0.0,
    },
}

_N_DIAG_RUNS = 10
_POP_SIZE = 50
_MAX_ITER = 200


# ═════════════════════════════════════════════════════════════════════
#  Tracking wrapper — records all evaluated positions
# ═════════════════════════════════════════════════════════════════════

class _TrackingWrapper:
    """Wraps an objective function to record evaluated positions."""

    def __init__(self, func, pop_size):
        self.func = func
        self.pop_size = pop_size
        self.positions = []
        self._batch = []

    def __call__(self, x):
        self._batch.append(x.copy())
        if len(self._batch) == self.pop_size:
            self.positions.append(np.array(self._batch))
            self._batch = []
        return self.func(x)


# ═════════════════════════════════════════════════════════════════════
#  Metric Calculations
# ═════════════════════════════════════════════════════════════════════

def _calc_diversity(positions_per_gen: List[np.ndarray]) -> List[float]:
    """Population diversity (mean std across dimensions) per generation."""
    diversities = []
    for pop in positions_per_gen:
        if len(pop.shape) == 2 and pop.shape[0] > 1:
            diversities.append(float(np.mean(np.std(pop, axis=0))))
        else:
            diversities.append(0.0)
    return diversities


def _detect_stagnation(convergence: list, window: int = 10) -> int:
    """Return the iteration where stagnation begins (no improvement for `window` iters)."""
    conv = np.array(convergence)
    for i in range(len(conv) - window):
        segment = conv[i:i + window]
        if len(segment) > 0 and (segment[0] - segment[-1]) / (abs(segment[0]) + 1e-30) < 1e-8:
            return i
    return len(conv)


def _convergence_speed(convergence: list) -> float:
    """Fraction of total improvement achieved in first 25% of iterations."""
    conv = np.array(convergence)
    total_drop = conv[0] - conv[-1]
    if total_drop <= 0:
        return 0.0
    quarter = max(1, len(conv) // 4)
    early_drop = conv[0] - conv[quarter]
    return float(np.clip(early_drop / total_drop, 0, 1))


def _exploitation_score(best_fitness: float, optimum: float, search_range: float) -> float:
    """How close to the known optimum (0–100)."""
    error = abs(best_fitness - optimum)
    # Normalize by search range squared * dim (the worst-case fitness for Sphere-like)
    if error < 1e-10:
        return 100.0
    log_err = np.log10(error + 1e-30)
    # Score: 0 at error=1e6, 100 at error=1e-10
    score = np.clip((6 - log_err) / 16 * 100, 0, 100)
    return float(score)


# ═════════════════════════════════════════════════════════════════════
#  Enhancement Knowledge Base
# ═════════════════════════════════════════════════════════════════════

ENHANCEMENTS = [
    {
        "name": "Chaotic Inertia / Step-Size Decay",
        "fixes": ["exploitation", "convergence_speed"],
        "impact": 5,
        "description": (
            "Replace linear parameter decay with chaotic or cosine-based decay.\n"
            "This creates irregular exploration bursts that prevent premature convergence."
        ),
        "code": (
            "# Instead of: param = param_max - (param_max - param_min) * t / max_iter\n"
            "# Use chaotic cosine decay:\n"
            "param = 0.5 * (param_max + param_min) + \\\n"
            "        0.5 * (param_max - param_min) * np.cos(np.pi * t / max_iter)"
        ),
    },
    {
        "name": "Levy Flight Mutation",
        "fixes": ["local_optima_escape", "multimodal"],
        "impact": 5,
        "description": (
            "Add occasional long-range jumps using Levy flights.\n"
            "Levy flights produce mostly small steps with rare large jumps,\n"
            "ideal for escaping local optima on multimodal landscapes."
        ),
        "code": (
            "def levy_flight(dim, beta=1.5):\n"
            "    sigma_u = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2) /\n"
            "              (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)\n"
            "    u = np.random.randn(dim) * sigma_u\n"
            "    v = np.random.randn(dim)\n"
            "    return u / (np.abs(v) ** (1/beta))\n"
            "\n"
            "# Apply with small probability after position update:\n"
            "if np.random.rand() < 0.1:\n"
            "    X[i] += 0.01 * levy_flight(dim) * (X[i] - best)"
        ),
    },
    {
        "name": "Opposition-Based Learning (OBL)",
        "fixes": ["exploration", "local_optima_escape"],
        "impact": 4,
        "description": (
            "Periodically generate opposite solutions: X_opp = lb + ub - X.\n"
            "Keep whichever is better. Doubles search coverage by checking\n"
            "mirror positions in the search space."
        ),
        "code": (
            "# Apply every N iterations (e.g., every 10):\n"
            "if t % 10 == 0:\n"
            "    for i in range(pop_size):\n"
            "        X_opp = self.lb + self.ub - X[i]\n"
            "        X_opp = self._clip(X_opp)\n"
            "        fit_opp = self._eval(X_opp)\n"
            "        if fit_opp < fitness[i]:\n"
            "            X[i] = X_opp\n"
            "            fitness[i] = fit_opp"
        ),
    },
    {
        "name": "Stagnation Escape (Population Restart)",
        "fixes": ["stagnation", "local_optima_escape"],
        "impact": 4,
        "description": (
            "If no improvement for K iterations, reinitialize the worst portion\n"
            "of the population randomly. Keeps the best solutions while injecting\n"
            "fresh diversity."
        ),
        "code": (
            "# Track stagnation counter:\n"
            "if best_fit_new >= best_fit_old:\n"
            "    stag_count += 1\n"
            "else:\n"
            "    stag_count = 0\n"
            "\n"
            "# Restart worst 30% if stagnated for 15 iterations:\n"
            "if stag_count >= 15:\n"
            "    n_restart = int(0.3 * pop_size)\n"
            "    worst_idx = np.argsort(fitness)[-n_restart:]\n"
            "    X[worst_idx] = np.random.uniform(self.lb, self.ub, (n_restart, dim))\n"
            "    for i in worst_idx:\n"
            "        fitness[i] = self._eval(X[i])\n"
            "    stag_count = 0"
        ),
    },
    {
        "name": "Gaussian Local Search",
        "fixes": ["exploitation", "convergence_speed"],
        "impact": 4,
        "description": (
            "Apply small Gaussian perturbation around the best solution.\n"
            "Refines the best solution with fine-grained local search,\n"
            "improving exploitation in the final convergence phase."
        ),
        "code": (
            "# Apply local search around best every iteration:\n"
            "sigma = (self.ub - self.lb) * 0.01 * (1 - t/max_iter)\n"
            "x_local = best + np.random.randn(dim) * sigma\n"
            "x_local = self._clip(x_local)\n"
            "fit_local = self._eval(x_local)\n"
            "if fit_local < best_fit:\n"
            "    best = x_local.copy()\n"
            "    best_fit = fit_local"
        ),
    },
    {
        "name": "Adaptive Parameter Control",
        "fixes": ["exploration", "exploitation", "multimodal"],
        "impact": 3,
        "description": (
            "Track which parameter values produce improvements and bias sampling\n"
            "toward successful values. Automatically balances exploration and exploitation."
        ),
        "code": (
            "# Success-history adaptive parameter (SHADE-style):\n"
            "memory_F = [0.5] * H   # H = memory size (e.g., 5)\n"
            "memory_CR = [0.5] * H\n"
            "k = 0\n"
            "\n"
            "# Each generation, sample from memory:\n"
            "F_i = np.clip(np.random.standard_cauchy() * 0.1 + memory_F[k], 0, 1)\n"
            "CR_i = np.clip(np.random.randn() * 0.1 + memory_CR[k], 0, 1)\n"
            "\n"
            "# Update memory with successful values:\n"
            "if improvement:\n"
            "    memory_F[k] = F_i\n"
            "    memory_CR[k] = CR_i\n"
            "    k = (k + 1) % H"
        ),
    },
    {
        "name": "Elite Preservation with Perturbation",
        "fixes": ["stability", "exploitation"],
        "impact": 3,
        "description": (
            "Keep the top K solutions unchanged, but create perturbed copies.\n"
            "Ensures the best solutions survive while still generating diverse offspring."
        ),
        "code": (
            "# After position update, preserve top 3:\n"
            "elite_n = 3\n"
            "elite_idx = np.argsort(fitness)[:elite_n]\n"
            "elite_X = X[elite_idx].copy()\n"
            "elite_fit = fitness[elite_idx].copy()\n"
            "\n"
            "# ... (run algorithm update) ...\n"
            "\n"
            "# Restore elites:\n"
            "worst_idx = np.argsort(fitness)[-elite_n:]\n"
            "X[worst_idx] = elite_X\n"
            "fitness[worst_idx] = elite_fit"
        ),
    },
    {
        "name": "Dimensional Learning (Random Dimension Selection)",
        "fixes": ["highdim", "scalability"],
        "impact": 3,
        "description": (
            "In high dimensions, update only a random subset of dimensions per iteration.\n"
            "Reduces interference between dimensions and improves scalability."
        ),
        "code": (
            "# Update only random dimensions:\n"
            "n_update = max(1, dim // 3)  # Update 1/3 of dimensions\n"
            "dims_to_update = np.random.choice(dim, n_update, replace=False)\n"
            "X_new = X[i].copy()\n"
            "X_new[dims_to_update] = <your update logic for selected dims>\n"
            "X_new = self._clip(X_new)"
        ),
    },
    {
        "name": "Boundary Reflection (instead of Clipping)",
        "fixes": ["exploration", "boundary_bias"],
        "impact": 2,
        "description": (
            "When solutions go out of bounds, reflect them back instead of clipping.\n"
            "Clipping causes boundary accumulation; reflection preserves momentum."
        ),
        "code": (
            "def _reflect(self, x):\n"
            "    for d in range(self.dim):\n"
            "        while x[d] < self.lb[d] or x[d] > self.ub[d]:\n"
            "            if x[d] < self.lb[d]:\n"
            "                x[d] = 2 * self.lb[d] - x[d]\n"
            "            if x[d] > self.ub[d]:\n"
            "                x[d] = 2 * self.ub[d] - x[d]\n"
            "    return x"
        ),
    },
    {
        "name": "Cauchy Mutation for Best Solution",
        "fixes": ["local_optima_escape", "exploitation"],
        "impact": 3,
        "description": (
            "Apply Cauchy-distributed perturbation to the best solution.\n"
            "Cauchy has heavier tails than Gaussian, creating larger occasional jumps\n"
            "that help escape narrow local optima."
        ),
        "code": (
            "# Cauchy mutation on best:\n"
            "scale = (self.ub - self.lb) * 0.1 * (1 - t/max_iter)\n"
            "x_cauchy = best + np.random.standard_cauchy(dim) * scale\n"
            "x_cauchy = self._clip(x_cauchy)\n"
            "fit_cauchy = self._eval(x_cauchy)\n"
            "if fit_cauchy < best_fit:\n"
            "    best = x_cauchy.copy()\n"
            "    best_fit = fit_cauchy"
        ),
    },
]


# ═════════════════════════════════════════════════════════════════════
#  Main Diagnostic Runner
# ═════════════════════════════════════════════════════════════════════

def _run_diagnostic(algo_name: str, algo_class: Type, test_key: str,
                    test_cfg: dict, n_runs: int, pop_size: int,
                    max_iter: int) -> dict:
    """Run one diagnostic test and collect metrics."""
    results = {
        "fitnesses": [],
        "convergences": [],
        "diversities": [],
        "stagnation_points": [],
        "times": [],
    }

    for _ in range(n_runs):
        wrapper = _TrackingWrapper(test_cfg["func"], pop_size)
        algo = algo_class(
            pop_size=pop_size,
            dim=test_cfg["dim"],
            lb=test_cfg["lb"],
            ub=test_cfg["ub"],
            max_iter=max_iter,
            obj_func=wrapper,
        )

        t0 = time.time()
        _, best_fit, conv = algo.optimize()
        elapsed = time.time() - t0

        conv = list(conv)
        diversity = _calc_diversity(wrapper.positions)
        stag = _detect_stagnation(conv)

        results["fitnesses"].append(best_fit)
        results["convergences"].append(conv)
        results["diversities"].append(diversity)
        results["stagnation_points"].append(stag)
        results["times"].append(elapsed)

    return results


# ═════════════════════════════════════════════════════════════════════
#  Scoring
# ═════════════════════════════════════════════════════════════════════

def _compute_scores(all_results: Dict[str, dict]) -> dict:
    """Compute scores (0–100) from diagnostic results."""
    scores = {}

    # --- Exploitation: how close to optimum on unimodal (F1) ---
    uni = all_results["unimodal"]
    mean_fit = np.mean(uni["fitnesses"])
    scores["exploitation"] = float(np.clip(100 - np.log10(mean_fit + 1e-30) * 8, 0, 100))

    # --- Exploration: initial diversity maintenance ---
    divs = []
    for d_list in uni["diversities"]:
        if len(d_list) >= 2:
            divs.append(d_list[0])
    init_div = np.mean(divs) if divs else 0
    # Normalize: high diversity = good exploration
    scores["exploration"] = float(np.clip(init_div / 30 * 100, 0, 100))

    # --- Local optima escape: multimodal vs unimodal ratio ---
    multi = all_results["multimodal"]
    multi_mean = np.mean(multi["fitnesses"])
    # Rastrigin optimum is 0, good if < 50
    scores["local_optima_escape"] = float(np.clip(100 - np.log10(multi_mean + 1e-30) * 15, 0, 100))

    # --- Convergence speed: how fast in first 25% ---
    speeds = []
    for conv in uni["convergences"]:
        speeds.append(_convergence_speed(conv))
    scores["convergence_speed"] = float(np.clip(np.mean(speeds) * 100, 0, 100))

    # --- Stability: inverse of coefficient of variation ---
    fits = np.array(uni["fitnesses"])
    if np.mean(fits) > 0:
        cv = np.std(fits) / (np.mean(fits) + 1e-30)
        scores["stability"] = float(np.clip((1 - cv) * 100, 0, 100))
    else:
        scores["stability"] = 100.0

    # --- Stagnation resistance ---
    stags = uni["stagnation_points"]
    max_iter_used = len(uni["convergences"][0]) if uni["convergences"] else 200
    mean_stag = np.mean(stags)
    scores["stagnation"] = float(np.clip(mean_stag / max_iter_used * 100, 0, 100))

    # --- Scalability: high-dim performance ---
    hd = all_results["highdim"]
    hd_mean = np.mean(hd["fitnesses"])
    scores["scalability"] = float(np.clip(100 - np.log10(hd_mean + 1e-30) * 6, 0, 100))

    # --- Multimodal overall ---
    ackley_mean = np.mean(all_results["ackley"]["fitnesses"])
    griewank_mean = np.mean(all_results["griewank"]["fitnesses"])
    multi_avg = (multi_mean + ackley_mean + griewank_mean) / 3
    scores["multimodal"] = float(np.clip(100 - np.log10(multi_avg + 1e-30) * 12, 0, 100))

    # --- Valley navigation (Rosenbrock) ---
    rosen_mean = np.mean(all_results["rosenbrock"]["fitnesses"])
    scores["valley_navigation"] = float(np.clip(100 - np.log10(rosen_mean + 1e-30) * 10, 0, 100))

    # --- Overall ---
    weights = {
        "exploitation": 0.20, "exploration": 0.10, "local_optima_escape": 0.20,
        "convergence_speed": 0.10, "stability": 0.10, "stagnation": 0.10,
        "scalability": 0.10, "multimodal": 0.05, "valley_navigation": 0.05,
    }
    scores["overall"] = float(sum(scores[k] * weights[k] for k in weights))

    return scores


# ═════════════════════════════════════════════════════════════════════
#  Weakness Detection & Enhancement Mapping
# ═════════════════════════════════════════════════════════════════════

def _detect_weaknesses(scores: dict) -> List[dict]:
    """Detect weaknesses and map to enhancements."""
    weaknesses = []

    thresholds = {
        "exploitation":       ("Poor exploitation — cannot refine solutions well", 50),
        "exploration":        ("Weak exploration — population converges too fast", 50),
        "local_optima_escape":("Gets trapped in local optima on multimodal functions", 50),
        "convergence_speed":  ("Slow initial convergence", 40),
        "stability":          ("Unstable — high variance across runs", 60),
        "stagnation":         ("Early stagnation — stops improving too soon", 50),
        "scalability":        ("Struggles with high-dimensional problems", 40),
        "multimodal":         ("Poor performance on multimodal landscapes", 50),
        "valley_navigation":  ("Difficulty navigating narrow valleys (Rosenbrock)", 50),
    }

    for key, (desc, threshold) in thresholds.items():
        if scores.get(key, 100) < threshold:
            weaknesses.append({"key": key, "description": desc, "score": scores[key]})

    weaknesses.sort(key=lambda w: w["score"])
    return weaknesses


def _suggest_enhancements(weaknesses: List[dict]) -> List[dict]:
    """Match weaknesses to enhancement suggestions, ranked by impact."""
    weakness_keys = {w["key"] for w in weaknesses}
    suggestions = []
    seen = set()

    for enh in ENHANCEMENTS:
        overlap = weakness_keys & set(enh["fixes"])
        if overlap and enh["name"] not in seen:
            suggestions.append({**enh, "fixes_found": list(overlap)})
            seen.add(enh["name"])

    suggestions.sort(key=lambda e: -e["impact"])
    return suggestions


# ═════════════════════════════════════════════════════════════════════
#  Report Formatting
# ═════════════════════════════════════════════════════════════════════

def _score_icon(score: float) -> str:
    if score >= 70:
        return "Strong"
    elif score >= 50:
        return "OK"
    elif score >= 35:
        return "Weak"
    else:
        return "Poor"


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_report(algo_name: str, scores: dict, weaknesses: List[dict],
                   suggestions: List[dict], all_results: dict) -> str:
    """Build the full text report."""
    lines = []
    lines.append("")
    lines.append("=" * 62)
    lines.append(f"  ENHANCEMENT ADVISOR  —  {algo_name}")
    lines.append("=" * 62)

    # ── Scores ──
    lines.append("")
    lines.append("  PERFORMANCE PROFILE")
    lines.append("  " + "-" * 56)

    score_labels = [
        ("exploitation",        "Exploitation"),
        ("exploration",         "Exploration"),
        ("local_optima_escape", "Local Optima Escape"),
        ("convergence_speed",   "Convergence Speed"),
        ("stability",           "Stability"),
        ("stagnation",          "Stagnation Resist."),
        ("scalability",         "Scalability (d=100)"),
        ("multimodal",          "Multimodal"),
        ("valley_navigation",   "Valley Navigation"),
    ]

    for key, label in score_labels:
        s = scores.get(key, 0)
        icon = _score_icon(s)
        bar = _score_bar(s)
        lines.append(f"    {label:<22s} {s:5.1f}/100  {bar}  {icon}")

    s_overall = scores.get("overall", 0)
    lines.append("  " + "-" * 56)
    lines.append(f"    {'OVERALL':<22s} {s_overall:5.1f}/100  {_score_bar(s_overall)}  {_score_icon(s_overall)}")

    # ── Test Results Summary ──
    lines.append("")
    lines.append("  DIAGNOSTIC TEST RESULTS")
    lines.append("  " + "-" * 56)
    lines.append(f"    {'Test':<30s} {'Mean':>12s} {'Std':>12s}")
    for key, cfg in _DIAG_TESTS.items():
        res = all_results[key]
        mean_f = np.mean(res["fitnesses"])
        std_f = np.std(res["fitnesses"])
        lines.append(f"    {cfg['name']:<30s} {mean_f:>12.4e} {std_f:>12.4e}")

    # ── Strengths ──
    lines.append("")
    lines.append("  STRENGTHS")
    lines.append("  " + "-" * 56)
    strengths_found = False
    for key, label in score_labels:
        if scores.get(key, 0) >= 65:
            strengths_found = True
            lines.append(f"    + {label}: {scores[key]:.0f}/100")
    if not strengths_found:
        lines.append("    (no strong areas detected)")

    # ── Weaknesses ──
    lines.append("")
    lines.append("  WEAKNESSES")
    lines.append("  " + "-" * 56)
    if weaknesses:
        for w in weaknesses:
            lines.append(f"    - {w['description']} ({w['score']:.0f}/100)")
    else:
        lines.append("    (no major weaknesses detected — algorithm is well-balanced!)")

    # ── Stagnation Analysis ──
    lines.append("")
    lines.append("  PHASE ANALYSIS (Unimodal F1)")
    lines.append("  " + "-" * 56)
    uni_convs = all_results["unimodal"]["convergences"]
    if uni_convs:
        avg_conv = np.mean([c[:_MAX_ITER+1] for c in uni_convs
                            if len(c) >= _MAX_ITER+1], axis=0) \
                   if all(len(c) >= _MAX_ITER+1 for c in uni_convs) \
                   else np.mean([np.pad(c, (0, max(0, _MAX_ITER+1-len(c))),
                                        mode='edge')[:_MAX_ITER+1]
                                 for c in uni_convs], axis=0)
        total_drop = avg_conv[0] - avg_conv[-1]
        if total_drop > 0:
            q1 = _MAX_ITER // 4
            q2 = _MAX_ITER // 2
            q3 = 3 * _MAX_ITER // 4
            p1 = (avg_conv[0] - avg_conv[q1]) / total_drop * 100
            p2 = (avg_conv[q1] - avg_conv[q2]) / total_drop * 100
            p3 = (avg_conv[q2] - avg_conv[q3]) / total_drop * 100
            p4 = (avg_conv[q3] - avg_conv[-1]) / total_drop * 100
            labels = [
                (f"Iter 1–{q1}", p1, "Exploring" if p1 > 40 else "Slow start"),
                (f"Iter {q1}–{q2}", p2, "Active" if p2 > 15 else "Slowing"),
                (f"Iter {q2}–{q3}", p3, "Refining" if p3 > 10 else "Stagnating"),
                (f"Iter {q3}–{_MAX_ITER}", p4, "Converging" if p4 > 5 else "Stagnated"),
            ]
            for rng, pct, phase in labels:
                icon = ">>>" if pct > 30 else ">>" if pct > 10 else ">"
                lines.append(f"    {rng:<16s}  {pct:5.1f}% improvement  {icon} {phase}")

    # ── Enhancement Suggestions ──
    lines.append("")
    lines.append("  RECOMMENDED ENHANCEMENTS (ranked by impact)")
    lines.append("  " + "=" * 56)
    if suggestions:
        for i, sug in enumerate(suggestions, 1):
            stars = "*" * sug["impact"]
            fixes_str = ", ".join(sug["fixes_found"])
            lines.append("")
            lines.append(f"  [{i}] {sug['name']}")
            lines.append(f"      Impact: {stars}  |  Fixes: {fixes_str}")
            lines.append(f"      {sug['description']}")
            lines.append("")
            lines.append("      Code:")
            for code_line in sug["code"].split("\n"):
                lines.append(f"        {code_line}")
            lines.append("      " + "-" * 50)
    else:
        lines.append("    No enhancements needed — algorithm is performing well!")

    lines.append("")
    lines.append("=" * 62)
    lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════

def _plot_scores_radar(algo_name: str, scores: dict, output_dir: str):
    """Radar chart of performance scores."""
    labels = [
        "Exploitation", "Exploration", "Local Optima\nEscape",
        "Conv. Speed", "Stability", "Stagnation\nResist.",
        "Scalability", "Multimodal", "Valley Nav."
    ]
    keys = [
        "exploitation", "exploration", "local_optima_escape",
        "convergence_speed", "stability", "stagnation",
        "scalability", "multimodal", "valley_navigation",
    ]
    values = [scores.get(k, 0) for k in keys]
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="#4363d8")
    ax.plot(angles, values, "o-", linewidth=2, color="#4363d8")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=8)
    ax.set_title(f"{algo_name} — Performance Profile", size=14, pad=20, fontweight="bold")

    # Color zones
    for r, color, alpha in [(100, "#e6194b", 0.03), (70, "#f58231", 0.05),
                             (50, "#3cb44b", 0.07)]:
        circle = plt.Circle((0, 0), r, transform=ax.transData, fill=True,
                             color=color, alpha=alpha, zorder=0)

    path = os.path.join(output_dir, f"{algo_name}_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence_diagnostic(algo_name: str, all_results: dict, output_dir: str):
    """Convergence curves for all diagnostic tests."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (key, cfg) in enumerate(_DIAG_TESTS.items()):
        ax = axes[idx]
        convs = all_results[key]["convergences"]
        for c in convs:
            ax.plot(c, alpha=0.2, color="#4363d8", linewidth=0.8)
        # Mean
        min_len = min(len(c) for c in convs)
        avg = np.mean([c[:min_len] for c in convs], axis=0)
        ax.plot(avg, color="#e6194b", linewidth=2, label="Mean")
        ax.set_title(cfg["name"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{algo_name} — Convergence Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"{algo_name}_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diversity(algo_name: str, all_results: dict, output_dir: str):
    """Diversity curves for unimodal and multimodal tests."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [(axes[0], "unimodal", "Unimodal (F1)"),
                            (axes[1], "multimodal", "Multimodal (F9)")]:
        divs = all_results[key]["diversities"]
        for d in divs:
            ax.plot(d, alpha=0.2, color="#3cb44b", linewidth=0.8)
        if divs:
            min_len = min(len(d) for d in divs)
            avg = np.mean([d[:min_len] for d in divs], axis=0)
            ax.plot(avg, color="#e6194b", linewidth=2, label="Mean")
        ax.set_title(f"{title} — Population Diversity", fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Diversity (mean std)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{algo_name} — Diversity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, f"{algo_name}_diversity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_score_bar(algo_name: str, scores: dict, output_dir: str):
    """Horizontal bar chart of scores with color coding."""
    labels = [
        "Exploitation", "Exploration", "Local Optima Escape",
        "Convergence Speed", "Stability", "Stagnation Resist.",
        "Scalability", "Multimodal", "Valley Navigation",
    ]
    keys = [
        "exploitation", "exploration", "local_optima_escape",
        "convergence_speed", "stability", "stagnation",
        "scalability", "multimodal", "valley_navigation",
    ]
    values = [scores.get(k, 0) for k in keys]
    colors = ["#3cb44b" if v >= 65 else "#f58231" if v >= 45 else "#e6194b" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (0–100)", fontsize=11)
    ax.set_title(f"{algo_name} — Performance Scores", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=10, fontweight="bold")

    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{algo_name}_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#  Custom Benchmark Scoring & Reporting
# ═════════════════════════════════════════════════════════════════════

def _compute_custom_scores(all_results: dict, tests: dict) -> dict:
    """Compute scores from custom benchmark results."""
    scores = {}
    n_tests = len(all_results)
    keys = list(all_results.keys())

    # --- Per-function scores based on closeness to optimum ---
    func_scores = []
    for key in keys:
        res = all_results[key]
        optimum = tests[key].get("optimum", 0.0)
        mean_fit = np.mean(res["fitnesses"])
        error = abs(mean_fit - optimum)
        if error < 1e-10:
            s = 100.0
        else:
            log_err = np.log10(error + 1e-30)
            s = float(np.clip((6 - log_err) / 16 * 100, 0, 100))
        func_scores.append(s)

    # --- Exploitation: average quality on all functions ---
    scores["exploitation"] = float(np.mean(func_scores))

    # --- Exploration: average initial diversity across tests ---
    divs = []
    for key in keys:
        for d_list in all_results[key]["diversities"]:
            if len(d_list) >= 2:
                divs.append(d_list[0])
    init_div = np.mean(divs) if divs else 0
    scores["exploration"] = float(np.clip(init_div / 30 * 100, 0, 100))

    # --- Convergence speed: average across all tests ---
    speeds = []
    for key in keys:
        for conv in all_results[key]["convergences"]:
            speeds.append(_convergence_speed(conv))
    scores["convergence_speed"] = float(np.clip(np.mean(speeds) * 100, 0, 100))

    # --- Stability: average inverse CV across tests ---
    stabilities = []
    for key in keys:
        fits = np.array(all_results[key]["fitnesses"])
        if np.mean(np.abs(fits)) > 0:
            cv = np.std(fits) / (np.mean(np.abs(fits)) + 1e-30)
            stabilities.append(float(np.clip((1 - cv) * 100, 0, 100)))
        else:
            stabilities.append(100.0)
    scores["stability"] = float(np.mean(stabilities))

    # --- Stagnation resistance: average across tests ---
    stags = []
    for key in keys:
        res = all_results[key]
        max_iter_used = len(res["convergences"][0]) if res["convergences"] else 200
        mean_stag = np.mean(res["stagnation_points"])
        stags.append(mean_stag / max_iter_used)
    scores["stagnation"] = float(np.clip(np.mean(stags) * 100, 0, 100))

    # --- Local optima escape: score on multimodal/hybrid/composition if present ---
    scores["local_optima_escape"] = float(np.median(func_scores))

    # --- Scalability & multimodal: use same as exploitation for custom ---
    scores["scalability"] = scores["exploitation"]
    scores["multimodal"] = float(np.mean(func_scores[-max(1, n_tests // 3):]))
    scores["valley_navigation"] = float(np.mean(func_scores[:max(1, n_tests // 3)]))

    # --- Overall ---
    weights = {
        "exploitation": 0.20, "exploration": 0.10, "local_optima_escape": 0.20,
        "convergence_speed": 0.10, "stability": 0.10, "stagnation": 0.10,
        "scalability": 0.10, "multimodal": 0.05, "valley_navigation": 0.05,
    }
    scores["overall"] = float(sum(scores[k] * weights[k] for k in weights))

    return scores


def _format_custom_report(algo_name: str, scores: dict, weaknesses: list,
                          suggestions: list, all_results: dict,
                          tests: dict) -> str:
    """Build the full text report for custom benchmarks."""
    lines = []
    lines.append("")
    lines.append("=" * 62)
    lines.append(f"  ENHANCEMENT ADVISOR  —  {algo_name}")
    lines.append("=" * 62)

    # ── Scores ──
    lines.append("")
    lines.append("  PERFORMANCE PROFILE")
    lines.append("  " + "-" * 56)

    score_labels = [
        ("exploitation",        "Exploitation"),
        ("exploration",         "Exploration"),
        ("local_optima_escape", "Local Optima Escape"),
        ("convergence_speed",   "Convergence Speed"),
        ("stability",           "Stability"),
        ("stagnation",          "Stagnation Resist."),
        ("scalability",         "Scalability"),
        ("multimodal",          "Multimodal"),
        ("valley_navigation",   "Valley Navigation"),
    ]

    for key, label in score_labels:
        s = scores.get(key, 0)
        icon = _score_icon(s)
        bar = _score_bar(s)
        lines.append(f"    {label:<22s} {s:5.1f}/100  {bar}  {icon}")

    s_overall = scores.get("overall", 0)
    lines.append("  " + "-" * 56)
    lines.append(f"    {'OVERALL':<22s} {s_overall:5.1f}/100  {_score_bar(s_overall)}  {_score_icon(s_overall)}")

    # ── Test Results Summary ──
    lines.append("")
    lines.append("  BENCHMARK TEST RESULTS")
    lines.append("  " + "-" * 56)
    lines.append(f"    {'Function':<35s} {'Mean':>12s} {'Std':>12s}")
    for key in all_results:
        cfg = tests[key]
        res = all_results[key]
        mean_f = np.mean(res["fitnesses"])
        std_f = np.std(res["fitnesses"])
        name = cfg["name"]
        if len(name) > 35:
            name = name[:32] + "..."
        lines.append(f"    {name:<35s} {mean_f:>12.4e} {std_f:>12.4e}")

    # ── Strengths ──
    lines.append("")
    lines.append("  STRENGTHS")
    lines.append("  " + "-" * 56)
    strengths_found = False
    for key, label in score_labels:
        if scores.get(key, 0) >= 65:
            strengths_found = True
            lines.append(f"    + {label}: {scores[key]:.0f}/100")
    if not strengths_found:
        lines.append("    (no strong areas detected)")

    # ── Weaknesses ──
    lines.append("")
    lines.append("  WEAKNESSES")
    lines.append("  " + "-" * 56)
    if weaknesses:
        for w in weaknesses:
            lines.append(f"    - {w['description']} ({w['score']:.0f}/100)")
    else:
        lines.append("    (no major weaknesses detected — algorithm is well-balanced!)")

    # ── Enhancement Suggestions ──
    lines.append("")
    lines.append("  RECOMMENDED ENHANCEMENTS (ranked by impact)")
    lines.append("  " + "=" * 56)
    if suggestions:
        for i, sug in enumerate(suggestions, 1):
            stars = "*" * sug["impact"]
            fixes_str = ", ".join(sug["fixes_found"])
            lines.append("")
            lines.append(f"  [{i}] {sug['name']}")
            lines.append(f"      Impact: {stars}  |  Fixes: {fixes_str}")
            lines.append(f"      {sug['description']}")
            lines.append("")
            lines.append("      Code:")
            for code_line in sug["code"].split("\n"):
                lines.append(f"        {code_line}")
            lines.append("      " + "-" * 50)
    else:
        lines.append("    No enhancements needed — algorithm is performing well!")

    lines.append("")
    lines.append("=" * 62)
    lines.append("")

    return "\n".join(lines)


def _plot_custom_convergence(algo_name: str, all_results: dict, tests: dict,
                             output_dir: str):
    """Convergence curves for custom benchmark tests."""
    keys = list(all_results.keys())
    n = len(keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        convs = all_results[key]["convergences"]
        for c in convs:
            ax.plot(c, alpha=0.2, color="#4363d8", linewidth=0.8)
        min_len = min(len(c) for c in convs)
        avg = np.mean([c[:min_len] for c in convs], axis=0)
        ax.plot(avg, color="#e6194b", linewidth=2, label="Mean")
        short = tests[key]["name"].replace("CEC17_", "")
        ax.set_title(short, fontsize=8, fontweight="bold")
        ax.set_xlabel("Iter", fontsize=7)
        ax.set_ylabel("Fitness", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{algo_name} — CEC 2017 Convergence", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"{algo_name}_convergence_cec2017.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_custom_diversity(algo_name: str, all_results: dict, tests: dict,
                           output_dir: str):
    """Diversity curves for custom benchmark tests (first 4)."""
    keys = list(all_results.keys())[:4]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        divs = all_results[key]["diversities"]
        for d in divs:
            ax.plot(d, alpha=0.2, color="#3cb44b", linewidth=0.8)
        if divs:
            min_len = min(len(d) for d in divs)
            avg = np.mean([d[:min_len] for d in divs], axis=0)
            ax.plot(avg, color="#e6194b", linewidth=2, label="Mean")
        short = tests[key]["name"].replace("CEC17_", "")
        ax.set_title(f"{short} — Diversity", fontsize=9, fontweight="bold")
        ax.set_xlabel("Generation", fontsize=8)
        ax.set_ylabel("Diversity", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{algo_name} — Diversity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, f"{algo_name}_diversity_cec2017.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#  Main Public API
# ═════════════════════════════════════════════════════════════════════

def enhance(
    algorithm: Tuple[str, Type],
    output_dir: str = "enhance_report",
    pop_size: int = _POP_SIZE,
    max_iter: int = _MAX_ITER,
    n_runs: int = _N_DIAG_RUNS,
    benchmarks: dict = None,
):
    """
    Run diagnostic benchmarks and provide enhancement suggestions.

    Parameters
    ----------
    algorithm : tuple (name, AlgorithmClass)
        The algorithm to analyze.
    output_dir : str
        Directory to save report and plots.
    pop_size : int
        Population size for diagnostic runs.
    max_iter : int
        Max iterations for diagnostic runs.
    n_runs : int
        Number of runs per diagnostic test.
    benchmarks : dict, optional
        Custom benchmark tests. Each key maps to a dict with keys:
        'name', 'func', 'lb', 'ub', 'dim', 'optimum'.
        If None, uses the built-in 6-function diagnostic suite.

    Returns
    -------
    dict with keys: scores, weaknesses, suggestions, report_text
    """
    algo_name, algo_class = algorithm
    os.makedirs(output_dir, exist_ok=True)

    tests = benchmarks if benchmarks is not None else _DIAG_TESTS
    custom_mode = benchmarks is not None

    print(f"\n{'=' * 62}")
    print(f"  Running Enhancement Advisor for: {algo_name}")
    print(f"  Pop={pop_size}, MaxIter={max_iter}, Runs={n_runs}")
    print(f"  Tests: {len(tests)} benchmarks" + (" (custom)" if custom_mode else ""))
    print(f"{'=' * 62}\n")

    # ── Run all diagnostic tests ──
    all_results = {}
    for i, (key, cfg) in enumerate(tests.items(), 1):
        print(f"  [{i}/{len(tests)}] {cfg['name']} ...", end=" ", flush=True)
        results = _run_diagnostic(algo_name, algo_class, key, cfg,
                                  n_runs, pop_size, max_iter)
        all_results[key] = results
        mean_f = np.mean(results["fitnesses"])
        print(f"mean={mean_f:.4e}")

    # ── Compute scores ──
    if custom_mode:
        scores = _compute_custom_scores(all_results, tests)
    else:
        scores = _compute_scores(all_results)

    # ── Detect weaknesses ──
    weaknesses = _detect_weaknesses(scores)

    # ── Suggest enhancements ──
    suggestions = _suggest_enhancements(weaknesses)

    # ── Generate report ──
    if custom_mode:
        report_text = _format_custom_report(algo_name, scores, weaknesses,
                                            suggestions, all_results, tests)
    else:
        report_text = _format_report(algo_name, scores, weaknesses, suggestions, all_results)
    print(report_text)

    # ── Save report ──
    report_path = os.path.join(output_dir, f"{algo_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── Generate plots ──
    print("  Generating plots...", flush=True)
    _plot_score_bar(algo_name, scores, output_dir)
    _plot_custom_convergence(algo_name, all_results, tests, output_dir) if custom_mode \
        else _plot_convergence_diagnostic(algo_name, all_results, output_dir)
    _plot_custom_diversity(algo_name, all_results, tests, output_dir) if custom_mode \
        else _plot_diversity(algo_name, all_results, output_dir)
    _plot_scores_radar(algo_name, scores, output_dir)

    print(f"\n  Report saved to: {report_path}")
    print(f"  Plots saved to:  {output_dir}/")
    print(f"{'=' * 62}\n")

    return {
        "scores": scores,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "report_text": report_text,
    }

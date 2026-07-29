"""
Deriving a structural feature vector from an algorithm implementation.

The source study coded its 145 feature vectors by hand from published
descriptions. HeuriLab has something better available: the implementations
themselves. This module derives the vector from an algorithm class in two ways.

**Runtime probing.** The algorithm is run once on an instrumented objective that
records every point it evaluates, with its random generator wrapped to count
draws. Several criteria are directly observable in that trace: whether more than
one candidate is maintained per iteration, how the initial population is
distributed, whether the search space is continuous or discrete, whether
randomness enters the update at all, and whether members are restarted mid-run.

**Source analysis.** The remaining criteria describe *how the code is written* -
whether it combines parent solutions, which selection rule it uses, whether it
keeps a memory - and are matched against the class source.

Every bit carries a confidence and a short evidence string, so a low-confidence
detection can be found and corrected rather than silently trusted.

.. warning::
   Detection is a heuristic, and the source-matched criteria in particular can be
   wrong on unusual implementations. What makes the resulting distances
   meaningful is *consistency*: the same detector scores the reference set and
   the algorithm under test, so systematic error largely cancels. Always read the
   evidence before quoting a verdict, and use ``overrides`` to correct a bit you
   know to be wrong.

Criteria are scored against the framework of Soto Calvo & Lee (2026); see
:mod:`heurilab.taxonomy.criteria`.
"""

import inspect
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Type

import numpy as np

from heurilab.taxonomy.criteria import CRITERIA, CRITERION_IDS


# ═════════════════════════════════════════════════════════════════════
#  Feature vector
# ═════════════════════════════════════════════════════════════════════

@dataclass
class FeatureVector:
    """A 19-bit structural fingerprint with provenance for each bit."""
    name: str
    bits: Dict[str, int] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)
    probe_failed: bool = False

    def as_array(self) -> np.ndarray:
        return np.array([self.bits[c] for c in CRITERION_IDS], dtype=int)

    def as_string(self) -> str:
        return "".join(str(self.bits[c]) for c in CRITERION_IDS)

    def low_confidence(self, below: float = 0.7):
        """Criterion ids whose detection should be reviewed by a human."""
        return [c for c in CRITERION_IDS if self.confidence.get(c, 0.0) < below]

    def explain(self) -> str:
        lines = [f"Structural feature vector - {self.name}",
                 f"  {self.as_string()}", ""]
        for c in CRITERIA:
            bit = self.bits[c.id]
            conf = self.confidence.get(c.id, 0.0)
            mark = "yes" if bit else "no "
            flag = "  <-- review" if conf < 0.7 else ""
            lines.append(f"  {c.id:>3} {mark}  conf={conf:.2f}  {c.name}{flag}")
            lines.append(f"        {self.evidence.get(c.id, '')}")
        return "\n".join(lines)

    def __repr__(self):
        return f"FeatureVector('{self.name}', {self.as_string()})"


# ═════════════════════════════════════════════════════════════════════
#  Runtime instrumentation
# ═════════════════════════════════════════════════════════════════════

class _RecordingObjective:
    """Sphere objective that records every point it is asked to evaluate."""

    def __init__(self, dim):
        self.dim = dim
        self.points = []

    def __call__(self, x):
        arr = np.asarray(x, dtype=float).ravel()
        self.points.append(arr.copy())
        return float(np.sum(arr ** 2))


class _CountingRNG:
    """Proxy around a numpy Generator that counts how often it is drawn from."""

    def __init__(self, rng):
        object.__setattr__(self, "_rng", rng)
        object.__setattr__(self, "draws", 0)

    def __getattr__(self, item):
        attr = getattr(object.__getattribute__(self, "_rng"), item)
        if callable(attr):
            def counted(*args, **kwargs):
                object.__setattr__(self, "draws",
                                   object.__getattribute__(self, "draws") + 1)
                return attr(*args, **kwargs)
            return counted
        return attr


def _probe(algo_class: Type, dim: int, pop_size: int, max_iter: int, seed: int):
    """Run the algorithm once on an instrumented objective."""
    obj = _RecordingObjective(dim)
    algo = algo_class(pop_size=pop_size, dim=dim, lb=-100.0, ub=100.0,
                      max_iter=max_iter, obj_func=obj, seed=seed)
    counting = _CountingRNG(algo.rng)
    algo.rng = counting
    _, _, convergence = algo.optimize()
    return obj, counting, len(list(convergence))


# ═════════════════════════════════════════════════════════════════════
#  Runtime-observable criteria
# ═════════════════════════════════════════════════════════════════════

def _runtime_bits(obj, counting, n_conv, pop_size, dim, lb=-100.0, ub=100.0):
    bits, conf, ev = {}, {}, {}
    pts = np.asarray(obj.points, dtype=float) if obj.points else np.zeros((0, dim))
    n_evals = len(pts)
    span = ub - lb

    # ── C1 population-based ──────────────────────────────────────────
    per_iter = n_evals / max(n_conv - 1, 1)
    bits["C1"] = int(per_iter > 1.5)
    conf["C1"] = 0.95
    ev["C1"] = f"{n_evals} evaluations over {n_conv - 1} iterations ({per_iter:.1f}/iter)"

    # ── C2 / C4 initialization shape ─────────────────────────────────
    # The opening batch is the initial population.
    init = pts[:pop_size] if n_evals >= pop_size else pts
    if len(init) >= 4:
        coverage = float(np.mean((init.max(axis=0) - init.min(axis=0)) / span))
        centred = np.clip((init - lb) / span, 0.0, 1.0)
        # Mean of a uniform sample is 0.5 and its std is 1/sqrt(12) ~ 0.289.
        spread = float(np.mean(np.std(centred, axis=0)))
        uniform_like = coverage > 0.55 and spread > 0.20
        bits["C2"] = int(uniform_like)
        bits["C4"] = int(not uniform_like)
        conf["C2"] = conf["C4"] = 0.85
        ev["C2"] = f"initial coverage {coverage:.2f} of the box, dispersion {spread:.2f}"
        ev["C4"] = ("initial sample is not uniform" if not uniform_like
                    else "initial sample is uniform, so no shaped distribution")
    else:
        bits["C2"], bits["C4"] = 1, 0
        conf["C2"] = conf["C4"] = 0.3
        ev["C2"] = ev["C4"] = "too few initial evaluations to characterise"

    # ── C5 / C6 representation ───────────────────────────────────────
    if n_evals:
        fractional = float(np.mean(np.abs(pts - np.round(pts)) > 1e-9))
        bits["C5"] = int(fractional > 0.05)
        bits["C6"] = int(fractional <= 0.05)
        conf["C5"] = conf["C6"] = 0.9
        ev["C5"] = f"{fractional:.0%} of evaluated coordinates are non-integral"
        ev["C6"] = ev["C5"]
    else:
        bits["C5"], bits["C6"] = 1, 0
        conf["C5"] = conf["C6"] = 0.2
        ev["C5"] = ev["C6"] = "no evaluations recorded"

    # ── C7 stochastic perturbation ───────────────────────────────────
    draws_per_eval = counting.draws / max(n_evals, 1)
    bits["C7"] = int(counting.draws > pop_size)   # beyond initialization alone
    conf["C7"] = 0.9
    ev["C7"] = f"{counting.draws} generator draws ({draws_per_eval:.2f} per evaluation)"

    # ── C10 solution reinitialization ────────────────────────────────
    # A restart shows up as a late point drawn from across the whole box while
    # the rest of the search has contracted around the incumbent.
    if n_evals > 4 * pop_size:
        early = pts[:pop_size]
        late = pts[len(pts) // 2:]
        early_spread = float(np.mean(early.max(axis=0) - early.min(axis=0)))
        late_spread = float(np.mean(late.max(axis=0) - late.min(axis=0)))
        # Points in the last quarter that sit far from the running centroid.
        tail = pts[-max(pop_size, len(pts) // 4):]
        centroid = np.median(pts[len(pts) // 2:], axis=0)
        far = float(np.mean(np.linalg.norm(tail - centroid, axis=1)
                            > 0.35 * span * np.sqrt(dim)))
        restarts = far > 0.05 and late_spread > 0.30 * early_spread
        bits["C10"] = int(restarts)
        conf["C10"] = 0.6
        ev["C10"] = (f"{far:.0%} of late evaluations lie far from the search centroid; "
                     f"late spread is {late_spread / max(early_spread, 1e-9):.2f} of initial")
    else:
        bits["C10"] = 0
        conf["C10"] = 0.4
        ev["C10"] = "run too short to observe restarts"

    # ── C13 elitist selection ────────────────────────────────────────
    # Elite preservation shows up as sustained resampling near the incumbent.
    if n_evals > 3 * pop_size:
        best_idx = int(np.argmin(np.sum(pts ** 2, axis=1)))
        best_pt = pts[best_idx]
        tail = pts[len(pts) // 2:]
        near = float(np.mean(np.linalg.norm(tail - best_pt, axis=1)
                             < 0.10 * span * np.sqrt(dim)))
        bits["C13"] = int(near > 0.15)
        conf["C13"] = 0.6
        ev["C13"] = f"{near:.0%} of later evaluations lie close to the incumbent best"
    else:
        bits["C13"] = 1
        conf["C13"] = 0.3
        ev["C13"] = "run too short to observe elite retention"

    return bits, conf, ev


# ═════════════════════════════════════════════════════════════════════
#  Source-matched criteria
# ═════════════════════════════════════════════════════════════════════

#: (criterion, regex, confidence, human-readable meaning of a match)
_SOURCE_RULES = [
    ("C8", r"runge|kutta|\bK1\b|gradient|reflect|expansion|contraction|"
           r"newton|simplex|deterministic",
     0.7, "systematic, non-random update rule"),
    ("C9", r"crossover|\bCR\b|\btrial\b|mutant|offspring|parent|"
           r"X\[\w+\]\s*[-+]\s*X\[\w+\]|recombin",
     0.8, "combines components of two or more solutions"),
    ("C11", r"p\s*=\s*probs|probs\s*=|roulette|fitness_?prob|"
            r"/\s*np\.sum\(fit",
     0.7, "selection probability proportional to fitness"),
    ("C12", r"argsort|rankdata|tournament|sorted_idx|np\.sort|\brank",
     0.8, "selection driven by ordering rather than magnitude"),
    ("C14", r"np\.std\(|diversity|adaptive|self\.?adapt|update.*param|"
            r"success_rate|mean_?CR|mean_?F",
     0.6, "parameters respond to population state"),
    ("C15", r"subpop|island|migrat|\bgroup|complex(es)?\b|partition|tribe|clan",
     0.6, "population split into exchanging subpopulations"),
    ("C18", r"alpha|beta|delta|employed|onlooker|scout|leader|follower|"
            r"producer|scrounger|explor\w+.*exploit|elite_?size|\bhalf\b",
     0.65, "distinct functional roles within the population"),
    ("C19", r"archive|memory|history|tabu|\bM_CR\b|\bM_F\b|success_?hist|"
            r"previous|record",
     0.7, "explicit memory of past search experience"),
]

#: A position update whose right-hand side references another agent - global
#: best, a leader, a neighbour, a pheromone trail or another population member.
#: Matching a bare ``best`` would fire on every implementation, because almost
#: all of them track an incumbent for bookkeeping without being coordinated.
_COORDINATION_UPDATE = re.compile(
    r"^\s*(?:new_?X\w*|X\[[^\]]+\]|pos\w*|V\b|velocit\w*)\s*(?:\[[^\]]*\])?\s*[+\-*/]?=\s*"
    r"[^\n=]*\b(?:best|gbest|leader|elite|pheromone|centroid|mean_?pos|"
    r"X\[\w*(?:rand|r1|r2|r3|j|k)\w*\])",
    re.I | re.M)

_MULTI_NEIGHBOURHOOD = re.compile(
    r"neighbou?rhood_?(list|set|structures)|k_max|shaking|vns|"
    r"multiple.*neighbou?rhood", re.I)


def _class_source(algo_class: Type) -> str:
    """Source of the class and every base up to ``_Base``, concatenated.

    A user who subclasses an existing optimizer and overrides one method should
    be analysed on the code that actually runs, so inherited behaviour has to be
    included. Classes defined interactively have no retrievable source; those
    contribute nothing rather than raising.
    """
    chunks = []
    for cls in inspect.getmro(algo_class):
        if cls.__name__ in ("_Base", "object"):
            break
        try:
            chunks.append(inspect.getsource(cls))
        except (OSError, TypeError):
            continue
    return "\n".join(chunks)


def _source_bits(algo_class: Type):
    bits, conf, ev = {}, {}, {}
    src = _class_source(algo_class)
    if not src:
        for cid, _, _, _ in _SOURCE_RULES:
            bits[cid], conf[cid], ev[cid] = 0, 0.1, "source unavailable"
        for cid, default in (("C3", 0), ("C16", 1), ("C17", 0)):
            bits[cid], conf[cid], ev[cid] = default, 0.1, "source unavailable"
        return bits, conf, ev

    for cid, pattern, confidence, meaning in _SOURCE_RULES:
        match = re.search(pattern, src, re.I)
        bits[cid] = int(bool(match))
        conf[cid] = confidence
        ev[cid] = (f"matched {match.group(0)!r} - {meaning}" if match
                   else f"no evidence of: {meaning}")

    # C17 multi-agent coordination: an agent's update must be driven by another
    # agent's position, not merely by tracking a global incumbent.
    coord = _COORDINATION_UPDATE.search(src)
    bits["C17"] = int(bool(coord))
    conf["C17"] = 0.75
    ev["C17"] = (f"position update references another agent: "
                 f"{coord.group(0).strip()[:70]!r}" if coord
                 else "updates do not reference other agents' positions")

    # C16 single neighbourhood: true unless the code alternates structures.
    multi = _MULTI_NEIGHBOURHOOD.search(src)
    bits["C16"] = int(not multi)
    conf["C16"] = 0.7
    ev["C16"] = ("alternates between neighbourhood structures" if multi
                 else "one unified neighbourhood structure")

    # C3 strategic oversampling is a design intent that rarely leaves a
    # syntactic trace; default to absent and flag it for human review.
    oversample = re.search(r"oversampl|promising_?region|biased_?init|"
                           r"seeded_?init|heuristic_?init", src, re.I)
    bits["C3"] = int(bool(oversample))
    conf["C3"] = 0.5 if oversample else 0.45
    ev["C3"] = ("initialization is biased toward selected regions" if oversample
                else "no sign of biased initialization (low-confidence default)")
    return bits, conf, ev


# ═════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════

def detect_features(name: str,
                    algo_class: Type,
                    dim: int = 10,
                    pop_size: int = 20,
                    max_iter: int = 40,
                    seed: int = 20260728,
                    overrides: Optional[Dict[str, int]] = None) -> FeatureVector:
    """
    Derive the 19-bit structural fingerprint of an algorithm.

    Parameters
    ----------
    name : str
        Label used in reports.
    algo_class : type
        A ``_Base`` subclass implementing ``optimize()``.
    dim, pop_size, max_iter, seed : int
        Settings for the instrumented probe run. Defaults are large enough to
        observe restarts and elite retention while staying fast.
    overrides : dict, optional
        Manual corrections, e.g. ``{"C14": 1}``. Applied last, with full
        confidence, and recorded as such in the evidence.

    Returns
    -------
    FeatureVector

    Notes
    -----
    Review ``vector.low_confidence()`` before quoting a verdict; those bits are
    the ones the detector is least sure about.
    """
    vec = FeatureVector(name=name)

    try:
        obj, counting, n_conv = _probe(algo_class, dim, pop_size, max_iter, seed)
        r_bits, r_conf, r_ev = _runtime_bits(obj, counting, n_conv, pop_size, dim)
    except Exception as exc:                       # noqa: BLE001 - report, don't crash
        vec.probe_failed = True
        r_bits = {c: 0 for c in ("C1", "C2", "C4", "C5", "C6", "C7", "C10", "C13")}
        r_bits.update({"C1": 1, "C2": 1, "C5": 1, "C7": 1, "C13": 1})
        r_conf = {c: 0.1 for c in r_bits}
        r_ev = {c: f"probe failed ({type(exc).__name__}: {exc})" for c in r_bits}

    s_bits, s_conf, s_ev = _source_bits(algo_class)

    vec.bits.update(r_bits); vec.bits.update(s_bits)
    vec.confidence.update(r_conf); vec.confidence.update(s_conf)
    vec.evidence.update(r_ev); vec.evidence.update(s_ev)

    for cid in CRITERION_IDS:
        vec.bits.setdefault(cid, 0)
        vec.confidence.setdefault(cid, 0.1)
        vec.evidence.setdefault(cid, "not determined")

    if overrides:
        unknown = set(overrides) - set(CRITERION_IDS)
        if unknown:
            raise ValueError(f"unknown criterion id(s) in overrides: {sorted(unknown)}")
        for cid, bit in overrides.items():
            vec.bits[cid] = int(bool(bit))
            vec.confidence[cid] = 1.0
            vec.evidence[cid] = "set manually via overrides"

    return vec


def detect_many(algorithms: Sequence, **kw) -> Dict[str, FeatureVector]:
    """Fingerprint a list of ``(name, class)`` pairs. Failures are kept, not dropped."""
    return {name: detect_features(name, cls, **kw) for name, cls in algorithms}

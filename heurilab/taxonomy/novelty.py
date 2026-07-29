"""
Novelty assessment for a proposed algorithm.

The question this answers is the one a reviewer will ask: *is this new, or is it
an existing algorithm wearing a different metaphor?* The proposed algorithm is
fingerprinted on the nineteen structural criteria, compared against a reference
set (HeuriLab's registry by default), and placed in one of the regimes the
source study identified from its distance distribution.

Usage::

    from heurilab.taxonomy import check_novelty

    report = check_novelty(("MYALGO", MyAlgorithm))
    print(report)
    report.save("novelty_report")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from heurilab.taxonomy.criteria import (
    CRITERIA, CRITERIA_BY_ID, CRITERION_IDS, DISTINCTION_THRESHOLD,
    REDUNDANCY_THRESHOLD, SOURCE, VERDICTS,
)
from heurilab.taxonomy.detect import FeatureVector, detect_features
from heurilab.taxonomy.distance import rogers_tanimoto


@dataclass
class NoveltyReport:
    """Outcome of assessing one algorithm against a reference set."""
    name: str
    vector: FeatureVector
    neighbours: List[Tuple[str, float]]        # (name, distance), nearest first
    verdict: str                               # key of criteria.VERDICTS
    reference_size: int
    reference_label: str = "HeuriLab registry"
    distance_stats: Dict[str, float] = field(default_factory=dict)
    differing_criteria: Dict[str, List[str]] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────
    @property
    def nearest_name(self) -> str:
        return self.neighbours[0][0] if self.neighbours else ""

    @property
    def nearest_distance(self) -> float:
        return self.neighbours[0][1] if self.neighbours else float("nan")

    @property
    def is_novel(self) -> bool:
        """True when the algorithm is not a structural duplicate of a known one."""
        return self.verdict in ("normal", "distinct")

    def explanation(self) -> str:
        return VERDICTS[self.verdict]

    # ── rendering ────────────────────────────────────────────────────
    def __str__(self) -> str:
        w = 74
        lines = [
            "=" * w,
            f"  STRUCTURAL NOVELTY ASSESSMENT - {self.name}",
            "=" * w,
            "",
            f"  Feature vector : {self.vector.as_string()}",
            f"  Reference set  : {self.reference_label} ({self.reference_size} algorithms)",
            "",
            f"  VERDICT: {self.verdict.upper()}",
            f"  {self.explanation()}",
            "",
            f"  Nearest known algorithm: {self.nearest_name} "
            f"(distance {self.nearest_distance:.4f})",
            "",
            "  Thresholds (Soto Calvo & Lee, 2026):",
            f"    < {REDUNDANCY_THRESHOLD:.3f}          structurally redundant",
            f"    {REDUNDANCY_THRESHOLD:.3f} - {DISTINCTION_THRESHOLD:.3f}   normal algorithmic difference",
            f"    > {DISTINCTION_THRESHOLD:.3f}          fundamentally different approach",
            "",
            "-" * w,
            "  CLOSEST KNOWN ALGORITHMS",
            "-" * w,
        ]
        for rank, (nm, dist) in enumerate(self.neighbours, 1):
            diff = self.differing_criteria.get(nm, [])
            note = ("identical on all 19 criteria" if not diff
                    else "differs on " + ", ".join(diff))
            lines.append(f"  {rank}. {nm:<12} d={dist:.4f}   {note}")

        if self.distance_stats:
            lines += ["", "-" * w, "  POSITION IN THE REFERENCE DISTRIBUTION", "-" * w,
                      f"  mean distance to all references : {self.distance_stats['mean']:.4f}",
                      f"  median                          : {self.distance_stats['median']:.4f}",
                      f"  references closer than {REDUNDANCY_THRESHOLD:.3f}     : "
                      f"{self.distance_stats['n_redundant']:.0f}"]

        review = self.vector.low_confidence()
        if review:
            lines += ["", "-" * w, "  CRITERIA TO REVIEW BY HAND", "-" * w,
                      "  The detector is least certain about these. Correct any that are",
                      "  wrong with overrides={...} and re-run before quoting the verdict.",
                      ""]
            for cid in review:
                c = CRITERIA_BY_ID[cid]
                lines.append(f"    {cid} {c.name}: detected {self.vector.bits[cid]} "
                             f"(confidence {self.vector.confidence[cid]:.2f})")
                lines.append(f"       {self.vector.evidence[cid]}")

        lines += ["", "-" * w,
                  "  Criteria framework: " + SOURCE, "=" * w]
        return "\n".join(lines)

    def save(self, output_dir: str, plots: bool = True) -> str:
        """Write the report, and optionally the figures, to a directory."""
        from heurilab.taxonomy.report import save_novelty_report
        return save_novelty_report(self, output_dir, plots=plots)


# ═════════════════════════════════════════════════════════════════════

def _classify(distance: float) -> str:
    if distance == 0.0:
        return "identical"
    if distance < REDUNDANCY_THRESHOLD:
        return "redundant"
    if distance > DISTINCTION_THRESHOLD:
        return "distinct"
    if distance < 0.15:
        return "incremental"
    return "normal"


def _differences(a: Sequence[int], b: Sequence[int]) -> List[str]:
    return [cid for cid, x, y in zip(CRITERION_IDS, a, b) if x != y]


def check_novelty(algorithm,
                  reference: Optional[Dict[str, FeatureVector]] = None,
                  k: int = 5,
                  overrides: Optional[Dict[str, int]] = None,
                  reference_label: str = None,
                  **detect_kwargs) -> NoveltyReport:
    """
    Assess how structurally novel an algorithm is.

    Parameters
    ----------
    algorithm : (str, type) or FeatureVector
        The algorithm to assess. Pass ``(name, AlgorithmClass)`` for a
        ``_Base`` subclass, or a pre-computed :class:`FeatureVector` if you
        scored the criteria by hand.
    reference : dict, optional
        ``{name: FeatureVector}`` to compare against. Defaults to HeuriLab's
        registry, computed once and cached.
    k : int
        Number of nearest neighbours to report.
    overrides : dict, optional
        Manual criterion corrections for the algorithm under test, e.g.
        ``{"C14": 1, "C19": 0}``.
    reference_label : str, optional
        Label for the reference set, used in the report header.
    **detect_kwargs
        Passed to :func:`~heurilab.taxonomy.detect.detect_features`.

    Returns
    -------
    NoveltyReport

    Examples
    --------
    >>> from heurilab.taxonomy import check_novelty
    >>> from heurilab.algorithms import GWO
    >>> report = check_novelty(("GWO-copy", GWO))
    >>> report.verdict
    'identical'
    """
    if isinstance(algorithm, FeatureVector):
        vector = algorithm
        if overrides:
            for cid, bit in overrides.items():
                vector.bits[cid] = int(bool(bit))
                vector.confidence[cid] = 1.0
                vector.evidence[cid] = "set manually via overrides"
    else:
        name, cls = algorithm
        vector = detect_features(name, cls, overrides=overrides, **detect_kwargs)

    if reference is None:
        from heurilab.taxonomy import reference_vectors
        reference = reference_vectors()
        reference_label = reference_label or "HeuriLab registry"
    reference_label = reference_label or "custom reference set"

    target = vector.as_array()
    scored = []
    for nm, ref_vec in reference.items():
        if nm == vector.name:
            continue                                  # never compare with itself
        scored.append((nm, rogers_tanimoto(target, ref_vec.as_array())))
    if not scored:
        raise ValueError("reference set is empty after excluding the algorithm itself")
    scored.sort(key=lambda pair: (pair[1], pair[0]))

    all_d = np.array([d for _, d in scored], dtype=float)
    stats = {
        "mean": float(all_d.mean()),
        "median": float(np.median(all_d)),
        "min": float(all_d.min()),
        "n_redundant": float(np.sum(all_d < REDUNDANCY_THRESHOLD)),
    }

    neighbours = scored[:k]
    diffs = {nm: _differences(target, reference[nm].as_array())
             for nm, _ in neighbours}

    return NoveltyReport(
        name=vector.name, vector=vector, neighbours=neighbours,
        verdict=_classify(neighbours[0][1]), reference_size=len(scored),
        reference_label=reference_label, distance_stats=stats,
        differing_criteria=diffs,
    )


def compare(a, b, **detect_kwargs) -> Tuple[float, List[str]]:
    """
    Distance between two algorithms and the criteria on which they differ.

    Each argument is ``(name, class)`` or a :class:`FeatureVector`.

    >>> from heurilab.algorithms import WOA, GWO
    >>> d, differing = compare(("WOA", WOA), ("GWO", GWO))
    """
    def _vec(x):
        if isinstance(x, FeatureVector):
            return x
        return detect_features(x[0], x[1], **detect_kwargs)

    va, vb = _vec(a), _vec(b)
    return (rogers_tanimoto(va.as_array(), vb.as_array()),
            _differences(va.as_array(), vb.as_array()))

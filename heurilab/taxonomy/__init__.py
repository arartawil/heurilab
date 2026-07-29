"""
Structural taxonomy and novelty checking.

Answers the question a reviewer asks about any proposed metaheuristic: *is this
actually new, or an existing algorithm under a different metaphor?*

An algorithm is fingerprinted on nineteen metaphor-free structural criteria,
compared against a reference set with the Rogers-Tanimoto distance, and placed in
one of three regimes drawn from the source study's distance distribution.

Implements the framework of:

    Soto Calvo, M. & Lee, H. S. (2026). "Systematic taxonomic framework of
    metaheuristic algorithms using hierarchical clustering and structural
    criteria: how novel is the novelty?" *Artificial Intelligence Review*,
    59(61). https://doi.org/10.1007/s10462-025-11456-8

Check your own algorithm::

    from heurilab.taxonomy import check_novelty

    report = check_novelty(("MYALGO", MyAlgorithm))
    print(report)
    report.save("novelty_out")

Audit a whole library::

    from heurilab.taxonomy import taxonomy_report
    taxonomy_report("taxonomy_out")

Compare two algorithms directly::

    from heurilab.taxonomy import compare
    from heurilab.algorithms import WOA, GWO
    distance, differing_criteria = compare(("WOA", WOA), ("GWO", GWO))
"""

from typing import Dict, Optional, Sequence

from heurilab.taxonomy.criteria import (
    CRITERIA, CRITERIA_BY_ID, CRITERION_IDS, DISTINCTION_THRESHOLD, FAMILIES,
    REDUNDANCY_THRESHOLD, SOURCE, VERDICTS, Criterion, describe,
)
from heurilab.taxonomy.detect import (
    FeatureVector, detect_features, detect_many,
)
from heurilab.taxonomy.distance import (
    condensed_distances, distance_matrix, nearest, rogers_tanimoto,
)
from heurilab.taxonomy.cluster import (
    Taxonomy, build_taxonomy, criterion_frequencies, pca,
)
from heurilab.taxonomy.novelty import NoveltyReport, check_novelty, compare
from heurilab.taxonomy.profile import (
    AlgorithmProfile, budget_fairness, profile_algorithm, profile_many, profile_table,
)

__all__ = [
    # criteria
    "CRITERIA", "CRITERIA_BY_ID", "CRITERION_IDS", "FAMILIES", "Criterion",
    "describe", "SOURCE", "VERDICTS",
    "REDUNDANCY_THRESHOLD", "DISTINCTION_THRESHOLD",
    # detection
    "FeatureVector", "detect_features", "detect_many",
    # distance
    "rogers_tanimoto", "distance_matrix", "condensed_distances", "nearest",
    # clustering
    "Taxonomy", "build_taxonomy", "pca", "criterion_frequencies",
    # novelty
    "NoveltyReport", "check_novelty", "compare",
    # cost profiling
    "AlgorithmProfile", "profile_algorithm", "profile_many", "profile_table",
    "budget_fairness",
    # convenience
    "reference_vectors", "clear_reference_cache", "taxonomy_report",
]

# ── Cached reference set ─────────────────────────────────────────────
# Fingerprinting the whole registry means running every algorithm once, so the
# result is computed on first use and reused afterwards.

_REFERENCE_CACHE: Optional[Dict[str, FeatureVector]] = None


def reference_vectors(algorithms: Sequence = None,
                      refresh: bool = False,
                      **detect_kwargs) -> Dict[str, FeatureVector]:
    """
    Structural fingerprints of the reference algorithm set.

    Defaults to every algorithm in ``heurilab.algorithms.ALL_ALGORITHMS``,
    computed once per session and cached.

    Parameters
    ----------
    algorithms : sequence of (name, class), optional
        Use a custom reference set instead of the registry. Not cached.
    refresh : bool
        Recompute the cached registry fingerprints.
    **detect_kwargs
        Passed through to :func:`detect_features`.
    """
    global _REFERENCE_CACHE
    if algorithms is not None:
        return detect_many(algorithms, **detect_kwargs)
    if _REFERENCE_CACHE is None or refresh:
        from heurilab.algorithms import ALL_ALGORITHMS
        _REFERENCE_CACHE = detect_many(ALL_ALGORITHMS, **detect_kwargs)
    return _REFERENCE_CACHE


def clear_reference_cache():
    """Drop the cached registry fingerprints."""
    global _REFERENCE_CACHE
    _REFERENCE_CACHE = None


def taxonomy_report(output_dir: str = "taxonomy_report",
                    algorithms: Sequence = None,
                    plots: bool = True,
                    **detect_kwargs):
    """
    Cluster a whole algorithm set and write the report and figures.

    Parameters
    ----------
    output_dir : str
        Directory for ``taxonomy_report.md`` plus the dendrogram, distance
        heatmap, PCA biplot and criterion-loading figures.
    algorithms : sequence of (name, class), optional
        Defaults to HeuriLab's registry.
    plots : bool
        Set ``False`` to write only the markdown.

    Returns
    -------
    (Taxonomy, str)
        The taxonomy object and the path to the markdown report.
    """
    from heurilab.taxonomy.report import save_taxonomy_report

    vectors = reference_vectors(algorithms=algorithms, **detect_kwargs)
    names = list(vectors)
    tax = build_taxonomy(names, [vectors[n].as_array() for n in names])
    path = save_taxonomy_report(tax, output_dir, vectors=vectors, plots=plots)
    return tax, path

"""
Structural taxonomy and novelty checking.

Two things need locking down. The metric and clustering are exact mathematics
and are tested as such. The *detector* is heuristic, so the tests assert the
properties that make it usable rather than specific bit patterns: it must be
deterministic, self-consistent, and it must return distance zero when an
algorithm is compared against itself.
"""
import os

import numpy as np
import pytest

from heurilab.algorithms import GWO, PSO, WOA, DE, HS, RUN, GBO
from heurilab.taxonomy import (
    CRITERIA, CRITERION_IDS, DISTINCTION_THRESHOLD, REDUNDANCY_THRESHOLD,
    build_taxonomy, check_novelty, compare, criterion_frequencies,
    detect_features, distance_matrix, pca, rogers_tanimoto,
)
from heurilab.taxonomy.criteria import SOURCE


# ── criteria ─────────────────────────────────────────────────────────

def test_exactly_nineteen_criteria_with_unique_ids():
    assert len(CRITERIA) == 19
    assert CRITERION_IDS == [f"C{i}" for i in range(1, 20)]
    assert len({c.id for c in CRITERIA}) == 19


def test_criteria_are_documented_and_attributed():
    for c in CRITERIA:
        assert c.description and c.positive_examples and c.negative_examples
        assert c.detection in ("runtime", "source", "manual")
    assert "10.1007/s10462-025-11456-8" in SOURCE


def test_thresholds_match_the_published_confidence_interval():
    assert REDUNDANCY_THRESHOLD == pytest.approx(0.040)
    assert DISTINCTION_THRESHOLD == pytest.approx(0.680)


# ── distance metric ──────────────────────────────────────────────────

def test_identical_vectors_are_at_distance_zero():
    v = [1, 0, 1, 1, 0] * 3 + [1, 0, 1, 1]
    assert rogers_tanimoto(v, v) == 0.0


def test_distance_is_symmetric_and_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(0, 2, 19)
        b = rng.integers(0, 2, 19)
        d = rogers_tanimoto(a, b)
        assert d >= 0.0
        assert d == pytest.approx(rogers_tanimoto(b, a))


def test_distance_grows_with_disagreement():
    base = np.zeros(19, dtype=int)
    prev = 0.0
    for k in (1, 3, 6, 12, 19):
        other = base.copy()
        other[:k] = 1
        d = rogers_tanimoto(base, other)
        assert d > prev
        prev = d


def test_maximum_distance_is_reached_on_total_disagreement():
    a = np.zeros(19, dtype=int)
    b = np.ones(19, dtype=int)
    # b + c = 19, a + d = 0  ->  19/19 + 19/38 = 1.5
    assert rogers_tanimoto(a, b) == pytest.approx(1.5)


def test_distance_rejects_malformed_input():
    with pytest.raises(ValueError, match="length mismatch"):
        rogers_tanimoto([1, 0, 1], [1, 0])
    with pytest.raises(ValueError, match="binary"):
        rogers_tanimoto([1, 2, 0], [1, 0, 0])


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    vectors = rng.integers(0, 2, (12, 19))
    d = distance_matrix(vectors)
    assert d.shape == (12, 12)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0)


# ── detection ────────────────────────────────────────────────────────

def test_detection_produces_a_complete_binary_vector():
    v = detect_features("PSO", PSO)
    assert set(v.bits) == set(CRITERION_IDS)
    assert all(b in (0, 1) for b in v.bits.values())
    assert v.as_array().shape == (19,)
    assert len(v.as_string()) == 19
    assert all(0.0 <= c <= 1.0 for c in v.confidence.values())
    assert all(v.evidence[c] for c in CRITERION_IDS)


def test_detection_is_deterministic():
    """Same algorithm, same settings, same vector - otherwise distances drift."""
    a = detect_features("GWO", GWO)
    b = detect_features("GWO", GWO)
    assert a.as_string() == b.as_string()


def test_an_algorithm_is_at_distance_zero_from_itself():
    v = detect_features("WOA", WOA)
    assert rogers_tanimoto(v.as_array(), v.as_array()) == 0.0


def test_overrides_are_applied_with_full_confidence():
    v = detect_features("PSO", PSO, overrides={"C14": 1, "C19": 1})
    assert v.bits["C14"] == 1 and v.bits["C19"] == 1
    assert v.confidence["C14"] == 1.0
    assert "manually" in v.evidence["C14"]


def test_overrides_reject_unknown_criteria():
    with pytest.raises(ValueError, match="unknown criterion"):
        detect_features("PSO", PSO, overrides={"C99": 1})


def test_subclass_inherits_the_structure_of_its_parent():
    """A subclass adding nothing must fingerprint identically to its parent."""
    class Renamed(GWO):
        pass
    assert (detect_features("Renamed", Renamed).as_string()
            == detect_features("GWO", GWO).as_string())


def test_single_point_and_deterministic_methods_are_identified():
    """Sanity anchors: HS is the one single-solution method, RUN/GBO the
    two deterministic-update ones in the registry."""
    assert detect_features("HS", HS).bits["C1"] == 0
    assert detect_features("PSO", PSO).bits["C1"] == 1
    assert detect_features("RUN", RUN).bits["C8"] == 1
    assert detect_features("GBO", GBO).bits["C8"] == 1


def test_explain_mentions_every_criterion():
    text = detect_features("DE", DE).explain()
    for cid in CRITERION_IDS:
        assert cid in text


# ── clustering ───────────────────────────────────────────────────────

def _toy_set():
    rng = np.random.default_rng(7)
    a = np.tile([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], (8, 1))
    b = np.tile([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], (8, 1))
    a[rng.integers(0, 8, 3), rng.integers(0, 19, 3)] ^= 1
    b[rng.integers(0, 8, 3), rng.integers(0, 19, 3)] ^= 1
    names = [f"A{i}" for i in range(8)] + [f"B{i}" for i in range(8)]
    return names, np.vstack([a, b])


def test_clustering_recovers_two_well_separated_groups():
    names, vectors = _toy_set()
    tax = build_taxonomy(names, vectors, k_range=(2, 6))
    groups = tax.clusters()
    assert tax.silhouette > 0.5
    for members in groups.values():
        prefixes = {m[0] for m in members}
        assert len(prefixes) == 1, f"cluster mixes families: {members}"


def test_taxonomy_summary_is_self_consistent():
    names, vectors = _toy_set()
    tax = build_taxonomy(names, vectors, k_range=(2, 6))
    s = tax.summary()
    assert s["n_algorithms"] == 16
    assert s["n_pairs"] == 16 * 15 // 2
    assert s["min_distance"] <= s["median_distance"] <= s["max_distance"]
    assert s["q1"] <= s["q3"]
    assert 1 <= s["distinct_vectors"] <= 16


def test_identical_pairs_are_found():
    names = ["X", "Y", "Z"]
    vectors = [[1] * 19, [1] * 19, [0] * 19]
    tax = build_taxonomy(names, vectors, k_range=(2, 2))
    assert ("X", "Y") in tax.identical_pairs()
    assert all("Z" not in pair for pair in tax.identical_pairs())


def test_clustering_needs_at_least_three_algorithms():
    with pytest.raises(ValueError, match="at least three"):
        build_taxonomy(["A", "B"], [[1] * 19, [0] * 19])


def test_pca_shapes_and_variance():
    _, vectors = _toy_set()
    coords, loadings, explained = pca(vectors)
    assert coords.shape == (16, 2)
    assert loadings.shape == (19, 2)
    assert 0.0 <= explained.sum() <= 1.0 + 1e-9


def test_criterion_frequencies_flag_constant_columns():
    vectors = np.array([[1, 0, 1] + [0] * 16, [1, 1, 0] + [0] * 16])
    freq = criterion_frequencies(vectors)
    assert freq[0] == 1.0        # constant, carries no information
    assert freq[1] == 0.5


# ── novelty assessment ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_reference():
    from heurilab.taxonomy.detect import detect_many
    return detect_many([("PSO", PSO), ("GWO", GWO), ("WOA", WOA),
                        ("DE", DE), ("HS", HS), ("RUN", RUN)])


def test_a_renamed_algorithm_is_reported_as_identical(small_reference):
    class TotallyNewOptimizer(GWO):
        """A 'new' algorithm that is a straight copy of GWO."""
    report = check_novelty(("TotallyNewOptimizer", TotallyNewOptimizer),
                           reference=small_reference)
    assert report.nearest_distance == 0.0
    assert report.verdict == "identical"
    assert report.is_novel is False


def test_a_structurally_different_algorithm_is_not_flagged(small_reference):
    """A vector far from everything in the reference set must not read as a copy."""
    from heurilab.taxonomy.detect import FeatureVector
    exotic = FeatureVector(name="Exotic")
    for i, cid in enumerate(CRITERION_IDS):
        exotic.bits[cid] = int(i % 2 == 0)
        exotic.confidence[cid] = 1.0
        exotic.evidence[cid] = "hand-scored"
    report = check_novelty(exotic, reference=small_reference)
    assert report.nearest_distance > REDUNDANCY_THRESHOLD
    assert report.is_novel


def test_report_never_compares_an_algorithm_with_itself(small_reference):
    report = check_novelty(("GWO", GWO), reference=small_reference)
    assert all(name != "GWO" for name, _ in report.neighbours)
    assert report.reference_size == len(small_reference) - 1


def test_report_renders_and_names_the_source(small_reference):
    text = str(check_novelty(("PSO", PSO), reference=small_reference))
    assert "VERDICT" in text
    assert "10.1007/s10462-025-11456-8" in text
    assert "CLOSEST KNOWN ALGORITHMS" in text


def test_report_lists_the_criteria_that_differ(small_reference):
    report = check_novelty(("WOA", WOA), reference=small_reference)
    for name, dist in report.neighbours:
        diff = report.differing_criteria[name]
        assert (dist == 0.0) == (len(diff) == 0)
        assert all(c in CRITERION_IDS for c in diff)


def test_compare_returns_distance_and_differing_criteria():
    d, diff = compare(("WOA", WOA), ("GWO", GWO))
    assert 0.0 <= d <= 1.5
    assert all(c in CRITERION_IDS for c in diff)
    assert (d == 0.0) == (len(diff) == 0)


def test_compare_of_an_algorithm_with_itself_is_zero():
    d, diff = compare(("PSO", PSO), ("PSO", PSO))
    assert d == 0.0 and diff == []


def test_novelty_report_is_written_to_disk(tmp_path, small_reference):
    report = check_novelty(("PSO", PSO), reference=small_reference)
    path = report.save(str(tmp_path), plots=False)
    text = open(path, encoding="utf-8").read()
    assert "Structural novelty assessment" in text
    assert report.vector.as_string() in text


# ── cost profiling ───────────────────────────────────────────────────

def test_profiling_separates_evaluation_cost_from_implementation_cost():
    from heurilab.taxonomy import profile_algorithm
    p = profile_algorithm("PSO", PSO, dim=10, pop_size=10, max_iter=10,
                          expensive=False)
    assert p.evaluations > 0
    assert p.evals_per_iteration > 1.0        # PSO is population-based
    assert p.overhead_us_per_eval > 0.0
    assert 0.0 <= p.vectorization_ratio <= 1.0


def test_budget_fairness_detects_unequal_evaluation_budgets():
    """Equal max_iter does not mean equal search budget."""
    from heurilab.taxonomy import profile_many, budget_fairness
    profiles = profile_many([("PSO", PSO), ("HS", HS)],
                            dim=10, pop_size=10, max_iter=10, expensive=False)
    fairness = budget_fairness(profiles)
    assert fairness["unfairness_ratio"] > 1.5
    assert fairness["min_algorithm"] == "HS"


def test_profile_table_renders():
    from heurilab.taxonomy import profile_many, profile_table
    table = profile_table(profile_many([("PSO", PSO), ("DE", DE)],
                                       dim=10, pop_size=10, max_iter=8,
                                       expensive=False))
    assert "evals/iter" in table and "PSO" in table and "DE" in table


# ── exploration-coefficient traces ───────────────────────────────────

def test_exploration_trace_shapes_and_contraction():
    from heurilab.analyzer import exploration_trace
    tr = exploration_trace("GWO", GWO, dim=10, pop_size=15, max_iter=60)
    assert tr.coefficients.shape[0] == 2
    assert tr.coefficients.shape[1] == len(tr.iterations)
    assert len(tr.envelope) == len(tr.iterations)
    assert tr.contraction >= 0.0
    # GWO's step size decays by construction, so it must contract.
    assert tr.contracts


def test_exploration_trace_is_deterministic():
    from heurilab.analyzer import exploration_trace
    a = exploration_trace("PSO", PSO, dim=8, pop_size=12, max_iter=40)
    b = exploration_trace("PSO", PSO, dim=8, pop_size=12, max_iter=40)
    np.testing.assert_allclose(a.coefficients, b.coefficients)


def test_exploration_trace_accepts_a_custom_landscape():
    from heurilab.analyzer import exploration_trace
    calls = {"n": 0}

    def sphere(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2))

    exploration_trace("PSO", PSO, dim=6, pop_size=10, max_iter=30, obj_func=sphere)
    assert calls["n"] > 0


def test_exploration_trace_rejects_too_short_a_run():
    from heurilab.analyzer import exploration_trace
    with pytest.raises(ValueError, match="usable iteration blocks"):
        exploration_trace("PSO", PSO, dim=5, pop_size=10, max_iter=2)


def test_coefficient_plot_writes_a_figure(tmp_path):
    from heurilab.analyzer import coefficient_plot
    path = coefficient_plot("GWO", GWO, output_dir=str(tmp_path),
                            dim=8, pop_size=12, max_iter=50)
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_control_law_plot_writes_a_figure(tmp_path):
    from heurilab.analyzer import plot_control_law
    path = plot_control_law(lambda t, T, rng: (1 - t / T) * rng.normal(),
                            max_iter=200, output_dir=str(tmp_path),
                            title=r"$\Phi = (1 - t/T)\,\mathcal{N}(0,1)$")
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_linear_decay_envelope_falls_from_one_to_zero():
    from heurilab.analyzer import linear_decay_envelope
    env = linear_decay_envelope(100)
    assert len(env) == 100
    assert env[0] == pytest.approx(0.99)
    assert env[-1] == pytest.approx(0.0)
    assert np.all(np.diff(env) < 0)


# ── qualitative six-panel analysis ───────────────────────────────────

def _bench2d():
    from heurilab.core.benchmarks import BenchmarkConfig
    return BenchmarkConfig("Sphere2D",
                           lambda x: float(np.sum(np.asarray(x) ** 2)),
                           -100.0, 100.0, 2)


def test_qualitative_analysis_produces_every_panel():
    from heurilab.analyzer import qualitative_analysis
    a = qualitative_analysis(("GWO", GWO), _bench2d(), pop_size=20,
                             max_iter=60, resolution=12)
    T = len(a.convergence)
    assert a.grid_z.shape == (12, 12)
    assert a.search_history.shape[1] == 2 and len(a.search_history) > 0
    for series in (a.average_fitness, a.trajectory, a.exploration, a.exploitation):
        assert len(series) == T
    assert np.all(np.diff(a.convergence) <= 1e-12), "best-so-far must not worsen"


def test_exploration_and_exploitation_are_complementary_percentages():
    from heurilab.analyzer import qualitative_analysis
    a = qualitative_analysis(("PSO", PSO), _bench2d(), pop_size=20,
                             max_iter=60, resolution=10)
    assert a.population_based
    assert np.all(a.exploration >= -1e-9) and np.all(a.exploration <= 100 + 1e-9)
    np.testing.assert_allclose(a.exploration + a.exploitation, 100.0, atol=1e-6)
    assert a.exploration.max() == pytest.approx(100.0)   # peak diversity is the reference


def test_single_solution_methods_report_diversity_as_undefined():
    """Reporting 0% exploration for a one-candidate method would be an artifact."""
    from heurilab.analyzer import qualitative_analysis
    a = qualitative_analysis(("HS", HS), _bench2d(), pop_size=20,
                             max_iter=120, resolution=10)
    assert a.population_based is False
    assert np.all(np.isnan(a.exploration))
    assert a.crossover_iteration is None
    assert np.isnan(a.mean_exploration)


def test_exploratory_algorithms_score_higher_than_exploitative_ones():
    from heurilab.analyzer import qualitative_analysis
    from heurilab.algorithms import SCA
    sca = qualitative_analysis(("SCA", SCA), _bench2d(), pop_size=20,
                               max_iter=120, resolution=8)
    gwo = qualitative_analysis(("GWO", GWO), _bench2d(), pop_size=20,
                               max_iter=120, resolution=8)
    assert sca.mean_exploration > gwo.mean_exploration


@pytest.mark.parametrize("layout", ["row", "grid"])
def test_qualitative_figure_is_written(tmp_path, layout):
    from heurilab.analyzer import plot_qualitative, qualitative_analysis
    a = qualitative_analysis(("GWO", GWO), _bench2d(), pop_size=20,
                             max_iter=50, resolution=10)
    path = plot_qualitative(a, str(tmp_path), layout=layout)
    assert os.path.exists(path) and os.path.getsize(path) > 20000


def test_balance_table_marks_single_solution_methods():
    from heurilab.analyzer import balance_table, qualitative_analysis
    rows = [qualitative_analysis((n, c), _bench2d(), pop_size=20,
                                 max_iter=100, resolution=8)
            for n, c in [("GWO", GWO), ("HS", HS)]]
    table = balance_table(rows)
    assert "crossover iteration" in table
    assert "n/a" in table

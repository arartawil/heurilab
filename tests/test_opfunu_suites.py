"""
opfunu-backed CEC suites.

Skipped in full when opfunu is not installed — it is an optional dependency.
"""
import numpy as np
import pytest

opfunu = pytest.importorskip("opfunu", reason="opfunu is an optional dependency")

from heurilab.core.opfunu_suites import (          # noqa: E402
    OPFUNU_YEARS, get_opfunu_suite, list_opfunu_functions, get_opfunu_optimum,
    official_number, get_cec2022_opfunu_suite,
)
from heurilab.core.benchmarks import BenchmarkSuite   # noqa: E402


def test_every_advertised_edition_is_reachable():
    for year in OPFUNU_YEARS:
        nums = list_opfunu_functions(year)
        assert nums, f"CEC{year} exposed no functions"
        assert nums == sorted(nums)
        assert len(set(nums)) == len(nums), f"CEC{year} has duplicate numbers"


def test_cec2017_uses_official_numbering_not_opfunu_numbering():
    """Official CEC 2017 is F1 and F3-F30; opfunu packs them as F1..F29.

    If this regresses, every CEC 2017 results table is silently mislabelled:
    opfunu's F3 is Rosenbrock while the official F3 is Zakharov.
    """
    nums = list_opfunu_functions("2017")
    assert nums[0] == 1
    assert 2 not in nums, "F2 was withdrawn from CEC 2017 and must not appear"
    assert nums[-1] == 30
    assert len(nums) == 29
    assert official_number("2017", 1) == 1
    assert official_number("2017", 2) == 3
    assert official_number("2022", 3) == 3          # other editions unmapped


def test_cec2017_official_f3_is_zakharov():
    suite = get_opfunu_suite("2017", ndim=10, functions=[3], verbose=False)
    assert "Zakharov" in suite.benchmarks[0].obj_func._fn.name


@pytest.mark.parametrize("year,ndim,expected", [("2022", 10, 12), ("2020", 10, 10)])
def test_suite_sizes(year, ndim, expected):
    suite = get_opfunu_suite(year, ndim=ndim, verbose=False)
    assert isinstance(suite, BenchmarkSuite)
    assert len(suite) == expected
    assert all(b.dim == ndim for b in suite.benchmarks)


def test_objectives_are_finite_and_bounded():
    suite = get_cec2022_opfunu_suite(ndim=10, verbose=False)
    rng = np.random.default_rng(0)
    for b in suite.benchmarks:
        for x in rng.uniform(b.lb, b.ub, size=(5, b.dim)):
            assert np.isfinite(b.obj_func(x)), f"{b.name} produced a non-finite value"
        assert b.lb < b.ub


def test_known_optimum_is_attained_at_the_shift_vector():
    """f(x*) must equal the documented f_global, which validates the wrapper."""
    from opfunu.cec_based import cec2022
    for k in (1, 2, 3):
        fn = cec2022.__dict__[f"F{k}2022"](ndim=10)
        assert float(fn.evaluate(fn.x_global)) == pytest.approx(fn.f_global, rel=1e-6)
        assert get_opfunu_optimum("2022", k, 10) == pytest.approx(fn.f_global)


def test_function_subset_selection_and_bad_input():
    suite = get_opfunu_suite("2022", ndim=10, functions=[1, 5, 12], verbose=False)
    assert [b.name for b in suite.benchmarks] == ["F1", "F5", "F12"]
    with pytest.raises(ValueError, match="no function"):
        get_opfunu_suite("2022", ndim=10, functions=[99], verbose=False)
    with pytest.raises(ValueError, match="not available"):
        get_opfunu_suite("1999", ndim=10, verbose=False)


def test_unsupported_dimension_is_reported_not_silently_wrong():
    """CEC 2022 is defined for 2, 10 and 20 only."""
    with pytest.raises(ValueError, match="No CEC 2022 function supports"):
        get_opfunu_suite("2022", ndim=37, verbose=False)


def test_suite_is_picklable_for_parallel_runs():
    import pickle
    suite = get_opfunu_suite("2022", ndim=10, functions=[1, 2], verbose=False)
    restored = pickle.loads(pickle.dumps(suite))
    x = np.zeros(10)
    for a, b in zip(suite.benchmarks, restored.benchmarks):
        assert a.obj_func(x) == pytest.approx(b.obj_func(x))


def test_suite_runs_end_to_end(tmp_path):
    from heurilab import run_experiment
    from heurilab.algorithms import PSO, GWO
    out = tmp_path / "out"
    run_experiment(
        algorithms=[("PSO", PSO), ("GWO", GWO)],
        benchmark_suites=[get_opfunu_suite("2022", ndim=10, functions=[1, 2],
                                           verbose=False)],
        output_dir=str(out), pop_size=8, max_iter=5, n_runs=2,
        seed=7, run_engineering=False,
    )
    assert (out / "CSV Data" / "results.csv").exists()

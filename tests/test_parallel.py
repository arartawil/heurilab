"""
Parallel execution.

The property that matters is *equivalence*: with a seed set, ``n_jobs`` must
change only how long a campaign takes, never what it produces. Per-run seeds are
derived before any work is dispatched, so execution order cannot leak into the
results — these tests are what keep that true.
"""
import numpy as np
import pytest

pytest.importorskip("joblib", reason="joblib is an optional dependency")

from heurilab import run_experiment, get_unimodal_suite      # noqa: E402
from heurilab.algorithms import PSO, GWO                     # noqa: E402
from heurilab.core.runner import _execute_run, _make_executor  # noqa: E402


def _sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def _campaign(where, n_jobs, seed=2024):
    run_experiment(
        algorithms=[("PSO", PSO), ("GWO", GWO)],
        benchmark_suites=[get_unimodal_suite()],
        output_dir=str(where), pop_size=8, max_iter=5, dim=4, n_runs=3,
        seed=seed, n_jobs=n_jobs, run_engineering=False,
    )
    return (where / "CSV Data" / "results.csv").read_text()


def test_parallel_matches_serial_exactly(tmp_path):
    serial = _campaign(tmp_path / "serial", n_jobs=1)
    parallel = _campaign(tmp_path / "parallel", n_jobs=2)
    assert serial == parallel, "n_jobs changed the results - seeding is order-dependent"


def test_parallel_records_the_same_seeds_as_serial(tmp_path):
    def seeds(where, n_jobs):
        _campaign(where, n_jobs)
        lines = (where / "CSV Data" / "raw_runs.csv").read_text().splitlines()
        col = lines[0].split(",").index("Seed")
        return sorted(ln.split(",")[col] for ln in lines[1:])

    assert seeds(tmp_path / "a", 1) == seeds(tmp_path / "b", 2)


def test_all_cores_setting_is_accepted(tmp_path):
    out = tmp_path / "allcores"
    run_experiment(
        algorithms=[("PSO", PSO)], benchmark_suites=[get_unimodal_suite()],
        output_dir=str(out), pop_size=6, max_iter=4, dim=3, n_runs=2,
        seed=11, n_jobs=-1, run_engineering=False,
    )
    assert (out / "CSV Data" / "results.csv").exists()


def test_execute_run_contract():
    spec = dict(pop_size=8, dim=4, lb=-10.0, ub=10.0, max_iter=5, obj_func=_sphere)
    fit, conv, elapsed, n_fes = _execute_run(PSO, spec, 123)
    assert np.isfinite(fit)
    assert len(conv) == 6                       # max_iter + 1
    assert elapsed >= 0.0
    assert n_fes > 0                            # evaluations are reported back
    # Same seed -> same result, which is what makes the worker deterministic.
    assert _execute_run(PSO, spec, 123)[0] == fit
    assert _execute_run(PSO, spec, 124)[0] != fit


def test_serial_executor_is_lazy():
    """Serial mode must stream, so raw_runs.csv is written run by run."""
    import types
    spec = dict(pop_size=6, dim=3, lb=-5.0, ub=5.0, max_iter=3, obj_func=_sphere)
    result = _make_executor(1)(PSO, spec, [1, 2, 3])
    assert isinstance(result, types.GeneratorType)


def test_missing_joblib_raises_a_useful_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "joblib":
            raise ImportError("no joblib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError, match="pip install joblib"):
        _make_executor(4)


def test_engineering_runs_in_parallel(tmp_path):
    from heurilab.engineering.runner import run_engineering_problems
    out = tmp_path / "eng"
    run_engineering_problems("PSO", PSO, str(out), pop_size=8, max_iter=5,
                             n_runs=2, seed=5, n_jobs=2)
    assert out.exists() and any(out.rglob("*"))

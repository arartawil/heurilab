"""Core API tests: version, benchmark suites, and a tiny end-to-end run."""
import os

import numpy as np
import pytest

import heurilab
from heurilab import (
    get_classical_suite, get_unimodal_suite, get_multimodal_suite,
    get_fixeddim_suite, get_cec2017_suite,
)
from heurilab.core.benchmarks import BenchmarkSuite


def test_version_is_populated():
    assert isinstance(heurilab.__version__, str)
    assert heurilab.__version__.count(".") >= 2


@pytest.mark.parametrize("suite_fn,expected", [
    (get_classical_suite, 23),
    (get_unimodal_suite, 7),
    (get_multimodal_suite, 6),
    (get_fixeddim_suite, 10),
    (get_cec2017_suite, 29),
])
def test_suite_sizes(suite_fn, expected):
    suite = suite_fn()
    assert isinstance(suite, BenchmarkSuite)
    assert len(suite) == expected


def test_cec2020_available_via_core():
    from heurilab.core import get_cec2020_suite
    assert len(get_cec2020_suite()) == 10


def test_end_to_end_run(tmp_path):
    """run_experiment must produce CSVs without error on a tiny config."""
    np.random.seed(0)
    from heurilab import run_experiment
    from heurilab.algorithms import PSO, GWO

    out = tmp_path / "out"
    run_experiment(
        algorithms=[("PSO", PSO), ("GWO", GWO)],
        benchmark_suites=[get_unimodal_suite()],
        output_dir=str(out),
        pop_size=6, max_iter=5, dim=4, n_runs=2,
        run_engineering=False,
    )
    assert (out / "CSV Data" / "results.csv").exists()
    assert (out / "CSV Data" / "raw_runs.csv").exists()
    assert (out / "CSV Data" / "convergence.csv").exists()

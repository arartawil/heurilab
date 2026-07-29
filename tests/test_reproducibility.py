"""
Reproducibility contract.

Every algorithm must draw its randomness from ``self.rng`` (seeded via the
``seed`` argument) rather than the global ``numpy.random`` module. These tests
are what stop that regressing: a single stray ``np.random.*`` call in any
algorithm makes ``test_same_seed_is_deterministic`` fail for that algorithm.
"""
import re
import pathlib

import numpy as np
import pytest

from heurilab.algorithms import ALL_ALGORITHMS

POP, DIM, ITER = 10, 5, 6
LB, UB = -100.0, 100.0

# Shifted so that an algorithm collapsing to the origin cannot look optimal.
def shifted_sphere(x):
    return float(np.sum((np.asarray(x) - 1.234) ** 2))


def _run(cls, seed):
    algo = cls(pop_size=POP, dim=DIM, lb=LB, ub=UB, max_iter=ITER,
               obj_func=shifted_sphere, seed=seed)
    sol, fit, conv = algo.optimize()
    return np.asarray(sol, dtype=float), float(fit), np.asarray(list(conv), dtype=float)


@pytest.mark.parametrize("name,cls", ALL_ALGORITHMS, ids=[n for n, _ in ALL_ALGORITHMS])
def test_same_seed_is_deterministic(name, cls):
    """Identical seed -> byte-identical solution, fitness and convergence."""
    s1, f1, c1 = _run(cls, 4242)
    s2, f2, c2 = _run(cls, 4242)
    assert f1 == f2, f"{name}: fitness differs across identical seeds"
    np.testing.assert_array_equal(s1, s2, err_msg=f"{name}: solution differs")
    np.testing.assert_array_equal(c1, c2, err_msg=f"{name}: convergence differs")


@pytest.mark.parametrize("name,cls", ALL_ALGORITHMS, ids=[n for n, _ in ALL_ALGORITHMS])
def test_different_seeds_explore_differently(name, cls):
    """Different seeds must not produce an identical convergence trace.

    An algorithm that ignores its seed, or that collapses deterministically to
    a fixed point regardless of initialisation, is not searching.
    """
    _, _, c1 = _run(cls, 1)
    _, _, c2 = _run(cls, 987654)
    assert not np.array_equal(c1, c2), \
        f"{name}: identical convergence for different seeds - seed ignored or search degenerate"


def test_no_algorithm_touches_the_global_rng():
    """Static guard: no `np.random.*` calls anywhere in the algorithm package."""
    root = pathlib.Path(__file__).resolve().parents[1] / "heurilab" / "algorithms"
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path) or path.name == "base.py":
            continue  # base.py is the one place allowed to build the Generator
        hits = sorted(set(re.findall(r"np\.random\.\w+", path.read_text(encoding="utf-8"))))
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"global RNG used in: {offenders}"


def test_experiment_campaign_is_reproducible(tmp_path):
    """Two identically seeded run_experiment campaigns produce identical CSVs."""
    from heurilab import run_experiment, get_unimodal_suite
    from heurilab.algorithms import PSO, GWO

    def campaign(where):
        run_experiment(
            algorithms=[("PSO", PSO), ("GWO", GWO)],
            benchmark_suites=[get_unimodal_suite()],
            output_dir=str(where),
            pop_size=6, max_iter=4, dim=3, n_runs=2,
            seed=12345, run_engineering=False,
        )
        return (where / "CSV Data" / "results.csv").read_text()

    assert campaign(tmp_path / "a") == campaign(tmp_path / "b")


def test_raw_runs_csv_records_the_seed(tmp_path):
    """Each run's seed is written out, so any single run can be re-executed."""
    from heurilab import run_experiment, get_unimodal_suite
    from heurilab.algorithms import PSO

    out = tmp_path / "out"
    run_experiment(
        algorithms=[("PSO", PSO)], benchmark_suites=[get_unimodal_suite()],
        output_dir=str(out), pop_size=6, max_iter=4, dim=3, n_runs=2,
        seed=777, run_engineering=False,
    )
    lines = (out / "CSV Data" / "raw_runs.csv").read_text().splitlines()
    header = lines[0].split(",")
    assert "Seed" in header, "raw_runs.csv has no Seed column"
    col = header.index("Seed")
    seeds = [ln.split(",")[col] for ln in lines[1:]]
    assert all(s.strip() for s in seeds), "blank seed recorded despite seed= being set"
    assert len(set(seeds)) == len(seeds), "runs share a seed - they are not independent"

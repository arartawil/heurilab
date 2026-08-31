"""
Validity of HeuriLab's official CEC 2017 / CEC 2022 suites.

Four checks per (suite, function, dimension):

1. ``f(x*) == bias`` exactly.
2. ``scipy.optimize.differential_evolution`` inside the box never returns a
   value below the bias.
3. 200,000 uniform samples inside the box never fall below the bias.
4. The value agrees with the organisers' reference C code at 1000 identical
   points.

Every one of these is a hard assertion.  If ``heurilab.core.cec_official``
ever drifts from the reference, this module fails loudly and names the
function, the dimension and the size of the discrepancy.

Running it
----------
The whole matrix is slow (the 200k-sample sweep dominates).  It is worth
parallelising::

    pytest tests/test_cec_validity.py -n auto

Knobs:

``HEURILAB_CEC_SAMPLES``
    Uniform samples per function.  Default 200000 - lower it for a quick pass,
    but the number in the spec is 200000.
``HEURILAB_CEC_REQUIRE_REFERENCE=1``
    Fail instead of skipping when the reference C cannot be built here.
    See :mod:`tests.cec_reference` for the rest.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # the package under test
sys.path.insert(0, _HERE)                    # tests/cec_reference.py

from heurilab.core.cec_official import (          # noqa: E402
    CEC2017_FUNCTION_NUMBERS,
    CEC2022_FUNCTION_NUMBERS,
    OfficialCEC,
    official_bias,
    supported_dimensions,
)

import cec_reference                               # noqa: E402

pytest.importorskip("opfunu", reason="the official CEC data files ship with opfunu")

#: The dimensions the brief asks for.  CEC 2022 is only defined at 2, 10 and
#: 20, and CEC 2017 ships no D=20 data for most hybrids, so the matrix below
#: is filtered per function by `supported_dimensions`.
DIMENSIONS = (10, 20, 30, 50)

N_SAMPLES = int(os.environ.get("HEURILAB_CEC_SAMPLES", 200_000))
N_CROSSCHECK = 1000

#: Relative tolerance for the cross-check.  The reference C accumulates its
#: sums sequentially while NumPy sums pairwise, so agreement is limited by
#: summation order rather than by the formulae - a few ulps, never more.
CROSSCHECK_RTOL = 1e-11

#: Slack for the "never below the bias" checks - see BIAS_RESIDUE below for why
#: the floor is not enforced to the last bit.
FLOOR_ATOL = 1e-6

#: Cases where the *official code itself* does not return the bias exactly at
#: its own optimum, mapped to the residue it does return.
#:
#: CEC 2017 F10 is Schwefel, ``4.189828872724338e+2 * n - sum(z sin(sqrt|z|))``.
#: At D=50 the constant times 50 does not cancel the 50 summed terms to the
#: last bit and the reference C returns ``1000 + 1.8189894035458565e-11``.
#: That is a property of the official definition, not of this port: the value
#: below is the compiled reference's own output, and
#: `test_optimum_matches_reference_at_x_star` re-derives it from the C rather
#: than trusting this table.  Every other function returns its bias exactly.
BIAS_RESIDUE = {(2017, 10, 50): 1.8189894035458565e-11}

#: However far off the bias any function is allowed to be at its optimum.  A
#: genuine indexing or scaling bug misses by orders of magnitude, so this is a
#: floating-point allowance, not a loophole.
BIAS_RESIDUE_ATOL = 1e-9


def _cases(year, numbers):
    out = []
    for ndim in DIMENSIONS:
        for number in numbers:
            if ndim in supported_dimensions(year, number):
                out.append((year, number, ndim))
    return out


CASES = _cases(2017, CEC2017_FUNCTION_NUMBERS) + _cases(2022, CEC2022_FUNCTION_NUMBERS)
CASE_IDS = [f"CEC{y}-F{n}-D{d}" for y, n, d in CASES]

assert CASES, "no CEC cases were collected - is the opfunu data missing?"


# ── 0. the matrix itself ────────────────────────────────────────────────

def test_expected_functions_are_present():
    """Guard against a data or numbering regression silently shrinking the run."""
    have_2017 = {(n, d) for y, n, d in CASES if y == 2017}
    have_2022 = {(n, d) for y, n, d in CASES if y == 2022}

    # CEC 2017 defines F1 and F3..F30 and ships data for all of them at the
    # competition dimensions 10, 30 and 50.
    for ndim in (10, 30, 50):
        missing = [n for n in CEC2017_FUNCTION_NUMBERS if (n, ndim) not in have_2017]
        assert not missing, f"CEC2017 D{ndim} is missing F{missing}"
    # CEC 2022 defines F1..F12 at D=10 and D=20.
    for ndim in (10, 20):
        missing = [n for n in CEC2022_FUNCTION_NUMBERS if (n, ndim) not in have_2022]
        assert not missing, f"CEC2022 D{ndim} is missing F{missing}"


# ── 1. f(x*) == bias, exactly ───────────────────────────────────────────

@pytest.mark.parametrize("year,number,ndim", CASES, ids=CASE_IDS)
def test_optimum_hits_bias_exactly(year, number, ndim):
    fn = OfficialCEC(year, number, ndim)
    bias = official_bias(year, number)

    assert fn.f_global == bias
    assert np.all(fn.x_global >= -100.0) and np.all(fn.x_global <= 100.0), (
        f"CEC{year} F{number} D{ndim}: x* is outside the [-100, 100] box")

    value = fn.evaluate(fn.x_global)
    expected = bias + BIAS_RESIDUE.get((year, number, ndim), 0.0)
    assert value == expected, (
        f"CEC{year} F{number} D{ndim}: f(x*) = {value!r}, expected exactly "
        f"{expected!r} (difference {value - expected!r})")
    assert abs(value - bias) <= BIAS_RESIDUE_ATOL, (
        f"CEC{year} F{number} D{ndim}: f(x*) misses the bias {bias!r} by "
        f"{value - bias!r}, far more than floating-point error")


@pytest.mark.parametrize("year,number,ndim", CASES, ids=CASE_IDS)
def test_optimum_matches_reference_at_x_star(year, number, ndim):
    """Whatever f(x*) is, the organisers' C must return the same bits.

    This is what keeps `BIAS_RESIDUE` honest: the one non-exact entry is not an
    excuse baked into the test, it is the reference's own output.
    """
    reference = _reference_or_skip(year, number, ndim)
    fn = OfficialCEC(year, number, ndim)
    ours = fn.evaluate(fn.x_global)
    theirs = reference(year, number, ndim, [fn.x_global])[0]
    assert ours == theirs, (
        f"CEC{year} F{number} D{ndim}: f(x*) = {ours!r} but the organisers' "
        f"C returns {theirs!r}")


# ── 2. differential evolution never dips below the bias ─────────────────

@pytest.mark.parametrize("year,number,ndim", CASES, ids=CASE_IDS)
def test_differential_evolution_never_beats_the_bias(year, number, ndim):
    """No point in the box may score below f(x*).

    A bounded DE budget, not a convergence test: we only care that the search
    never *finds* something better than the documented optimum, which is what a
    mis-scaled or mis-shifted sub-function would let it do.
    """
    from scipy.optimize import differential_evolution

    fn = OfficialCEC(year, number, ndim)
    bias = official_bias(year, number)

    result = differential_evolution(
        fn.evaluate,
        bounds=[(-100.0, 100.0)] * ndim,
        strategy="best1bin",
        maxiter=60,
        popsize=12,
        tol=0.0,
        mutation=(0.5, 1.0),
        recombination=0.9,
        polish=False,
        init="sobol",
        seed=12345,
    )
    assert result.fun >= bias - FLOOR_ATOL, (
        f"CEC{year} F{number} D{ndim}: differential_evolution reached "
        f"{result.fun!r}, which is below the documented optimum {bias!r} "
        f"by {bias - result.fun!r}")


# ── 3. 200,000 uniform samples never dip below the bias ─────────────────

@pytest.mark.parametrize("year,number,ndim", CASES, ids=CASE_IDS)
def test_uniform_sampling_never_beats_the_bias(year, number, ndim):
    fn = OfficialCEC(year, number, ndim)
    bias = official_bias(year, number)
    sampler = np.random.default_rng((year * 100 + number) * 1000 + ndim)

    worst = np.inf
    worst_at = None
    remaining = N_SAMPLES
    while remaining > 0:
        block = min(remaining, 20_000)
        remaining -= block
        points = sampler.uniform(-100.0, 100.0, size=(block, ndim))
        for point in points:
            value = fn.evaluate(point)
            if value < worst:
                worst, worst_at = value, point
            if not np.isfinite(value):
                pytest.fail(f"CEC{year} F{number} D{ndim}: non-finite value "
                            f"{value!r} at {point!r}")

    assert worst >= bias - FLOOR_ATOL, (
        f"CEC{year} F{number} D{ndim}: {N_SAMPLES} uniform samples found "
        f"{worst!r}, below the documented optimum {bias!r} by "
        f"{bias - worst!r}, at x = {np.array2string(worst_at, precision=6)}")


# ── 4. agreement with the organisers' reference C ───────────────────────

def _reference_or_skip(year, number, ndim):
    """``reference_values``, or skip/fail if the organisers' C is unavailable."""
    try:
        cec_reference.prepare(year)
    except cec_reference.ReferenceUnavailable as exc:
        if os.environ.get("HEURILAB_CEC_REQUIRE_REFERENCE"):
            pytest.fail(f"HEURILAB_CEC_REQUIRE_REFERENCE is set but the "
                        f"CEC {year} reference could not be prepared: {exc}")
        pytest.skip(f"official CEC {year} reference unavailable: {exc}")

    if not cec_reference.reference_supports(year, number, ndim):
        pytest.skip(f"the organisers ship no CEC{year} F{number} D{ndim} data")
    return cec_reference.reference_values


@pytest.mark.parametrize("year,number,ndim", CASES, ids=CASE_IDS)
def test_matches_official_reference_code(year, number, ndim):
    reference = _reference_or_skip(year, number, ndim)

    fn = OfficialCEC(year, number, ndim)
    sampler = np.random.default_rng(0x5EED + year * 1000 + number * 10 + ndim)
    points = sampler.uniform(-100.0, 100.0, size=(N_CROSSCHECK, ndim))
    points[0] = fn.x_global                       # include the optimum itself

    ours = np.array([fn.evaluate(p) for p in points])
    theirs = reference(year, number, ndim, points)

    assert np.all(np.isfinite(ours)), f"CEC{year} F{number} D{ndim}: non-finite output"
    absolute = np.abs(ours - theirs)
    relative = absolute / np.maximum(1.0, np.abs(theirs))
    worst = int(np.argmax(relative))

    assert relative[worst] <= CROSSCHECK_RTOL, (
        f"CEC{year} F{number} D{ndim}: disagrees with the organisers' C code.\n"
        f"  max relative difference {relative[worst]:.3e} "
        f"(tolerance {CROSSCHECK_RTOL:.0e})\n"
        f"  max absolute difference {absolute.max():.6e}\n"
        f"  heurilab = {ours[worst]!r}\n"
        f"  official = {theirs[worst]!r}\n"
        f"  at x = {np.array2string(points[worst], precision=6)}")


# ── the shipped suites are wired to the verified implementation ─────────

def test_cec2022_suite_uses_the_corrected_functions():
    from heurilab.core.cec2022_fixed import get_cec2022_suite

    suite = get_cec2022_suite(ndim=10, verbose=False)
    assert [b.name for b in suite] == [f"F{n}" for n in CEC2022_FUNCTION_NUMBERS]
    for benchmark in suite:
        number = int(benchmark.name[1:])
        reference = OfficialCEC(2022, number, 10)
        assert benchmark.lb == -100.0 and benchmark.ub == 100.0
        assert benchmark.dim == 10
        assert benchmark.obj_func(reference.x_global) == official_bias(2022, number)


def test_cec2017_official_suite_uses_the_corrected_functions():
    from heurilab.core.cec2017_fixed import get_cec2017_official_suite

    suite = get_cec2017_official_suite(ndim=30, verbose=False)
    assert [b.name for b in suite] == [f"F{n}" for n in CEC2017_FUNCTION_NUMBERS]
    for benchmark in suite:
        number = int(benchmark.name[1:])
        reference = OfficialCEC(2017, number, 30)
        assert benchmark.obj_func(reference.x_global) == official_bias(2017, number)


def test_suite_objectives_survive_pickling():
    """run_experiment(n_jobs=...) ships the objective to worker processes."""
    import pickle

    from heurilab.core.cec2022_fixed import get_cec2022_suite

    suite = get_cec2022_suite(ndim=10, functions=[9, 10, 11, 12], verbose=False)
    for benchmark in suite:
        restored = pickle.loads(pickle.dumps(benchmark.obj_func))
        point = np.full(10, 12.5)
        assert restored(point) == benchmark.obj_func(point)


def test_fixed_classes_subclass_opfunu_and_override_evaluate():
    """The brief asks for subclasses of opfunu's F9-F12, not replacements."""
    import opfunu.cec_based.cec2022 as upstream

    from heurilab.core import cec2022_fixed

    for number in (9, 10, 11, 12):
        fixed = getattr(cec2022_fixed, f"CEC2022F{number}")
        base = getattr(upstream, f"F{number}2022")
        assert issubclass(fixed, base)
        assert fixed.evaluate is not base.evaluate

        instance = fixed(ndim=10)
        reference = OfficialCEC(2022, number, 10)
        point = np.linspace(-90.0, 90.0, 10)
        assert instance.evaluate(point) == reference.evaluate(point)
        # ...and the upstream version really is different, i.e. this is a fix.
        assert base(ndim=10).evaluate(point) != reference.evaluate(point)


def test_legacy_cec2017_module_warns_that_it_is_not_official():
    from heurilab.core import cec2017 as legacy

    with pytest.warns(UserWarning, match="not official CEC 2017"):
        legacy.get_cec2017_suite()
    with pytest.warns(UserWarning, match="not official CEC 2017"):
        legacy.get_cec2017_composition_suite()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        legacy.get_cec2017_suite(official=False)      # opt-out is honoured


def test_opfunu_route_warns_that_it_is_not_official():
    from heurilab.core import opfunu_suites

    with pytest.warns(UserWarning, match="do not match the organisers"):
        opfunu_suites.get_cec2017_opfunu_suite(ndim=30, verbose=False)
    with pytest.warns(UserWarning, match="do not match the organisers"):
        opfunu_suites.get_cec2022_opfunu_suite(ndim=10, verbose=False)

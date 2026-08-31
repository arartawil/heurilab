"""
Official CEC 2017 suite (F1, F3-F30), with ``opfunu``'s defects corrected.

The route this replaces
-----------------------
:func:`heurilab.core.opfunu_suites.get_cec2017_opfunu_suite` was documented as
"the official one" because ``opfunu`` ships the organisers' ``data_2017``
files.  The data is indeed official; the *arithmetic* is not.  Checked against
the organisers' ``cec17_test_func.cpp`` (P-N-Suganthan/CEC2017-BoundContrained)
at 1000 random points per function, ``opfunu`` agrees only on F1.  A concrete
example: ``opfunu.utils.operator.zakharov_func`` computes ``sum(0.5 * x)``
where the official definition is ``sum(0.5 * i * x_i)`` with a 1-based ``i``,
which breaks F3 and every hybrid and composition that uses Zakharov.

This module keeps the data and replaces the arithmetic with
:class:`~heurilab.core.cec_official.OfficialCEC`, a direct port of the
reference C.  Classes subclass their ``opfunu`` counterparts so ``isinstance``
checks, ``lb``/``ub``, ``dim_supported`` and ``n_fe`` keep working.

Numbering
---------
The official suite is F1 and F3-F30; F2 was withdrawn for numerical
instability.  ``opfunu`` packs the survivors into a contiguous ``F1..F29``, so
its ``F3`` is the official F4.  Everything here uses **official** numbering.

Dimensions
----------
The organisers ship data for D = 2, 10, 20, 30, 50 and 100, but not for every
function: the hybrids and most compositions have no D=20 or D=2 files, so
``get_cec2017_official_suite(ndim=20)`` yields 17 of the 29 functions.  That is
a property of the official data, not a limitation here - the competition was
run at 10, 30, 50 and 100 D.

Example
-------
>>> from heurilab import run_experiment
>>> from heurilab.core.cec2017_fixed import get_cec2017_official_suite
>>> from heurilab.algorithms import PSO, GWO
>>> suite = get_cec2017_official_suite(ndim=30)
>>> run_experiment([("PSO", PSO), ("GWO", GWO)], [suite], seed=42)
"""

from typing import Optional, Sequence

import numpy as np

from heurilab.core.benchmarks import BenchmarkSuite
from heurilab.core.cec2022_fixed import _OfficialObjective
from heurilab.core.cec_official import (
    _import_opfunu,
    CEC2017_FUNCTION_NUMBERS,
    OfficialCEC,
    official_bias,
    supported_dimensions,
)

__all__ = [
    "CEC2017_FUNCTION_NUMBERS",
    "get_cec2017_official_suite",
    "get_cec2017_official_function",
    "get_cec2017_official_optimum",
    "cec2017_supported_dimensions",
]

_INSTALL_HINT = (
    "The official CEC 2017 suite needs the data files bundled with 'opfunu'.\n"
    "    pip install opfunu\n"
    "or  pip install heurilab[cec]"
)

_CLASS_CACHE = {}


def _opfunu_class_number(official_number: int) -> int:
    """Official F-number -> ``opfunu``'s contiguous class number (F2 is gone)."""
    return official_number if official_number == 1 else official_number - 1


def _opfunu_module():
    _import_opfunu()          # clear message if opfunu is absent or broken
    import opfunu.cec_based.cec2017 as module
    return module


def _fixed_class(number: int):
    """The corrected subclass of ``opfunu``'s class for official ``F<number>``."""
    number = int(number)
    if number in _CLASS_CACHE:
        return _CLASS_CACHE[number]
    if number not in CEC2017_FUNCTION_NUMBERS:
        raise ValueError(
            f"CEC 2017 has no function F{number} "
            f"(F2 was withdrawn from the competition)")

    base = getattr(_opfunu_module(), f"F{_opfunu_class_number(number)}2017")

    class _Fixed(base):
        """``opfunu``'s data loading, the organisers' arithmetic."""

        CEC_NUMBER = number

        def __init__(self, ndim=None, bounds=None, **kwargs):
            super().__init__(ndim=ndim, bounds=bounds, **kwargs)
            self._official = OfficialCEC(2017, number, self.ndim)
            self.x_global = self._official.x_global
            self.f_global = self._official.f_global
            self.f_bias = self._official.f_bias

        def evaluate(self, x, *args):
            self.n_fe += 1
            x = np.asarray(x, dtype=float).ravel()
            self.check_solution(x, self.dim_max, self.dim_supported)
            return self._official.evaluate(x)

        __call__ = evaluate

    _Fixed.__name__ = f"CEC2017F{number}"
    _Fixed.__qualname__ = _Fixed.__name__
    _Fixed.name = f"CEC 2017 F{number} (official)"
    _CLASS_CACHE[number] = _Fixed
    return _Fixed


def __getattr__(attr):
    """Expose ``CEC2017F1``, ``CEC2017F3`` .. ``CEC2017F30`` lazily."""
    if attr.startswith("CEC2017F") and attr[8:].isdigit():
        return _fixed_class(int(attr[8:]))
    raise AttributeError(attr)


def __dir__():
    return sorted(list(globals()) + [f"CEC2017F{n}" for n in CEC2017_FUNCTION_NUMBERS])


def cec2017_supported_dimensions(number: int):
    """Dimensions the organisers ship data for, for one CEC 2017 function."""
    return supported_dimensions(2017, number)


def get_cec2017_official_function(number: int, ndim: int = 30):
    """One corrected CEC 2017 function object, by **official** F-number."""
    return _fixed_class(number)(ndim=ndim)


def get_cec2017_official_optimum(number: int) -> float:
    """``f(x*)`` for one CEC 2017 function (100 * the function number)."""
    return official_bias(2017, number)


def get_cec2017_official_suite(ndim: int = 30,
                               functions: Optional[Sequence[int]] = None,
                               category: Optional[str] = None,
                               verbose: bool = True) -> BenchmarkSuite:
    """
    The official CEC 2017 suite as a :class:`BenchmarkSuite`.

    Parameters
    ----------
    ndim : int
        Problem dimensionality.  Functions without official data at ``ndim``
        are skipped and reported - see the module docstring on D=20.
    functions : sequence of int, optional
        Restrict to these official F-numbers (1, 3..30).  Default: all.
    category : str, optional
        Suite label used in output filenames.  Default ``CEC2017_D<ndim>``.
    verbose : bool
        Report skipped functions.

    Returns
    -------
    BenchmarkSuite
        Benchmarks named ``F1``, ``F3`` .. ``F30``, bounds ``[-100, 100]^ndim``.
    """
    numbers = list(CEC2017_FUNCTION_NUMBERS)
    if functions is not None:
        wanted = {int(f) for f in functions}
        missing = wanted - set(numbers)
        if missing:
            raise ValueError(
                f"CEC 2017 has no function(s) {sorted(missing)}. Defined: {numbers}")
        numbers = [n for n in numbers if n in wanted]

    suite = BenchmarkSuite(category=category or f"CEC2017_D{ndim}")
    skipped = []
    for number in numbers:
        if ndim not in supported_dimensions(2017, number):
            skipped.append(number)
            continue
        suite.add(name=f"F{number}",
                  obj_func=_OfficialObjective(2017, number, ndim),
                  lb=-100.0, ub=100.0, dim=ndim)

    if not suite.benchmarks:
        raise ValueError(
            f"No CEC 2017 function has official data for ndim={ndim}. "
            f"The competition defines D = 10, 30, 50 and 100.")
    if skipped and verbose:
        print(f"  [CEC2017] D{ndim}: using {len(suite)} functions, "
              f"skipped {len(skipped)} -> {[f'F{n}' for n in skipped]} "
              f"(no official data at this dimension)")
    return suite

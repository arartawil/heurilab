"""
Official CEC 2022 suite (F1-F12), with ``opfunu``'s defects corrected.

``opfunu`` is the only Python package that ships the official CEC 2022 data
files, but its implementations of the functions disagree with the organisers'
reference C code (``cec22_test_func.cpp``, P-N-Suganthan/2022-SO-BO).  The
composition functions are the worst of it:

* **Shift indexing.**  F9-F12 z-transform *every* sub-function with
  ``f_shift[0]`` while weighting sub-function *i* with ``f_shift[i]``.  The
  reference uses ``&Os[i*nx]`` for both, so every sub-landscape is centred on
  its own optimum.  With ``f_shift[0]`` everywhere the sub-landscapes collapse
  onto one point and the composition stops being a composition.
* **Rotation-matrix slice.**  The per-sub-function slice
  ``f_matrix[i*ndim:(i+1)*ndim]`` is correct and matches ``&Mr[i*nx*nx]``; it
  is the *rotation flags* that are wrong.  The reference leaves the fifth term
  of F9 and the first term of F10 unrotated (``r_flag = 0``); ``opfunu``
  rotates both.
* **Sub-function scaling.**  F9's bent-cigar term should be scaled by
  ``10000/1e30``, not ``1e-6``; F11's five lambdas are in the wrong order
  entirely.

F1-F8 are wrong too, for unrelated reasons (for instance
``opfunu.utils.operator.zakharov_func`` computes ``sum(0.5*x)`` where the
official definition is ``sum(0.5*i*x_i)``), so this module replaces the whole
suite rather than just the four composition functions.

Every class here subclasses its ``opfunu`` counterpart - so ``isinstance``
checks, ``lb``/``ub``, ``dim_supported`` and ``n_fe`` keep working - and
overrides ``evaluate()`` with :class:`~heurilab.core.cec_official.OfficialCEC`,
a direct port of the reference C.

Example
-------
>>> from heurilab import run_experiment
>>> from heurilab.core.cec2022_fixed import get_cec2022_suite
>>> from heurilab.algorithms import PSO, GWO
>>> suite = get_cec2022_suite(ndim=20)
>>> run_experiment([("PSO", PSO), ("GWO", GWO)], [suite], seed=42)
"""

from typing import Optional, Sequence

import numpy as np

from heurilab.core.benchmarks import BenchmarkSuite
from heurilab.core.cec_official import (
    _import_opfunu,
    CEC2022_FUNCTION_NUMBERS,
    OfficialCEC,
    official_bias,
    supported_dimensions,
)

__all__ = [
    "CEC2022_FUNCTION_NUMBERS",
    "get_cec2022_suite",
    "get_cec2022_function",
    "get_cec2022_optimum",
    "cec2022_supported_dimensions",
]

_INSTALL_HINT = (
    "The official CEC 2022 suite needs the data files bundled with 'opfunu'.\n"
    "    pip install opfunu\n"
    "or  pip install heurilab[cec]"
)

_CLASS_CACHE = {}


def _opfunu_module():
    _import_opfunu()          # clear message if opfunu is absent or broken
    import opfunu.cec_based.cec2022 as module
    return module


def _fixed_class(number: int):
    """The corrected subclass of ``opfunu``'s ``F<number>2022``."""
    number = int(number)
    if number in _CLASS_CACHE:
        return _CLASS_CACHE[number]
    if number not in CEC2022_FUNCTION_NUMBERS:
        raise ValueError(f"CEC 2022 has no function F{number}")

    base = getattr(_opfunu_module(), f"F{number}2022")

    class _Fixed(base):
        """``opfunu``'s data loading, the organisers' arithmetic."""

        CEC_NUMBER = number

        def __init__(self, ndim=None, bounds=None, **kwargs):
            super().__init__(ndim=ndim, bounds=bounds, **kwargs)
            self._official = OfficialCEC(2022, number, self.ndim)
            # opfunu points x_global at the raw shift vector; for the
            # compositions that is only the first sub-function's centre, and
            # the corrected object knows where f actually attains its bias.
            self.x_global = self._official.x_global
            self.f_global = self._official.f_global
            self.f_bias = self._official.f_bias

        def evaluate(self, x, *args):
            self.n_fe += 1
            x = np.asarray(x, dtype=float).ravel()
            self.check_solution(x, self.dim_max, self.dim_supported)
            return self._official.evaluate(x)

        __call__ = evaluate

    _Fixed.__name__ = f"CEC2022F{number}"
    _Fixed.__qualname__ = _Fixed.__name__
    _Fixed.name = f"CEC 2022 F{number} (official)"
    _CLASS_CACHE[number] = _Fixed
    return _Fixed


def __getattr__(attr):
    """Expose ``CEC2022F1`` .. ``CEC2022F12`` without importing opfunu eagerly."""
    if attr.startswith("CEC2022F") and attr[8:].isdigit():
        return _fixed_class(int(attr[8:]))
    raise AttributeError(attr)


def __dir__():
    return sorted(list(globals()) + [f"CEC2022F{n}" for n in CEC2022_FUNCTION_NUMBERS])


def cec2022_supported_dimensions(number: int):
    """Dimensions the organisers ship data for, for one CEC 2022 function."""
    return supported_dimensions(2022, number)


def get_cec2022_function(number: int, ndim: int = 10):
    """One corrected CEC 2022 function object, by **official** F-number."""
    return _fixed_class(number)(ndim=ndim)


def get_cec2022_optimum(number: int) -> float:
    """``f(x*)`` for one CEC 2022 function (300, 400, 600, ... 2700)."""
    return official_bias(2022, number)


class _OfficialObjective:
    """Picklable ``f(x) -> float`` adapter, so parallel runs can ship it.

    Holds the plain :class:`OfficialCEC` rather than the ``opfunu`` subclass:
    the subclass carries loaded ``opfunu`` state that is pointless to pickle,
    and ``n_fe`` counting across processes would be meaningless anyway.
    """

    __slots__ = ("year", "number", "ndim", "name", "_fn")

    def __init__(self, year, number, ndim):
        self.year, self.number, self.ndim = year, number, ndim
        self.name = f"CEC{year}-F{number}-D{ndim}"
        self._fn = OfficialCEC(year, number, ndim)

    def __call__(self, x):
        return float(self._fn.evaluate(x))

    def __repr__(self):
        return f"<official {self.name}>"

    def __getstate__(self):
        return (self.year, self.number, self.ndim)

    def __setstate__(self, state):
        self.__init__(*state)


def get_cec2022_suite(ndim: int = 10,
                      functions: Optional[Sequence[int]] = None,
                      category: Optional[str] = None,
                      verbose: bool = True) -> BenchmarkSuite:
    """
    The official CEC 2022 suite as a :class:`BenchmarkSuite`.

    Parameters
    ----------
    ndim : int
        Problem dimensionality.  CEC 2022 is defined for 2, 10 and 20; F6-F8
        (the hybrids) are not defined for 2.  Functions without data for
        ``ndim`` are skipped and reported.
    functions : sequence of int, optional
        Restrict to these official F-numbers (1..12).  Default: all.
    category : str, optional
        Suite label used in output filenames.  Default ``CEC2022_D<ndim>``.
    verbose : bool
        Report skipped functions.

    Returns
    -------
    BenchmarkSuite
        Benchmarks named ``F1`` .. ``F12``, bounds ``[-100, 100]^ndim``.
    """
    numbers = list(CEC2022_FUNCTION_NUMBERS)
    if functions is not None:
        wanted = {int(f) for f in functions}
        missing = wanted - set(numbers)
        if missing:
            raise ValueError(
                f"CEC 2022 has no function(s) {sorted(missing)}. Defined: {numbers}")
        numbers = [n for n in numbers if n in wanted]

    suite = BenchmarkSuite(category=category or f"CEC2022_D{ndim}")
    skipped = []
    for number in numbers:
        if ndim not in supported_dimensions(2022, number):
            skipped.append(number)
            continue
        suite.add(name=f"F{number}",
                  obj_func=_OfficialObjective(2022, number, ndim),
                  lb=-100.0, ub=100.0, dim=ndim)

    if not suite.benchmarks:
        raise ValueError(
            f"No CEC 2022 function has official data for ndim={ndim}. "
            f"The competition defines D = 2, 10 and 20.")
    if skipped and verbose:
        print(f"  [CEC2022] D{ndim}: using {len(suite)} functions, "
              f"skipped {len(skipped)} -> {[f'F{n}' for n in skipped]} "
              f"(no official data at this dimension)")
    return suite

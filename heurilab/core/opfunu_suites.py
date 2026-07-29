"""
CEC benchmark suites backed by ``opfunu``.

HeuriLab ships native implementations of CEC 2017 and CEC 2020
(:mod:`heurilab.core.cec2017`, :mod:`heurilab.core.cec2020`). This module adds
the full CEC catalogue — 2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020,
2021 and 2022 — by wrapping `opfunu <https://github.com/thieu1995/opfunu>`_,
which carries the official shift vectors and rotation matrices.

``opfunu`` is an optional dependency::

    pip install heurilab[cec]      # or: pip install opfunu

Example
-------
>>> from heurilab import run_experiment, get_opfunu_suite
>>> from heurilab.algorithms import PSO, GWO
>>> suite = get_opfunu_suite("2022", ndim=20)
>>> run_experiment([("PSO", PSO), ("GWO", GWO)], [suite], seed=42)

Notes
-----
Each CEC function supports only certain dimensionalities (CEC 2022 is defined
for 2, 10 and 20; CEC 2017 for 2, 10, 20, 30, 50 and 100). Functions that do
not support the requested ``ndim`` are skipped, and the skipped list is
reported rather than silently dropped.
"""

from typing import List, Optional, Sequence

from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite

#: CEC editions available through opfunu.
OPFUNU_YEARS = ("2005", "2008", "2010", "2013", "2014",
                "2015", "2017", "2019", "2020", "2021", "2022")

_INSTALL_HINT = (
    "The opfunu-backed CEC suites require the optional 'opfunu' package.\n"
    "    pip install opfunu\n"
    "or  pip install heurilab[cec]\n"
    "HeuriLab's built-in get_cec2017_suite() / get_cec2020_suite() work without it."
)


def _require_opfunu():
    try:
        import opfunu  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without opfunu
        raise ImportError(_INSTALL_HINT) from exc
    return opfunu


class _OpfunuObjective:
    """Picklable adapter turning an opfunu function object into ``f(x) -> float``.

    A plain ``instance.evaluate`` bound method would also pickle, but wrapping
    it keeps the return type pinned to ``float`` (opfunu returns numpy scalars)
    and gives worker processes in a parallel run a clean, self-describing
    callable.
    """

    __slots__ = ("_fn", "name")

    def __init__(self, fn, name: str):
        self._fn = fn
        self.name = name

    def __call__(self, x):
        return float(self._fn.evaluate(x))

    def __repr__(self):
        return f"<opfunu {self.name}>"


def _family_module(year: str):
    year = str(year)
    if year not in OPFUNU_YEARS:
        raise ValueError(
            f"CEC {year} is not available through opfunu. "
            f"Available editions: {', '.join(OPFUNU_YEARS)}"
        )
    _require_opfunu()
    import importlib
    return importlib.import_module(f"opfunu.cec_based.cec{year}")


#: Editions where opfunu's class numbering differs from the official CEC numbering.
#: CEC 2017 officially defines F1 and F3-F30 (F2 was withdrawn for instability),
#: but opfunu packs them into F1..F29. Left unmapped, opfunu's "F3" is Rosenbrock
#: while the official F3 is Zakharov - a silent mislabelling in any results table.
#: We relabel to the official numbering so suites line up with HeuriLab's native
#: CEC 2017 suite and with the published literature.
_OFFICIAL_NUMBERING = {
    "2017": lambda k: k if k == 1 else k + 1,
}


def official_number(year: str, opfunu_number: int) -> int:
    """Map an opfunu class number to the official CEC function number."""
    mapper = _OFFICIAL_NUMBERING.get(str(year))
    return mapper(opfunu_number) if mapper else opfunu_number


def _family_classes(year: str):
    """Return [(official_number, class), ...] for a CEC edition, in numeric order."""
    module = _family_module(year)
    suffix = str(year)
    out = []
    for attr in dir(module):
        if not attr.startswith("F") or not attr.endswith(suffix):
            continue
        number = attr[1:-len(suffix)]
        if not number.isdigit():
            continue
        out.append((official_number(year, int(number)), getattr(module, attr)))
    return sorted(out, key=lambda pair: pair[0])


def list_opfunu_functions(year: str = "2017") -> List[int]:
    """Function numbers defined for a CEC edition (CEC 2017 skips F2, for example)."""
    return [n for n, _ in _family_classes(year)]


def get_opfunu_suite(year: str = "2017",
                     ndim: int = 30,
                     functions: Optional[Sequence[int]] = None,
                     category: Optional[str] = None,
                     verbose: bool = True) -> BenchmarkSuite:
    """
    Build a :class:`BenchmarkSuite` from an opfunu CEC edition.

    Parameters
    ----------
    year : str
        CEC edition: one of :data:`OPFUNU_YEARS`.
    ndim : int
        Problem dimensionality. Functions that do not support it are skipped.
    functions : sequence of int, optional
        Restrict to these *official* CEC function numbers (e.g. ``[1, 3, 4]``).
        Default: all.
    category : str, optional
        Suite label used in output filenames. Defaults to ``CEC<year>_D<ndim>``.
    verbose : bool
        Report skipped functions. Set ``False`` to silence.

    Returns
    -------
    BenchmarkSuite

    Raises
    ------
    ImportError
        If ``opfunu`` is not installed.
    ValueError
        If the edition is unknown, or no function supports ``ndim``.
    """
    pairs = _family_classes(year)
    if functions is not None:
        wanted = set(int(f) for f in functions)
        missing = wanted - {n for n, _ in pairs}
        if missing:
            raise ValueError(
                f"CEC {year} has no function(s) {sorted(missing)}. "
                f"Defined: {[n for n, _ in pairs]}"
            )
        pairs = [(n, c) for n, c in pairs if n in wanted]

    suite = BenchmarkSuite(category=category or f"CEC{year}_D{ndim}")
    skipped = []

    for number, cls in pairs:
        try:
            fn = cls(ndim=ndim)
        except Exception as exc:                      # unsupported ndim, missing data file
            skipped.append((number, type(exc).__name__))
            continue

        supported = getattr(fn, "dim_supported", None)
        if supported and ndim not in supported:
            skipped.append((number, f"ndim {ndim} not in {supported}"))
            continue

        lb, ub = fn.lb, fn.ub
        lo, hi = float(min(lb)), float(max(ub))
        # CEC bounds are uniform across dimensions; fall back to the arrays if not.
        if not (all(float(v) == lo for v in lb) and all(float(v) == hi for v in ub)):
            lo, hi = lb, ub

        suite.add(
            name=f"F{number}",
            obj_func=_OpfunuObjective(fn, f"CEC{year}-F{number}-D{ndim}"),
            lb=lo, ub=hi, dim=int(fn.ndim),
        )

    if not suite.benchmarks:
        raise ValueError(
            f"No CEC {year} function supports ndim={ndim}. "
            f"Try one of the dimensions listed by each function's dim_supported."
        )
    if skipped and verbose:
        print(f"  [opfunu] CEC{year} D{ndim}: using {len(suite)} functions, "
              f"skipped {len(skipped)} -> {[f'F{n}' for n, _ in skipped]}")
    return suite


def get_opfunu_optimum(year: str = "2017", number: int = 1, ndim: int = 30) -> float:
    """
    Known global optimum value ``f(x*)`` for one CEC function.

    Useful for reporting error-to-optimum (``f(x) - f(x*)``) rather than raw
    fitness, which is the convention in the CEC competition reports.
    """
    for n, cls in _family_classes(year):
        if n == number:
            return float(cls(ndim=ndim).f_global)
    raise ValueError(f"CEC {year} has no function F{number}")


# ── Convenience wrappers for the editions used most often ────────────────

def get_cec2014_opfunu_suite(ndim: int = 30, **kw) -> BenchmarkSuite:
    """CEC 2014 (30 functions). Supported dimensions: 10, 20, 30, 50, 100."""
    return get_opfunu_suite("2014", ndim, **kw)


def get_cec2017_opfunu_suite(ndim: int = 30, **kw) -> BenchmarkSuite:
    """CEC 2017 (29 functions: F1, F3-F30). Supported dimensions: 2, 10, 20, 30, 50, 100."""
    return get_opfunu_suite("2017", ndim, **kw)


def get_cec2020_opfunu_suite(ndim: int = 10, **kw) -> BenchmarkSuite:
    """CEC 2020 (10 functions). Supported dimensions: 2, 5, 10, 15, 20, 30, 50, 100."""
    return get_opfunu_suite("2020", ndim, **kw)


def get_cec2021_opfunu_suite(ndim: int = 10, **kw) -> BenchmarkSuite:
    """CEC 2021 (10 functions, basic/bias/shift/rot variants)."""
    return get_opfunu_suite("2021", ndim, **kw)


def get_cec2022_opfunu_suite(ndim: int = 10, **kw) -> BenchmarkSuite:
    """CEC 2022 (12 functions). Supported dimensions: 2, 10, 20."""
    return get_opfunu_suite("2022", ndim, **kw)

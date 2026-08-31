"""
Faithful NumPy port of the official CEC 2017 / CEC 2022 reference C code.

Why this module exists
----------------------
HeuriLab used to reach the CEC suites two ways, and both were wrong:

* :mod:`heurilab.core.cec2017` generates its own shift vectors at runtime and
  applies no rotation matrices at all - a CEC-*inspired* suite, not CEC 2017.
* ``opfunu`` ships the **official** shift/rotation/shuffle data files, but its
  Python implementations of the functions themselves disagree with the
  organisers' reference code on almost every function.  Two representative
  examples, both verified against the C source:

  - ``opfunu.utils.operator.zakharov_func`` computes ``sum(0.5 * x)`` where the
    official definition is ``sum(0.5 * i * x_i)`` (i 1-based).  That breaks
    CEC 2022 F1 and CEC 2017 F3.
  - The CEC 2022 composition functions F9-F12 z-transform *every* sub-function
    with ``f_shift[0]`` while weighting sub-function *i* with ``f_shift[i]``.
    The official code uses ``&Os[i*nx]`` for both.  Several of the sub-function
    lambdas and rotation flags are wrong there too.

This module re-implements the suites directly from the organisers' sources:

* CEC 2017 - ``cec17_test_func.cpp`` from P-N-Suganthan/CEC2017-BoundContrained
* CEC 2022 - ``cec22_test_func.cpp`` from P-N-Suganthan/2022-SO-BO

Only the *data* (``shift_data_*.txt``, ``M_*_D*.txt``, ``shuffle_data_*.txt``)
is taken from ``opfunu``; those files are byte-identical to the organisers'
``input_data`` directories.

Fidelity notes
--------------
The reference C code carries a few quirks that change the returned value.  They
are reproduced here deliberately, because the competition results everyone
publishes against were produced by that code:

* ``schaffer_F7_func`` reads the *unrotated* buffer ``y`` inside its loop
  instead of the rotated ``z`` it just computed, so the rotation matrix has no
  effect on CEC 2017 F6 / CEC 2022 F3.  Inside a hybrid function it reads
  ``y[0:n]`` - the head of the permuted vector - not its own sub-block.
* ``step_rastrigin_func`` rounds the stale contents of ``y`` before calling
  ``sr_func``, which immediately overwrites ``y``; the rounding is therefore
  dead code and the function equals plain Rastrigin.
* The composition weight for a point sitting exactly on a sub-optimum is the
  *finite* sentinel ``1e99``, not a true infinity.

Everything here is verified point-for-point against the compiled reference C by
``tests/test_cec_validity.py``.
"""

import math
import os

import numpy as np

__all__ = [
    "CEC2017_FUNCTION_NUMBERS",
    "CEC2022_FUNCTION_NUMBERS",
    "CEC2017_DIMENSIONS",
    "CEC2022_DIMENSIONS",
    "OfficialCEC",
    "official_bias",
    "supported_dimensions",
]

_PI = 3.1415926535897932384626433832795029
_E = 2.7182818284590452353602874713526625
_INF = 1.0e99

#: Official CEC 2017 numbering.  F2 was withdrawn from the competition for
#: being numerically unstable in high dimensions, so it is not in the suite.
CEC2017_FUNCTION_NUMBERS = (1,) + tuple(range(3, 31))

#: Official CEC 2022 numbering - a contiguous F1..F12.
CEC2022_FUNCTION_NUMBERS = tuple(range(1, 13))

#: Dimensions the organisers ship data for.  Not every function supports every
#: entry; use :func:`supported_dimensions` for a per-function answer.
CEC2017_DIMENSIONS = (2, 10, 20, 30, 50, 100)
CEC2022_DIMENSIONS = (2, 10, 20)

_BIAS_2017 = {n: 100.0 * n for n in range(1, 31)}
_BIAS_2022 = {1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1800.0,
              7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0,
              12: 2700.0}


def official_bias(year, func_num):
    """``f(x*)`` for one official CEC function."""
    table = _BIAS_2017 if int(year) == 2017 else _BIAS_2022
    try:
        return table[int(func_num)]
    except KeyError:
        raise ValueError(f"CEC {year} has no function F{func_num}")


# ======================================================================
#  Official data files (shipped by opfunu, identical to the organisers')
# ======================================================================

_DATA_HINT = (
    "The official CEC suites need the data files bundled with 'opfunu'.\n"
    "    pip install opfunu\n"
    "or  pip install heurilab[cec]"
)


def _import_opfunu():
    """Import ``opfunu``, distinguishing "absent" from "present but broken".

    opfunu 1.0.x does ``import pkg_resources`` without declaring setuptools;
    Python >= 3.12 venvs do not preinstall it; and setuptools >= 81 removed
    pkg_resources outright.  So opfunu can install cleanly and still fail to
    import.  Reporting that as "opfunu is not installed" sends people down the
    wrong path, which is why the two cases are told apart here.
    """
    try:
        import opfunu
    except ImportError as exc:  # pragma: no cover - environment-dependent
        missing = getattr(exc, "name", "") or ""
        if missing == "opfunu" or missing.startswith("opfunu."):
            raise ImportError(_DATA_HINT) from exc
        raise ImportError(
            f"opfunu is installed but could not be imported: {exc}.\n"
            f"opfunu 1.0.x needs pkg_resources, which only setuptools<81 still "
            f"provides:\n"
            f"    pip install 'setuptools<81'\n"
            f"or  pip install heurilab[cec]   (which pins it for you)") from exc
    return opfunu


def _data_dir(year):
    opfunu = _import_opfunu()
    path = os.path.join(os.path.dirname(opfunu.__file__),
                        "cec_based", f"data_{int(year)}")
    if not os.path.isdir(path):  # pragma: no cover
        raise ImportError(f"opfunu is installed but {path} is missing.\n{_DATA_HINT}")
    return path


def _read(year, filename):
    path = os.path.join(_data_dir(year), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.genfromtxt(path, dtype=float)


def _is_composition(year, func_num):
    return func_num >= 21 if int(year) == 2017 else func_num >= 9


def _shuffle_file(year, func_num, ndim):
    if int(year) == 2017:
        if 11 <= func_num <= 20 or func_num in (29, 30):
            return f"shuffle_data_{func_num}_D{ndim}.txt"
    else:
        if 6 <= func_num <= 8:
            return f"shuffle_data_{func_num}_D{ndim}.txt"
    return None


_DATA_CACHE = {}


def _load_data(year, func_num, ndim):
    """Shift / rotation / shuffle data for one function, mirroring the C loader.

    Returns ``(shift, matrix, shuffle)`` where ``shift`` is ``(ndim,)`` for
    plain and hybrid functions and ``(cf_num, ndim)`` for compositions,
    ``matrix`` is ``(ndim, ndim)`` or ``(cf_num, ndim, ndim)`` to match, and
    ``shuffle`` is a 0-based permutation - ``(ndim,)``, or ``(3, ndim)`` for
    CEC 2017 F29/F30 - or ``None``.
    """
    year, func_num, ndim = int(year), int(func_num), int(ndim)
    key = (year, func_num, ndim)
    cached = _DATA_CACHE.get(key)
    if cached is not None:
        return cached

    matrix_raw = np.atleast_2d(_read(year, f"M_{func_num}_D{ndim}.txt"))
    if matrix_raw.shape[1] < ndim:
        raise ValueError(f"M_{func_num}_D{ndim}.txt has too few columns")
    shift_raw = _read(year, f"shift_data_{func_num}.txt")

    if _is_composition(year, func_num):
        cf_num = _COMPOSITION_SIZE[(year, func_num)]
        shift_raw = np.atleast_2d(shift_raw)
        if shift_raw.shape[0] < cf_num or shift_raw.shape[1] < ndim:
            raise ValueError(f"shift_data_{func_num}.txt is too small for D={ndim}")
        shift = np.array(shift_raw[:cf_num, :ndim], dtype=float)
        if matrix_raw.shape[0] < cf_num * ndim:
            raise ValueError(f"M_{func_num}_D{ndim}.txt is too small for D={ndim}")
        matrix = np.array(
            matrix_raw[:cf_num * ndim, :ndim].reshape(cf_num, ndim, ndim),
            dtype=float)
    else:
        flat = np.ravel(shift_raw)
        if flat.size < ndim:
            raise ValueError(f"shift_data_{func_num}.txt is too small for D={ndim}")
        shift = np.array(flat[:ndim], dtype=float)
        if matrix_raw.shape[0] < ndim:
            raise ValueError(f"M_{func_num}_D{ndim}.txt is too small for D={ndim}")
        matrix = np.array(matrix_raw[:ndim, :ndim], dtype=float)

    shuffle = None
    name = _shuffle_file(year, func_num, ndim)
    if name is not None:
        raw = np.ravel(_read(year, name)).astype(int) - 1   # the C uses S[i]-1
        if year == 2017 and func_num in (29, 30):
            if raw.size < 3 * ndim:
                raise ValueError(f"{name} is too small for D={ndim}")
            shuffle = raw[:10 * ndim].reshape(10, ndim)[:3]
        else:
            if raw.size < ndim:
                raise ValueError(f"{name} is too small for D={ndim}")
            shuffle = raw[:ndim]

    _DATA_CACHE[key] = (shift, matrix, shuffle)
    return _DATA_CACHE[key]


def supported_dimensions(year, func_num):
    """Dimensions for which the organisers ship data for one function."""
    year = int(year)
    candidates = CEC2017_DIMENSIONS if year == 2017 else CEC2022_DIMENSIONS
    out = []
    for ndim in candidates:
        try:
            _load_data(year, func_num, ndim)
        except (FileNotFoundError, ValueError):
            continue
        out.append(ndim)
    return tuple(out)


# ======================================================================
#  The evaluation engine - a line-by-line port of the reference C
# ======================================================================

class _Engine:
    """Mirrors the reference C, including its two global scratch buffers.

    ``y`` and ``z`` are ``double *`` globals in the C sources and several
    functions read one after another has written it.  Keeping them as explicit
    buffers is what makes the quirky cases (``schaffer_F7`` inside a hybrid,
    ``bi_rastrigin`` with ``s_flag=0``) come out comparable.
    """

    __slots__ = ("nx", "y", "z")

    def __init__(self, nx):
        self.nx = nx
        self.y = np.zeros(nx, dtype=float)
        self.z = np.zeros(nx, dtype=float)

    # -- shift / rotate ------------------------------------------------
    def sr(self, x, n, os_, mr, sh_rate, s_flag, r_flag):
        """``sr_func``: returns the view ``z[:n]`` it just filled."""
        y, z = self.y, self.z
        if s_flag == 1:
            if r_flag == 1:
                y[:n] = (x[:n] - os_[:n]) * sh_rate
                z[:n] = mr[:n, :n].dot(y[:n])
            else:
                z[:n] = (x[:n] - os_[:n]) * sh_rate
        else:
            if r_flag == 1:
                y[:n] = x[:n] * sh_rate
                z[:n] = mr[:n, :n].dot(y[:n])
            else:
                z[:n] = x[:n] * sh_rate
        return z[:n]

    # -- basic functions -----------------------------------------------
    def sphere(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        return float(np.sum(z * z))

    def ellips(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        if n == 1:
            return float(z[0] * z[0])
        i = np.arange(n, dtype=float)
        return float(np.sum(10.0 ** (6.0 * i / (n - 1)) * z * z))

    def bent_cigar(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        return float(z[0] * z[0] + 1.0e6 * np.sum(z[1:] * z[1:]))

    def discus(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        return float(1.0e6 * z[0] * z[0] + np.sum(z[1:] * z[1:]))

    def zakharov(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        sum1 = float(np.sum(z * z))
        sum2 = float(np.sum(0.5 * np.arange(1, n + 1, dtype=float) * z))
        return sum1 + sum2 ** 2 + sum2 ** 4

    def rosenbrock(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 2.048 / 100.0, s, r)
        z += 1.0                        # the C shifts the optimum to the origin
        t1 = z[:-1] * z[:-1] - z[1:]
        t2 = z[:-1] - 1.0
        return float(np.sum(100.0 * t1 * t1 + t2 * t2))

    def ackley(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        sum1 = -0.2 * math.sqrt(float(np.sum(z * z)) / n)
        sum2 = float(np.sum(np.cos(2.0 * _PI * z))) / n
        return _E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0

    def weierstrass(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 0.5 / 100.0, s, r)
        j = np.arange(0, 21, dtype=float)
        a_j = 0.5 ** j
        b_j = 3.0 ** j
        total = float(np.sum(a_j[None, :] *
                             np.cos(2.0 * _PI * b_j[None, :] * (z[:, None] + 0.5))))
        sum2 = float(np.sum(a_j * np.cos(2.0 * _PI * b_j * 0.5)))
        return total - n * sum2

    def griewank(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 600.0 / 100.0, s, r)
        s_ = float(np.sum(z * z))
        p = 1.0
        for i in range(n):                       # the C multiplies sequentially
            p *= math.cos(z[i] / math.sqrt(1.0 + i))
        return 1.0 + s_ / 4000.0 - p

    def rastrigin(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 5.12 / 100.0, s, r)
        return float(np.sum(z * z - 10.0 * np.cos(2.0 * _PI * z) + 10.0))

    def step_rastrigin(self, x, n, os_, mr, s, r):
        # The reference rounds `y` *before* sr_func refills it, so the rounding
        # never reaches the result.  Reproduced by simply not doing it.
        return self.rastrigin(x, n, os_, mr, s, r)

    def schwefel(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1000.0 / 100.0, s, r)
        z += 4.209687462275036e+002
        total = 0.0
        for i in range(n):
            zi = float(z[i])
            if zi > 500:
                m = math.fmod(zi, 500)
                total -= (500.0 - m) * math.sin(math.pow(500.0 - m, 0.5))
                tmp = (zi - 500.0) / 100
                total += tmp * tmp / n
            elif zi < -500:
                m = math.fmod(abs(zi), 500)
                total -= (-500.0 + m) * math.sin(math.pow(500.0 - m, 0.5))
                tmp = (zi + 500.0) / 100
                total += tmp * tmp / n
            else:
                total -= zi * math.sin(math.pow(abs(zi), 0.5))
        return total + 4.189828872724338e+002 * n

    def katsuura(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 5.0 / 100.0, s, r)
        tmp3 = math.pow(1.0 * n, 1.2)
        pow2 = 2.0 ** np.arange(1, 33, dtype=float)
        t = pow2[None, :] * z[:, None]
        temp = np.sum(np.abs(t - np.floor(t + 0.5)) / pow2[None, :], axis=1)
        out = 1.0
        for i in range(n):                       # the C multiplies sequentially
            out *= math.pow(1.0 + (i + 1) * float(temp[i]), 10.0 / tmp3)
        tmp1 = 10.0 / n / n
        return out * tmp1 - tmp1

    def grie_rosen(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 5.0 / 100.0, s, r)
        z += 1.0
        t1 = z[:-1] * z[:-1] - z[1:]
        t2 = z[:-1] - 1.0
        temp = 100.0 * t1 * t1 + t2 * t2
        total = float(np.sum(temp * temp / 4000.0 - np.cos(temp) + 1.0))
        w1 = float(z[n - 1]) * float(z[n - 1]) - float(z[0])
        w2 = float(z[n - 1]) - 1.0
        last = 100.0 * w1 * w1 + w2 * w2
        return total + last * last / 4000.0 - math.cos(last) + 1.0

    def escaffer6(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        a = np.empty(n, dtype=float)
        b = np.empty(n, dtype=float)
        a[:n - 1] = z[:n - 1]
        b[:n - 1] = z[1:n]
        a[n - 1] = z[n - 1]
        b[n - 1] = z[0]
        q = a * a + b * b
        t1 = np.sin(np.sqrt(q)) ** 2
        t2 = 1.0 + 0.001 * q
        return float(np.sum(0.5 + (t1 - 0.5) / (t2 * t2)))

    def happycat(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 5.0 / 100.0, s, r)
        z -= 1.0
        r2 = float(np.sum(z * z))
        sum_z = float(np.sum(z))
        return math.pow(abs(r2 - n), 2 * (1.0 / 8.0)) + (0.5 * r2 + sum_z) / n + 0.5

    def hgbat(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 5.0 / 100.0, s, r)
        z -= 1.0
        r2 = float(np.sum(z * z))
        sum_z = float(np.sum(z))
        return (math.pow(abs(math.pow(r2, 2.0) - math.pow(sum_z, 2.0)),
                         2 * (1.0 / 4.0))
                + (0.5 * r2 + sum_z) / n + 0.5)

    def schaffer_f7(self, x, n, os_, mr, s, r):
        # Deliberate: the reference fills z but then reads the *unrotated* y.
        self.sr(x, n, os_, mr, 1.0, s, r)
        y = self.y
        t = np.sqrt(y[:n - 1] * y[:n - 1] + y[1:n] * y[1:n])
        tmp = np.sin(50.0 * t ** 0.2)
        total = float(np.sum(t ** 0.5 + t ** 0.5 * tmp * tmp))
        return total * total / (n - 1) / (n - 1)

    def levy_2017(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        return self._levy(z, n, 1.0)

    def levy_2022(self, x, n, os_, mr, s, r):
        z = self.sr(x, n, os_, mr, 1.0, s, r)
        return self._levy(z, n, 0.0)

    @staticmethod
    def _levy(z, n, offset):
        # CEC 2017 uses w = 1 + (z - 1)/4; CEC 2022 uses w = 1 + (z - 0)/4.
        w = 1.0 + (z - offset) / 4.0
        term1 = math.sin(_PI * float(w[0])) ** 2
        term3 = ((float(w[n - 1]) - 1) ** 2 *
                 (1 + math.sin(2 * _PI * float(w[n - 1])) ** 2))
        wi = w[:n - 1]
        total = float(np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(_PI * wi + 1) ** 2)))
        return term1 + total + term3

    def bi_rastrigin(self, x, n, os_, mr, s, r):
        mu0, d = 2.5, 1.0
        s_ = 1.0 - 1.0 / (2.0 * math.pow(n + 20.0, 0.5) - 8.2)
        mu1 = -math.pow((mu0 * mu0 - d) / s_, 0.5)

        y, z = self.y, self.z
        head = np.array(x[:n], dtype=float, copy=True)     # x may alias y
        y[:n] = (head - os_[:n]) if s == 1 else head
        y[:n] *= 10.0 / 100.0

        tmpx = 2.0 * y[:n]
        tmpx = np.where(os_[:n] < 0.0, -tmpx, tmpx)
        z[:n] = tmpx
        tmpx = tmpx + mu0

        tmp1 = float(np.sum((tmpx - mu0) ** 2))
        tmp2 = float(np.sum((tmpx - mu1) ** 2)) * s_ + d * n

        if r == 1:
            y[:n] = mr[:n, :n].dot(z[:n])
            tmp = float(np.sum(np.cos(2.0 * _PI * y[:n])))
        else:
            tmp = float(np.sum(np.cos(2.0 * _PI * z[:n])))
        return min(tmp1, tmp2) + 10.0 * (n - tmp)

    # -- hybrid machinery ----------------------------------------------
    def hybrid(self, x, n, os_, mr, shuffle, s_flag, r_flag, gp, parts):
        sizes = [int(math.ceil(p * n)) for p in gp[:-1]]
        sizes.append(n - sum(sizes))
        starts = [0]
        for k in range(1, len(sizes)):
            starts.append(starts[-1] + sizes[k - 1])

        self.sr(x, n, os_, mr, 1.0, s_flag, r_flag)
        self.y[:n] = self.z[:n][shuffle[:n]]      # y[i] = z[S[i]-1]

        total = 0.0
        for part, start, size in zip(parts, starts, sizes):
            total += part(self.y[start:], size, os_, mr, 0, 0)
        return total

    # -- composition machinery -----------------------------------------
    def cf_cal(self, x, n, shift, delta, bias, fit):
        cf_num = len(fit)
        w = [0.0] * cf_num
        w_max = 0.0
        for i in range(cf_num):
            fit[i] += bias[i]
            d2 = float(np.sum((x[:n] - shift[i][:n]) ** 2))
            if d2 != 0:
                w[i] = math.pow(1.0 / d2, 0.5) * math.exp(
                    -d2 / 2.0 / n / math.pow(delta[i], 2.0))
            else:
                w[i] = _INF          # the C sentinel is finite, not math.inf
            if w[i] > w_max:
                w_max = w[i]
        w_sum = 0.0
        for i in range(cf_num):
            w_sum += w[i]
        if w_max == 0:
            w = [1.0] * cf_num
            w_sum = float(cf_num)
        out = 0.0
        for i in range(cf_num):
            out += w[i] / w_sum * fit[i]
        return out

    def composition(self, x, n, shift, matrix, delta, bias, terms, r_flag):
        fit = []
        for i, (part, num, den, sub_rotates) in enumerate(terms):
            v = part(x, n, shift[i], matrix[i], 1, r_flag if sub_rotates else 0)
            if num is not None:
                v = num * v / den                # exactly as the C writes it
            fit.append(v)
        return self.cf_cal(x, n, shift, delta, bias, fit)

    def composition_of_hybrids(self, x, n, shift, matrix, shuffle,
                               delta, bias, parts, r_flag):
        fit = [part(x, n, shift[i], matrix[i], shuffle[i], 1, r_flag)
               for i, part in enumerate(parts)]
        return self.cf_cal(x, n, shift, delta, bias, fit)


# ======================================================================
#  Suite definitions, transcribed from the reference C
# ======================================================================

# Hybrid layouts: (proportions, sub-function names)
_HYBRIDS_2017 = {
    1:  ((0.2, 0.4, 0.4), ("zakharov", "rosenbrock", "rastrigin")),
    2:  ((0.3, 0.3, 0.4), ("ellips", "schwefel", "bent_cigar")),
    3:  ((0.3, 0.3, 0.4), ("bent_cigar", "rosenbrock", "bi_rastrigin")),
    4:  ((0.2, 0.2, 0.2, 0.4), ("ellips", "ackley", "schaffer_f7", "rastrigin")),
    5:  ((0.2, 0.2, 0.3, 0.3), ("bent_cigar", "hgbat", "rastrigin", "rosenbrock")),
    6:  ((0.2, 0.2, 0.3, 0.3), ("escaffer6", "hgbat", "rosenbrock", "schwefel")),
    7:  ((0.1, 0.2, 0.2, 0.2, 0.3),
         ("katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin")),
    8:  ((0.2, 0.2, 0.2, 0.2, 0.2),
         ("ellips", "ackley", "rastrigin", "hgbat", "discus")),
    9:  ((0.2, 0.2, 0.2, 0.2, 0.2),
         ("bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "escaffer6")),
    10: ((0.1, 0.1, 0.2, 0.2, 0.2, 0.2),
         ("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7")),
}

_HYBRIDS_2022 = {
    2:  ((0.4, 0.4, 0.2), ("bent_cigar", "hgbat", "rastrigin")),
    6:  ((0.3, 0.2, 0.2, 0.1, 0.2),
         ("katsuura", "happycat", "grie_rosen", "schwefel", "ackley")),
    10: ((0.1, 0.2, 0.2, 0.2, 0.1, 0.2),
         ("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7")),
}

# Composition layouts: (delta, bias, [(sub-function, num, den, rotate?), ...]).
# `num`/`den` reproduce the C's `fit[i] = num * fit[i] / den` verbatim.
_COMPOSITIONS_2017 = {
    1: ((10, 20, 30), (0, 100, 200), [
        ("rosenbrock", None, None, True),
        ("ellips", 10000, 1e+10, True),
        ("rastrigin", None, None, True)]),
    2: ((10, 20, 30), (0, 100, 200), [
        ("rastrigin", None, None, True),
        ("griewank", 1000, 100, True),
        ("schwefel", None, None, True)]),
    3: ((10, 20, 30, 40), (0, 100, 200, 300), [
        ("rosenbrock", None, None, True),
        ("ackley", 1000, 100, True),
        ("schwefel", None, None, True),
        ("rastrigin", None, None, True)]),
    4: ((10, 20, 30, 40), (0, 100, 200, 300), [
        ("ackley", 1000, 100, True),
        ("ellips", 10000, 1e+10, True),
        ("griewank", 1000, 100, True),
        ("rastrigin", None, None, True)]),
    5: ((10, 20, 30, 40, 50), (0, 100, 200, 300, 400), [
        ("rastrigin", 10000, 1e+3, True),
        ("happycat", 1000, 1e+3, True),
        ("ackley", 1000, 100, True),
        ("discus", 10000, 1e+10, True),
        ("rosenbrock", None, None, True)]),
    6: ((10, 20, 20, 30, 40), (0, 100, 200, 300, 400), [
        ("escaffer6", 10000, 2e+7, True),
        ("schwefel", None, None, True),
        ("griewank", 1000, 100, True),
        ("rosenbrock", None, None, True),
        ("rastrigin", 10000, 1e+3, True)]),
    7: ((10, 20, 30, 40, 50, 60), (0, 100, 200, 300, 400, 500), [
        ("hgbat", 10000, 1000, True),
        ("rastrigin", 10000, 1e+3, True),
        ("schwefel", 10000, 4e+3, True),
        ("bent_cigar", 10000, 1e+30, True),
        ("ellips", 10000, 1e+10, True),
        ("escaffer6", 10000, 2e+7, True)]),
    8: ((10, 20, 30, 40, 50, 60), (0, 100, 200, 300, 400, 500), [
        ("ackley", 1000, 100, True),
        ("griewank", 1000, 100, True),
        ("discus", 10000, 1e+10, True),
        ("rosenbrock", None, None, True),
        ("happycat", 1000, 1e+3, True),
        ("escaffer6", 10000, 2e+7, True)]),
}

#: CEC 2017 F29/F30 compose *hybrid* functions rather than basic ones.
_COMPOSITIONS_OF_HYBRIDS_2017 = {
    9:  ((10, 30, 50), (0, 100, 200), (5, 6, 7)),
    10: ((10, 30, 50), (0, 100, 200), (5, 8, 9)),
}

_COMPOSITIONS_2022 = {
    1: ((10, 20, 30, 40, 50), (0, 200, 300, 100, 400), [
        ("rosenbrock", 10000, 1e+4, True),
        ("ellips", 10000, 1e+10, True),
        ("bent_cigar", 10000, 1e+30, True),
        ("discus", 10000, 1e+10, True),
        ("ellips", 10000, 1e+10, False)]),      # the 5th term is NOT rotated
    2: ((20, 10, 10), (0, 200, 100), [
        ("schwefel", None, None, False),        # nor is this one
        ("rastrigin", None, None, True),
        ("hgbat", None, None, True)]),
    6: ((20, 20, 30, 30, 20), (0, 200, 300, 400, 200), [
        ("escaffer6", 10000, 2e+7, True),
        ("schwefel", None, None, True),
        ("griewank", 1000, 100, True),
        ("rosenbrock", None, None, True),
        ("rastrigin", 10000, 1e+3, True)]),
    7: ((10, 20, 30, 40, 50, 60), (0, 300, 500, 100, 400, 200), [
        ("hgbat", 10000, 1000, True),
        ("rastrigin", 10000, 1e+3, True),
        ("schwefel", 10000, 4e+3, True),
        ("bent_cigar", 10000, 1e+30, True),
        ("ellips", 10000, 1e+10, True),
        ("escaffer6", 10000, 2e+7, True)]),
}

#: How many sub-functions each composition mixes - the C's ``cf_num``, which
#: also decides how many shift rows and rotation blocks to read.
_COMPOSITION_SIZE = {}
for _cf, _spec in _COMPOSITIONS_2017.items():
    _COMPOSITION_SIZE[(2017, 20 + _cf)] = len(_spec[2])
for _cf, _spec in _COMPOSITIONS_OF_HYBRIDS_2017.items():
    _COMPOSITION_SIZE[(2017, 20 + _cf)] = len(_spec[2])
for _cf, _spec in _COMPOSITIONS_2022.items():
    _COMPOSITION_SIZE[(2022, {1: 9, 2: 10, 6: 11, 7: 12}[_cf])] = len(_spec[2])
del _cf, _spec

#: Plain (non-hybrid, non-composition) functions, by official number.
_BASIC_2017 = {
    1: "bent_cigar", 3: "zakharov", 4: "rosenbrock", 5: "rastrigin",
    6: "schaffer_f7", 7: "bi_rastrigin", 8: "step_rastrigin",
    9: "levy_2017", 10: "schwefel",
}
_BASIC_2022 = {
    1: "zakharov", 2: "rosenbrock", 3: "schaffer_f7",
    4: "step_rastrigin", 5: "levy_2022",
}

#: CEC 2022 reuses three of the CEC 2017 hybrid layouts, under new numbers.
_CEC2022_HYBRID_IDS = {6: 2, 7: 10, 8: 6}
#: CEC 2022 composition numbers -> the C's cfNN.
_CEC2022_COMPOSITION_IDS = {9: 1, 10: 2, 11: 6, 12: 7}


class OfficialCEC:
    """One official CEC function: ``OfficialCEC(2022, 9, ndim=10)``.

    ``evaluate(x)`` returns the same value as the organisers' C code for the
    same point, to within floating-point summation order.
    """

    __slots__ = ("year", "func_num", "ndim", "f_bias", "f_global", "x_global",
                 "lb", "ub", "_shift", "_matrix", "_shuffle", "_engine")

    def __init__(self, year, func_num, ndim):
        year, func_num, ndim = int(year), int(func_num), int(ndim)
        if year not in (2017, 2022):
            raise ValueError("year must be 2017 or 2022")
        numbers = (CEC2017_FUNCTION_NUMBERS if year == 2017
                   else CEC2022_FUNCTION_NUMBERS)
        if func_num not in numbers:
            raise ValueError(
                f"CEC {year} has no function F{func_num}. Defined: {list(numbers)}")

        self.year = year
        self.func_num = func_num
        self.ndim = ndim
        self._shift, self._matrix, self._shuffle = _load_data(year, func_num, ndim)
        self._engine = _Engine(ndim)
        self.f_bias = official_bias(year, func_num)
        self.f_global = self.f_bias
        self.x_global = self._locate_optimum()
        self.lb = np.full(ndim, -100.0)
        self.ub = np.full(ndim, 100.0)

    def _locate_optimum(self):
        """Where ``f`` actually attains ``f_bias``.

        For all but one function that is the shift vector (the first row of it,
        for a composition).  CEC 2017 F9 is the exception: the reference Levy
        uses ``w = 1 + (z - 1)/4``, so it bottoms out at ``z = 1``, not
        ``z = 0``, and the optimum sits at ``o + M^-1 * 1`` instead of ``o``.
        Evaluating it at the raw shift vector returns roughly
        ``900 + 0.09 * ndim``, not 900 - an easy way to convince yourself a
        correct implementation is broken.
        """
        if _is_composition(self.year, self.func_num):
            return self._shift[0].copy()
        if self.year == 2017 and self.func_num == 9:
            return self._shift + np.linalg.solve(self._matrix, np.ones(self.ndim))
        return self._shift.copy()

    def __repr__(self):
        return f"OfficialCEC(CEC{self.year} F{self.func_num} D{self.ndim})"

    def _part(self, name):
        return getattr(self._engine, name)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.ndim:
            raise ValueError(
                f"CEC{self.year} F{self.func_num} expects {self.ndim} variables, "
                f"got {x.size}")
        return self._raw(x) + self.f_bias

    __call__ = evaluate

    def _raw(self, x):
        year, fn, n = self.year, self.func_num, self.ndim
        eng = self._engine

        if year == 2017:
            if fn in _BASIC_2017:
                return self._part(_BASIC_2017[fn])(
                    x, n, self._shift, self._matrix, 1, 1)
            if 11 <= fn <= 20:
                gp, parts = _HYBRIDS_2017[fn - 10]
                return eng.hybrid(x, n, self._shift, self._matrix, self._shuffle,
                                  1, 1, gp, [self._part(p) for p in parts])
            cf = fn - 20
            if cf in _COMPOSITIONS_2017:
                delta, bias, terms = _COMPOSITIONS_2017[cf]
                return eng.composition(
                    x, n, self._shift, self._matrix, delta, list(bias),
                    [(self._part(p), num, den, rot) for p, num, den, rot in terms],
                    1)
            delta, bias, hybrid_ids = _COMPOSITIONS_OF_HYBRIDS_2017[cf]
            parts = [self._hybrid_closure(h) for h in hybrid_ids]
            return eng.composition_of_hybrids(
                x, n, self._shift, self._matrix, self._shuffle,
                delta, list(bias), parts, 1)

        if fn in _BASIC_2022:
            return self._part(_BASIC_2022[fn])(x, n, self._shift, self._matrix, 1, 1)
        if fn in _CEC2022_HYBRID_IDS:
            gp, parts = _HYBRIDS_2022[_CEC2022_HYBRID_IDS[fn]]
            return eng.hybrid(x, n, self._shift, self._matrix, self._shuffle,
                              1, 1, gp, [self._part(p) for p in parts])
        delta, bias, terms = _COMPOSITIONS_2022[_CEC2022_COMPOSITION_IDS[fn]]
        return eng.composition(
            x, n, self._shift, self._matrix, delta, list(bias),
            [(self._part(p), num, den, rot) for p, num, den, rot in terms], 1)

    def _hybrid_closure(self, hybrid_id):
        """A CEC 2017 hybrid callable usable as a composition sub-function."""
        gp, parts = _HYBRIDS_2017[hybrid_id]
        engine = self._engine
        subs = [self._part(p) for p in parts]

        def run(x, n, os_, mr, shuffle, s_flag, r_flag):
            return engine.hybrid(x, n, os_, mr, shuffle, s_flag, r_flag, gp, subs)

        return run

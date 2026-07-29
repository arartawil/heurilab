"""
CEC 2020 Benchmark Functions (10 functions: F1–F10)
====================================================
Based on the CEC 2020 competition on single objective bound constrained
numerical optimization (Yue, Price, Suganthan, et al.).

Categories:
  Unimodal (1):           CEC20_F1
  Multimodal (3):         CEC20_F2, CEC20_F3, CEC20_F4
  Hybrid (3):             CEC20_F5, CEC20_F6, CEC20_F7
  Composition (3):        CEC20_F8, CEC20_F9, CEC20_F10

All functions:
  - Search range: [-100, 100]^D
  - Default dimension: D = 10 (also tested at D = 5, 15, 20)
  - Shifted and rotated versions
  - Global optimum at f* values listed below

Author: Arar Al Tawil
"""

import numpy as np
from heurilab.core.benchmarks import BenchmarkSuite


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _shift(x, seed):
    """Shift x by a reproducible random vector in [-80, 80]."""
    rng = np.random.RandomState(seed)
    o = rng.uniform(-80, 80, len(x))
    return x - o


def _rotation_matrix(dim, seed):
    """Generate a reproducible random orthogonal rotation matrix."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _shift_rotate(x, shift_seed, rot_seed):
    """Apply shift then rotation."""
    z = _shift(x, shift_seed)
    M = _rotation_matrix(len(z), rot_seed)
    return M @ z


def _lambda_transform(D, alpha=10):
    """Lambda diagonal matrix for conditioning."""
    diag = np.array([alpha ** (0.5 * i / (D - 1)) for i in range(D)])
    return np.diag(diag)


# ═══════════════════════════════════════════════════════════════════════
#  Base Functions
# ═══════════════════════════════════════════════════════════════════════

def _bent_cigar(z):
    return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)


def _schwefel(z):
    D = len(z)
    z = z + 4.209687462275036e+002
    result = 0.0
    for i in range(D):
        zi = z[i]
        if abs(zi) <= 500:
            result += zi * np.sin(np.sqrt(abs(zi)))
        elif zi > 500:
            result += (500 - zi % 500) * np.sin(
                np.sqrt(abs(500 - zi % 500))) - (zi - 500) ** 2 / (10000 * D)
        else:
            result += (abs(zi) % 500 - 500) * np.sin(
                np.sqrt(abs(abs(zi) % 500 - 500))) - (zi + 500) ** 2 / (10000 * D)
    return 418.9828872724339 * D - result


def _lunacek_rastrigin(z):
    D = len(z)
    mu0 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
    mu1 = -np.sqrt((mu0 ** 2 - d) / s)
    x_hat = 2 * np.sign(z[0]) * z
    sum1 = np.sum((x_hat - mu0) ** 2)
    sum2 = D * d + s * np.sum((x_hat - mu1) ** 2)
    sum3 = 10 * (D - np.sum(np.cos(2 * np.pi * (x_hat - mu0))))
    return min(sum1, sum2) + sum3


def _rosenbrock(z):
    z = z + 1
    return np.sum(100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1) ** 2)


def _schaffer_f7(z):
    D = len(z)
    result = 0.0
    for i in range(D - 1):
        si = np.sqrt(z[i] ** 2 + z[i + 1] ** 2)
        result += np.sqrt(si) * (np.sin(50 * si ** 0.2) + 1)
    return (result / (D - 1)) ** 2


def _rastrigin(z):
    return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)


def _levy(z):
    w = 1 + (z - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3


def _high_conditioned_elliptic(z):
    D = len(z)
    idx = np.arange(D)
    return np.sum(1e6 ** (idx / max(D - 1, 1)) * z ** 2)


def _discus(z):
    return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def _ackley(z):
    D = len(z)
    s1 = -0.2 * np.sqrt(np.sum(z ** 2) / D)
    s2 = np.sum(np.cos(2 * np.pi * z)) / D
    return -20 * np.exp(s1) - np.exp(s2) + 20 + np.e


def _griewank(z):
    D = len(z)
    return np.sum(z ** 2) / 4000 - np.prod(
        np.cos(z / np.sqrt(np.arange(1, D + 1)))) + 1


def _happycat(z):
    D = len(z)
    sum_sq = np.sum(z ** 2)
    return abs(sum_sq - D) ** 0.25 + (0.5 * sum_sq + np.sum(z)) / D + 0.5


def _hgbat(z):
    D = len(z)
    sum_sq = np.sum(z ** 2)
    sum_x = np.sum(z)
    return abs(sum_sq ** 2 - sum_x ** 2) ** 0.5 + (0.5 * sum_sq + sum_x) / D + 0.5


def _expanded_schaffer_f6(z):
    D = len(z)
    result = 0.0
    for i in range(D - 1):
        t = z[i] ** 2 + z[i + 1] ** 2
        result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
    t = z[-1] ** 2 + z[0] ** 2
    result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
    return result


def _modified_schwefel(z):
    return _schwefel(z)


# ═══════════════════════════════════════════════════════════════════════
#  Composition Helper
# ═══════════════════════════════════════════════════════════════════════

def _composition(x, shift_seeds, rot_seeds, base_funcs, sigmas, lambdas, biases):
    K = len(base_funcs)
    D = len(x)

    # Compute weights
    w = np.zeros(K)
    for i in range(K):
        rng = np.random.RandomState(shift_seeds[i])
        oi = rng.uniform(-80, 80, D)
        diff = x - oi
        w[i] = np.exp(-np.sum(diff ** 2) / (2 * D * sigmas[i] ** 2))

    w_sum = np.sum(w)
    if w_sum == 0:
        w = np.ones(K) / K
    else:
        w = w / w_sum

    result = 0.0
    for i in range(K):
        z = _shift_rotate(x, shift_seeds[i], rot_seeds[i])
        fi = lambdas[i] * base_funcs[i](z) + biases[i]
        result += w[i] * fi

    return result


# ═══════════════════════════════════════════════════════════════════════
#  CEC2020 10 Functions   (seed base = 2020)
# ═══════════════════════════════════════════════════════════════════════

_SB = 2020   # seed base

# Optimal values
_FSTAR = [100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500]


# ---- F1: Shifted and Rotated Bent Cigar (Unimodal) ----
def CEC20_F1(x):
    x = np.asarray(x, dtype=float)
    z = _shift_rotate(x, _SB + 0, _SB + 100)
    return _bent_cigar(z) + _FSTAR[0]


# ---- F2: Shifted and Rotated Schwefel (Multimodal) ----
def CEC20_F2(x):
    x = np.asarray(x, dtype=float)
    z = _shift(x, _SB + 1)
    L = _lambda_transform(len(z), alpha=10)
    z = L @ z
    M = _rotation_matrix(len(z), _SB + 101)
    z = M @ z
    return _schwefel(z) + _FSTAR[1]


# ---- F3: Shifted and Rotated Lunacek Bi-Rastrigin (Multimodal) ----
def CEC20_F3(x):
    x = np.asarray(x, dtype=float)
    z = _shift_rotate(x, _SB + 2, _SB + 102)
    return _lunacek_rastrigin(z) + _FSTAR[2]


# ---- F4: Expanded Rosenbrock plus Griewank (Multimodal) ----
def CEC20_F4(x):
    x = np.asarray(x, dtype=float)
    z = _shift_rotate(x, _SB + 3, _SB + 103)
    D = len(z)
    result = 0.0
    for i in range(D - 1):
        ri = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
        result += ri ** 2 / 4000 - np.cos(ri) + 1
    ri = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
    result += ri ** 2 / 4000 - np.cos(ri) + 1
    return result + _FSTAR[3]


# ---- F5: Hybrid 1 (Schwefel + Rastrigin + Elliptic) ----
def CEC20_F5(x):
    x = np.asarray(x, dtype=float)
    z = _shift_rotate(x, _SB + 4, _SB + 104)
    D = len(z)
    rng = np.random.RandomState(_SB + 205)
    perm = rng.permutation(D)
    z = z[perm]

    n1 = int(np.ceil(0.3 * D))
    n2 = int(np.ceil(0.3 * D))
    g1, g2, g3 = z[:n1], z[n1:n1 + n2], z[n1 + n2:]

    return _modified_schwefel(g1) + _rastrigin(g2) + _high_conditioned_elliptic(g3) + _FSTAR[4]


# ---- F6: Hybrid 2 (Schaffer F7 + HGBat + Rosenbrock + Schwefel) ----
def CEC20_F6(x):
    x = np.asarray(x, dtype=float)
    z = _shift_rotate(x, _SB + 5, _SB + 105)
    D = len(z)
    rng = np.random.RandomState(_SB + 206)
    perm = rng.permutation(D)
    z = z[perm]

    n1 = max(2, int(np.ceil(0.2 * D)))
    n2 = max(2, int(np.ceil(0.2 * D)))
    n3 = max(2, int(np.ceil(0.3 * D)))
    g1, g2, g3, g4 = z[:n1], z[n1:n1 + n2], z[n1 + n2:n1 + n2 + n3], z[n1 + n2 + n3:]

    result = 0.0
    if len(g1) >= 2:
        result += _schaffer_f7(g1)
    result += _hgbat(g2) + _rosenbrock(g3)
    if len(g4) > 0:
        result += _modified_schwefel(g4)
    return result + _FSTAR[5]


# ---- F7: Hybrid 3 (HappyCat + Ackley + Rastrigin + Schwefel + Schaffer) ----
def CEC20_F7(x):
    x = np.asarray(x, dtype=float)
    z = _shift_rotate(x, _SB + 6, _SB + 106)
    D = len(z)
    rng = np.random.RandomState(_SB + 207)
    perm = rng.permutation(D)
    z = z[perm]

    n1 = max(1, int(np.ceil(0.1 * D)))
    n2 = max(2, int(np.ceil(0.2 * D)))
    n3 = max(2, int(np.ceil(0.2 * D)))
    n4 = max(2, int(np.ceil(0.2 * D)))
    g1 = z[:n1]
    g2 = z[n1:n1 + n2]
    g3 = z[n1 + n2:n1 + n2 + n3]
    g4 = z[n1 + n2 + n3:n1 + n2 + n3 + n4]
    g5 = z[n1 + n2 + n3 + n4:]

    result = _happycat(g1) + _ackley(g2) + _rastrigin(g3)
    if len(g4) > 0:
        result += _modified_schwefel(g4)
    if len(g5) >= 2:
        result += _schaffer_f7(g5)
    return result + _FSTAR[6]


# ---- F8: Composition 1 (Rastrigin + Griewank + Schwefel) ----
def CEC20_F8(x):
    x = np.asarray(x, dtype=float)
    return _composition(
        x,
        shift_seeds=[_SB + 7, _SB + 8, _SB + 9],
        rot_seeds=[_SB + 107, _SB + 108, _SB + 109],
        base_funcs=[_rastrigin, _griewank, _modified_schwefel],
        sigmas=[10, 20, 30],
        lambdas=[1, 10, 1],
        biases=[0, 100, 200],
    ) + _FSTAR[7]


# ---- F9: Composition 2 (Ackley + Elliptic + Griewank + Rastrigin) ----
def CEC20_F9(x):
    x = np.asarray(x, dtype=float)
    return _composition(
        x,
        shift_seeds=[_SB + 10, _SB + 11, _SB + 12, _SB + 13],
        rot_seeds=[_SB + 110, _SB + 111, _SB + 112, _SB + 113],
        base_funcs=[_ackley, _high_conditioned_elliptic, _griewank, _rastrigin],
        sigmas=[10, 20, 30, 40],
        lambdas=[10, 1e-6, 10, 1],
        biases=[0, 100, 200, 300],
    ) + _FSTAR[8]


# ---- F10: Composition 3 (Rastrigin + HappyCat + Ackley + Discus + Rosenbrock) ----
def CEC20_F10(x):
    x = np.asarray(x, dtype=float)
    return _composition(
        x,
        shift_seeds=[_SB + 14, _SB + 15, _SB + 16, _SB + 17, _SB + 18],
        rot_seeds=[_SB + 114, _SB + 115, _SB + 116, _SB + 117, _SB + 118],
        base_funcs=[_rastrigin, _happycat, _ackley, _discus, _rosenbrock],
        sigmas=[10, 20, 30, 40, 50],
        lambdas=[10, 1, 10, 1e-6, 1],
        biases=[0, 100, 200, 300, 400],
    ) + _FSTAR[9]


# ═══════════════════════════════════════════════════════════════════════
#  Function Info Table  (name, func, lb, ub, dim)
# ═══════════════════════════════════════════════════════════════════════

CEC2020_FUNCTIONS = [
    # Unimodal
    ("CEC20_F1_BentCigar",           CEC20_F1,  -100, 100, 10),
    # Multimodal
    ("CEC20_F2_Schwefel",            CEC20_F2,  -100, 100, 10),
    ("CEC20_F3_LunacekBiRastrigin",  CEC20_F3,  -100, 100, 10),
    ("CEC20_F4_RosenbrockGriewank",  CEC20_F4,  -100, 100, 10),
    # Hybrid
    ("CEC20_F5_Hybrid1",             CEC20_F5,  -100, 100, 10),
    ("CEC20_F6_Hybrid2",             CEC20_F6,  -100, 100, 10),
    ("CEC20_F7_Hybrid3",             CEC20_F7,  -100, 100, 10),
    # Composition
    ("CEC20_F8_Composition1",        CEC20_F8,  -100, 100, 10),
    ("CEC20_F9_Composition2",        CEC20_F9,  -100, 100, 10),
    ("CEC20_F10_Composition3",       CEC20_F10, -100, 100, 10),
]


# ═══════════════════════════════════════════════════════════════════════
#  Pre-built Suites
# ═══════════════════════════════════════════════════════════════════════

def get_cec2020_suite(category="CEC2020"):
    """Return a BenchmarkSuite with all 10 CEC 2020 functions."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2020_FUNCTIONS:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2020_unimodal_suite(category="CEC2020-Unimodal"):
    """Return CEC 2020 unimodal function (F1)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2020_FUNCTIONS[:1]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2020_multimodal_suite(category="CEC2020-Multimodal"):
    """Return CEC 2020 multimodal functions (F2–F4)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2020_FUNCTIONS[1:4]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2020_hybrid_suite(category="CEC2020-Hybrid"):
    """Return CEC 2020 hybrid functions (F5–F7)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2020_FUNCTIONS[4:7]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2020_composition_suite(category="CEC2020-Composition"):
    """Return CEC 2020 composition functions (F8–F10)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2020_FUNCTIONS[7:]:
        suite.add(name, func, lb, ub, dim)
    return suite

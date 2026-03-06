"""
CEC 2017 Benchmark Functions (29 functions: F1, F3–F30)
========================================================
Based on the CEC 2017 competition on real-parameter single-objective
optimization (Awad et al., 2016). F2 is excluded per the official spec.

Categories:
  Unimodal (2):           CEC17_F1, CEC17_F3
  Simple Multimodal (7):  CEC17_F4 – CEC17_F10
  Hybrid (10):            CEC17_F11 – CEC17_F20
  Composition (10):       CEC17_F21 – CEC17_F30

All functions use seeded shift vectors for reproducibility,
so the global optimum is NOT at the origin.
Search range: [-100, 100]^D for all functions.
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


# ── Base functions used in hybrids and compositions ──────────────────

def _bent_cigar(z):
    return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)


def _zakharov(z):
    n = len(z)
    s1 = np.sum(z ** 2)
    s2 = np.sum(0.5 * np.arange(1, n + 1) * z)
    return s1 + s2 ** 2 + s2 ** 4


def _rosenbrock(z):
    z = z + 1  # shift optimum to [1,1,...,1]
    return np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)


def _rastrigin(z):
    return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)


def _schaffer_f6(x1, x2):
    t = x1 ** 2 + x2 ** 2
    return 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2


def _expanded_scaffer(z):
    n = len(z)
    s = 0.0
    for i in range(n - 1):
        s += _schaffer_f6(z[i], z[i + 1])
    s += _schaffer_f6(z[-1], z[0])
    return s


def _levy(z):
    w = 1 + (z - 1) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    t3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return t1 + t2 + t3


def _schwefel(z):
    n = len(z)
    z = z + 4.209687462275036e2
    s = 0.0
    for i in range(n):
        if abs(z[i]) <= 500:
            s += z[i] * np.sin(np.sqrt(abs(z[i])))
        elif z[i] > 500:
            s += (500 - z[i] % 500) * np.sin(np.sqrt(abs(500 - z[i] % 500))) - \
                 (z[i] - 500) ** 2 / (10000 * n)
        else:
            s += (z[i] % 500 - 500) * np.sin(np.sqrt(abs(z[i] % 500 - 500))) - \
                 (z[i] + 500) ** 2 / (10000 * n)
    return 418.9829 * n - s


def _lunacek_bi_rastrigin(z):
    n = len(z)
    mu0 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * np.sqrt(n + 20) - 8.2)
    mu1 = -np.sqrt((mu0 ** 2 - d) / s)
    s1 = np.sum((z - mu0) ** 2)
    s2 = n * d + s * np.sum((z - mu1) ** 2)
    s3 = np.sum(10 * (1 - np.cos(2 * np.pi * (z - mu0))))
    return min(s1, s2) + s3


def _hgbat(z):
    n = len(z)
    s1 = np.sum(z ** 2)
    s2 = np.sum(z)
    return np.sqrt(abs(s1 ** 2 - s2 ** 2)) + (0.5 * s1 + s2) / n + 0.5


def _katsuura(z):
    n = len(z)
    p = 1.0
    for i in range(n):
        s = 0.0
        for j in range(1, 33):
            t = 2 ** j
            s += abs(t * z[i] - round(t * z[i])) / t
        p *= (1 + (i + 1) * s) ** (10.0 / n ** 1.2)
    return (10.0 / n ** 2) * p - 10.0 / n ** 2


def _happy_cat(z):
    n = len(z)
    s1 = np.sum(z ** 2)
    s2 = np.sum(z)
    return abs(s1 - n) ** 0.25 + (0.5 * s1 + s2) / n + 0.5


def _griewank(z):
    n = len(z)
    s = np.sum(z ** 2) / 4000.0
    p = np.prod(np.cos(z / np.sqrt(np.arange(1, n + 1))))
    return s - p + 1


def _ackley(z):
    n = len(z)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(z ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * z)) / n) + 20 + np.e)


def _discus(z):
    return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def _elliptic(z):
    n = len(z)
    return np.sum(np.power(1e6, np.arange(n) / (n - 1 + 1e-16)) * z ** 2)


# ═══════════════════════════════════════════════════════════════════════
#  Unimodal Functions
# ═══════════════════════════════════════════════════════════════════════

def CEC17_F1(x):
    """Shifted Bent Cigar — F_min = 100"""
    z = _shift(x, seed=1)
    return _bent_cigar(z) + 100


def CEC17_F3(x):
    """Shifted Zakharov — F_min = 300"""
    z = _shift(x, seed=3)
    return _zakharov(z) + 300


# ═══════════════════════════════════════════════════════════════════════
#  Simple Multimodal Functions
# ═══════════════════════════════════════════════════════════════════════

def CEC17_F4(x):
    """Shifted Rosenbrock's — F_min = 400"""
    z = _shift(x, seed=4)
    return _rosenbrock(z) + 400


def CEC17_F5(x):
    """Shifted Rastrigin's — F_min = 500"""
    z = _shift(x, seed=5)
    return _rastrigin(z) + 500


def CEC17_F6(x):
    """Shifted Expanded Scaffer's F6 — F_min = 600"""
    z = _shift(x, seed=6)
    return _expanded_scaffer(z) + 600


def CEC17_F7(x):
    """Shifted Lunacek Bi-Rastrigin — F_min = 700"""
    z = _shift(x, seed=7)
    return _lunacek_bi_rastrigin(z) + 700


def CEC17_F8(x):
    """Shifted Non-Continuous Rastrigin — F_min = 800"""
    z = _shift(x, seed=8)
    y = np.where(np.abs(z) >= 0.5, np.round(2 * z) / 2, z)
    return _rastrigin(y) + 800


def CEC17_F9(x):
    """Shifted Levy — F_min = 900"""
    z = _shift(x, seed=9)
    return _levy(z) + 900


def CEC17_F10(x):
    """Shifted Schwefel — F_min = 1000"""
    z = _shift(x, seed=10)
    return _schwefel(z) + 1000


# ═══════════════════════════════════════════════════════════════════════
#  Hybrid Functions
# ═══════════════════════════════════════════════════════════════════════

def _hybrid(x, seed, funcs, fracs, bias):
    """Generic hybrid: split dims among funcs according to fracs."""
    z = _shift(x, seed)
    n = len(z)
    rng = np.random.RandomState(seed + 100)
    perm = rng.permutation(n)
    z = z[perm]

    result = 0.0
    start = 0
    for func, frac in zip(funcs, fracs):
        end = start + max(1, int(round(frac * n)))
        if end > n:
            end = n
        result += func(z[start:end])
        start = end
    return result + bias


def CEC17_F11(x):
    """Hybrid 1 — F_min = 1100"""
    return _hybrid(x, 11,
                   [_zakharov, _rosenbrock, _rastrigin],
                   [0.2, 0.4, 0.4], 1100)


def CEC17_F12(x):
    """Hybrid 2 — F_min = 1200"""
    return _hybrid(x, 12,
                   [_elliptic, _schwefel, _bent_cigar],
                   [0.3, 0.3, 0.4], 1200)


def CEC17_F13(x):
    """Hybrid 3 — F_min = 1300"""
    return _hybrid(x, 13,
                   [_bent_cigar, _rosenbrock, _lunacek_bi_rastrigin],
                   [0.3, 0.3, 0.4], 1300)


def CEC17_F14(x):
    """Hybrid 4 — F_min = 1400"""
    return _hybrid(x, 14,
                   [_elliptic, _ackley, _schwefel, _rastrigin],
                   [0.2, 0.2, 0.3, 0.3], 1400)


def CEC17_F15(x):
    """Hybrid 5 — F_min = 1500"""
    return _hybrid(x, 15,
                   [_bent_cigar, _hgbat, _rastrigin, _rosenbrock],
                   [0.2, 0.2, 0.3, 0.3], 1500)


def CEC17_F16(x):
    """Hybrid 6 — F_min = 1600"""
    return _hybrid(x, 16,
                   [_expanded_scaffer, _hgbat, _rosenbrock, _schwefel],
                   [0.2, 0.2, 0.3, 0.3], 1600)


def CEC17_F17(x):
    """Hybrid 7 — F_min = 1700"""
    return _hybrid(x, 17,
                   [_katsuura, _ackley, _griewank, _schwefel, _rastrigin],
                   [0.1, 0.2, 0.2, 0.2, 0.3], 1700)


def CEC17_F18(x):
    """Hybrid 8 — F_min = 1800"""
    return _hybrid(x, 18,
                   [_elliptic, _ackley, _schwefel, _rastrigin, _bent_cigar],
                   [0.2, 0.2, 0.2, 0.2, 0.2], 1800)


def CEC17_F19(x):
    """Hybrid 9 — F_min = 1900"""
    return _hybrid(x, 19,
                   [_happy_cat, _katsuura, _ackley, _rastrigin, _schwefel, _rosenbrock],
                   [0.15, 0.15, 0.15, 0.15, 0.2, 0.2], 1900)


def CEC17_F20(x):
    """Hybrid 10 — F_min = 2000"""
    return _hybrid(x, 20,
                   [_hgbat, _katsuura, _ackley, _rastrigin, _schwefel, _expanded_scaffer],
                   [0.1, 0.15, 0.15, 0.15, 0.2, 0.25], 2000)


# ═══════════════════════════════════════════════════════════════════════
#  Composition Functions
# ═══════════════════════════════════════════════════════════════════════

def _composition(x, seed, funcs, sigmas, lambdas, biases, Fstar):
    """Generic composition function."""
    n = len(x)
    K = len(funcs)
    rng = np.random.RandomState(seed + 200)

    # Generate shift vectors for each component
    shifts = [rng.uniform(-80, 80, n) for _ in range(K)]

    w = np.zeros(K)
    for i in range(K):
        z = x - shifts[i]
        dist = np.sum(z ** 2)
        w[i] = np.exp(-dist / (2 * n * sigmas[i] ** 2 + 1e-16))

    w_sum = np.sum(w)
    if w_sum == 0:
        w = np.ones(K) / K
    else:
        w = w / w_sum

    result = 0.0
    for i in range(K):
        z = x - shifts[i]
        fi = lambdas[i] * funcs[i](z) + biases[i]
        result += w[i] * fi

    return result + Fstar


def CEC17_F21(x):
    """Composition 1 — F_min = 2100"""
    return _composition(x, 21,
                        [_rosenbrock, _elliptic, _rastrigin],
                        [10, 20, 30],
                        [1, 1e-6, 1],
                        [0, 100, 200], 2100)


def CEC17_F22(x):
    """Composition 2 — F_min = 2200"""
    return _composition(x, 22,
                        [_rastrigin, _griewank, _schwefel],
                        [10, 20, 30],
                        [1, 10, 1],
                        [0, 100, 200], 2200)


def CEC17_F23(x):
    """Composition 3 — F_min = 2300"""
    return _composition(x, 23,
                        [_rosenbrock, _ackley, _schwefel, _rastrigin],
                        [10, 20, 30, 40],
                        [1, 10, 1, 1],
                        [0, 100, 200, 300], 2300)


def CEC17_F24(x):
    """Composition 4 — F_min = 2400"""
    return _composition(x, 24,
                        [_ackley, _elliptic, _griewank, _rastrigin],
                        [10, 20, 30, 40],
                        [10, 1e-6, 10, 1],
                        [0, 100, 200, 300], 2400)


def CEC17_F25(x):
    """Composition 5 — F_min = 2500"""
    return _composition(x, 25,
                        [_rastrigin, _happy_cat, _ackley, _discus, _rosenbrock],
                        [10, 20, 30, 40, 50],
                        [10, 1, 10, 1e-6, 1],
                        [0, 100, 200, 300, 400], 2500)


def CEC17_F26(x):
    """Composition 6 — F_min = 2600"""
    return _composition(x, 26,
                        [_expanded_scaffer, _schwefel, _griewank, _rosenbrock, _rastrigin],
                        [10, 20, 30, 40, 50],
                        [10, 10, 2.5, 1e-26, 1e-6],
                        [0, 100, 200, 300, 400], 2600)


def CEC17_F27(x):
    """Composition 7 — F_min = 2700"""
    return _composition(x, 27,
                        [_hgbat, _rastrigin, _schwefel, _bent_cigar, _elliptic, _expanded_scaffer],
                        [10, 20, 30, 40, 50, 60],
                        [10, 10, 2.5, 1e-26, 1e-6, 5e-4],
                        [0, 100, 200, 300, 400, 500], 2700)


def CEC17_F28(x):
    """Composition 8 — F_min = 2800"""
    return _composition(x, 28,
                        [_ackley, _griewank, _discus, _rosenbrock, _happy_cat, _expanded_scaffer],
                        [10, 20, 30, 40, 50, 60],
                        [10, 10, 1e-6, 1, 1, 5e-4],
                        [0, 100, 200, 300, 400, 500], 2800)


def CEC17_F29(x):
    """Composition 9 — F_min = 2900"""
    return _composition(x, 29,
                        [_rastrigin, _schwefel, _hgbat, _elliptic,
                         _expanded_scaffer, _bent_cigar, _griewank],
                        [10, 20, 30, 40, 50, 60, 70],
                        [100, 10, 2.5, 1e-26, 1e-6, 5e-4, 10],
                        [0, 100, 200, 300, 400, 500, 600], 2900)


def CEC17_F30(x):
    """Composition 10 — F_min = 3000"""
    return _composition(x, 30,
                        [_rastrigin, _happy_cat, _expanded_scaffer, _schwefel,
                         _elliptic, _bent_cigar, _griewank, _rosenbrock],
                        [10, 20, 30, 40, 50, 60, 70, 80],
                        [100, 10, 2.5, 25, 1e-6, 5e-4, 10, 1],
                        [0, 100, 200, 300, 400, 500, 600, 700], 3000)


# ═══════════════════════════════════════════════════════════════════════
#  Function Info Table
# ═══════════════════════════════════════════════════════════════════════

CEC2017_FUNCTIONS = [
    # (name, func, lb, ub, dim)
    # Unimodal
    ("CEC17_F1_BentCigar",         CEC17_F1,  -100, 100, 30),
    ("CEC17_F3_Zakharov",          CEC17_F3,  -100, 100, 30),
    # Simple Multimodal
    ("CEC17_F4_Rosenbrock",        CEC17_F4,  -100, 100, 30),
    ("CEC17_F5_Rastrigin",         CEC17_F5,  -100, 100, 30),
    ("CEC17_F6_ExpandedScaffer",   CEC17_F6,  -100, 100, 30),
    ("CEC17_F7_LunacekBiRastrigin", CEC17_F7, -100, 100, 30),
    ("CEC17_F8_NonContRastrigin",  CEC17_F8,  -100, 100, 30),
    ("CEC17_F9_Levy",              CEC17_F9,  -100, 100, 30),
    ("CEC17_F10_Schwefel",         CEC17_F10, -100, 100, 30),
    # Hybrid
    ("CEC17_F11_Hybrid1",          CEC17_F11, -100, 100, 30),
    ("CEC17_F12_Hybrid2",          CEC17_F12, -100, 100, 30),
    ("CEC17_F13_Hybrid3",          CEC17_F13, -100, 100, 30),
    ("CEC17_F14_Hybrid4",          CEC17_F14, -100, 100, 30),
    ("CEC17_F15_Hybrid5",          CEC17_F15, -100, 100, 30),
    ("CEC17_F16_Hybrid6",          CEC17_F16, -100, 100, 30),
    ("CEC17_F17_Hybrid7",          CEC17_F17, -100, 100, 30),
    ("CEC17_F18_Hybrid8",          CEC17_F18, -100, 100, 30),
    ("CEC17_F19_Hybrid9",          CEC17_F19, -100, 100, 30),
    ("CEC17_F20_Hybrid10",         CEC17_F20, -100, 100, 30),
    # Composition
    ("CEC17_F21_Composition1",     CEC17_F21, -100, 100, 30),
    ("CEC17_F22_Composition2",     CEC17_F22, -100, 100, 30),
    ("CEC17_F23_Composition3",     CEC17_F23, -100, 100, 30),
    ("CEC17_F24_Composition4",     CEC17_F24, -100, 100, 30),
    ("CEC17_F25_Composition5",     CEC17_F25, -100, 100, 30),
    ("CEC17_F26_Composition6",     CEC17_F26, -100, 100, 30),
    ("CEC17_F27_Composition7",     CEC17_F27, -100, 100, 30),
    ("CEC17_F28_Composition8",     CEC17_F28, -100, 100, 30),
    ("CEC17_F29_Composition9",     CEC17_F29, -100, 100, 30),
    ("CEC17_F30_Composition10",    CEC17_F30, -100, 100, 30),
]


# ═══════════════════════════════════════════════════════════════════════
#  Pre-built Suites
# ═══════════════════════════════════════════════════════════════════════

def get_cec2017_suite(category="CEC2017"):
    """Return a BenchmarkSuite with all 29 CEC 2017 functions."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2017_FUNCTIONS:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2017_unimodal_suite(category="CEC2017-Unimodal"):
    """Return CEC 2017 unimodal functions (F1, F3)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2017_FUNCTIONS[:2]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2017_multimodal_suite(category="CEC2017-Multimodal"):
    """Return CEC 2017 simple multimodal functions (F4–F10)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2017_FUNCTIONS[2:9]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2017_hybrid_suite(category="CEC2017-Hybrid"):
    """Return CEC 2017 hybrid functions (F11–F20)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2017_FUNCTIONS[9:19]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_cec2017_composition_suite(category="CEC2017-Composition"):
    """Return CEC 2017 composition functions (F21–F30)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CEC2017_FUNCTIONS[19:]:
        suite.add(name, func, lb, ub, dim)
    return suite

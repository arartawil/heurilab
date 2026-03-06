"""
Classic Benchmark Functions (F1–F23)
=====================================
Standard test suite from Yao et al. (1999), widely used in metaheuristic
literature (GWO, WOA, MFO, SSA, HHO, etc.).

Unimodal (F1–F7):       Sphere, Schwefel 2.22, Schwefel 1.2, Schwefel 2.21,
                         Rosenbrock, Step, Quartic (Noise)
Multimodal (F8–F13):     Schwefel 2.26, Rastrigin, Ackley, Griewank,
                         Penalized1, Penalized2
Fixed-dim (F14–F23):     Shekel Foxholes, Kowalik, Six-Hump Camel, Branin,
                         Goldstein-Price, Hartmann3, Hartmann6,
                         Shekel5, Shekel7, Shekel10
"""

import numpy as np
from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite


# ═══════════════════════════════════════════════════════════════════════
#  Unimodal Functions (F1–F7)
# ═══════════════════════════════════════════════════════════════════════

def F1(x):
    """Sphere"""
    return np.sum(x ** 2)


def F2(x):
    """Schwefel 2.22"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F3(x):
    """Schwefel 1.2"""
    n = len(x)
    s = 0.0
    for i in range(n):
        s += np.sum(x[:i + 1]) ** 2
    return s


def F4(x):
    """Schwefel 2.21"""
    return np.max(np.abs(x))


def F5(x):
    """Rosenbrock"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def F6(x):
    """Step"""
    return np.sum(np.floor(x + 0.5) ** 2)


def F7(x):
    """Quartic (with noise)"""
    n = len(x)
    return np.sum(np.arange(1, n + 1) * x ** 4) + np.random.rand()


# ═══════════════════════════════════════════════════════════════════════
#  Multimodal Functions (F8–F13)
# ═══════════════════════════════════════════════════════════════════════

def F8(x):
    """Schwefel 2.26"""
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    """Rastrigin"""
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def F10(x):
    """Ackley"""
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def F11(x):
    """Griewank"""
    n = len(x)
    s = np.sum(x ** 2) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return s - p + 1


def _u(x, a, k, m):
    """Penalty helper."""
    y = np.zeros_like(x)
    mask1 = x > a
    mask2 = x < -a
    y[mask1] = k * (x[mask1] - a) ** m
    y[mask2] = k * (-x[mask2] - a) ** m
    return y


def F12(x):
    """Penalized 1"""
    n = len(x)
    y = 1 + (x + 1) / 4.0
    s = (10 * np.sin(np.pi * y[0]) ** 2
         + np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
         + (y[-1] - 1) ** 2)
    return (np.pi / n) * s + np.sum(_u(x, 10, 100, 4))


def F13(x):
    """Penalized 2"""
    n = len(x)
    s = (0.1 * (np.sin(3 * np.pi * x[0]) ** 2
                + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
                + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)))
    return s + np.sum(_u(x, 5, 100, 4))


# ═══════════════════════════════════════════════════════════════════════
#  Fixed-Dimension Multimodal Functions (F14–F23)
# ═══════════════════════════════════════════════════════════════════════

def F14(x):
    """Shekel's Foxholes"""
    a = np.array([[-32, -16, 0, 16, 32] * 5,
                  [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5],
                 dtype=float)
    s = 0.0
    for j in range(25):
        s += 1.0 / (j + 1 + np.sum((x - a[:, j]) ** 6))
    return 1.0 / (1.0 / 500 + s)


def F15(x):
    """Kowalik"""
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                  0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    s = 0.0
    for i in range(11):
        s += (a[i] - (x[0] * (b[i] ** 2 + b[i] * x[1])) /
              (b[i] ** 2 + b[i] * x[2] + x[3])) ** 2
    return s


def F16(x):
    """Six-Hump Camel Back"""
    return (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + \
           x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2


def F17(x):
    """Branin"""
    return ((x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 +
             5 / np.pi * x[0] - 6) ** 2 +
            10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)


def F18(x):
    """Goldstein-Price"""
    p1 = (1 + (x[0] + x[1] + 1) ** 2 *
          (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] +
           6 * x[0] * x[1] + 3 * x[1] ** 2))
    p2 = (30 + (2 * x[0] - 3 * x[1]) ** 2 *
          (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] -
           36 * x[0] * x[1] + 27 * x[1] ** 2))
    return p1 * p2


def F19(x):
    """Hartmann 3-D"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3, 10, 30],
                  [0.1, 10, 35],
                  [3, 10, 30],
                  [0.1, 10, 35]], dtype=float)
    P = 1e-4 * np.array([[3689, 1170, 2673],
                          [4699, 4387, 7470],
                          [1091, 8732, 5547],
                          [381, 5743, 8828]], dtype=float)
    s = 0.0
    for i in range(4):
        s += alpha[i] * np.exp(-np.sum(A[i] * (x - P[i]) ** 2))
    return -s


def F20(x):
    """Hartmann 6-D"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                          [2329, 4135, 8307, 3736, 1004, 9991],
                          [2348, 1451, 3522, 2883, 3047, 6650],
                          [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)
    s = 0.0
    for i in range(4):
        s += alpha[i] * np.exp(-np.sum(A[i] * (x - P[i]) ** 2))
    return -s


# Shekel family
_SHEKEL_A = np.array([[4, 4, 4, 4],
                       [1, 1, 1, 1],
                       [8, 8, 8, 8],
                       [6, 6, 6, 6],
                       [3, 7, 3, 7],
                       [2, 9, 2, 9],
                       [5, 5, 3, 3],
                       [8, 1, 8, 1],
                       [6, 2, 6, 2],
                       [7, 3.6, 7, 3.6]], dtype=float)

_SHEKEL_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def _shekel(x, m):
    s = 0.0
    for i in range(m):
        s += 1.0 / (np.sum((x - _SHEKEL_A[i]) ** 2) + _SHEKEL_C[i])
    return -s


def F21(x):
    """Shekel 5"""
    return _shekel(x, 5)


def F22(x):
    """Shekel 7"""
    return _shekel(x, 7)


def F23(x):
    """Shekel 10"""
    return _shekel(x, 10)


# ═══════════════════════════════════════════════════════════════════════
#  Function Info Table
# ═══════════════════════════════════════════════════════════════════════

CLASSICAL_FUNCTIONS = [
    # (name, func, lb, ub, dim)
    # Unimodal
    ("F1_Sphere",          F1,  -100,   100,  30),
    ("F2_Schwefel2.22",    F2,   -10,    10,  30),
    ("F3_Schwefel1.2",     F3,  -100,   100,  30),
    ("F4_Schwefel2.21",    F4,  -100,   100,  30),
    ("F5_Rosenbrock",      F5,   -30,    30,  30),
    ("F6_Step",            F6,  -100,   100,  30),
    ("F7_Quartic",         F7,  -1.28, 1.28,  30),
    # Multimodal
    ("F8_Schwefel2.26",    F8,  -500,   500,  30),
    ("F9_Rastrigin",       F9, -5.12,  5.12,  30),
    ("F10_Ackley",         F10,  -32,    32,  30),
    ("F11_Griewank",       F11, -600,   600,  30),
    ("F12_Penalized1",     F12,  -50,    50,  30),
    ("F13_Penalized2",     F13,  -50,    50,  30),
    # Fixed-dimension multimodal
    ("F14_Foxholes",       F14, -65.536, 65.536, 2),
    ("F15_Kowalik",        F15,   -5,     5,     4),
    ("F16_SixHumpCamel",   F16,   -5,     5,     2),
    ("F17_Branin",         F17,   -5,    15,     2),
    ("F18_GoldsteinPrice", F18,   -2,     2,     2),
    ("F19_Hartmann3",      F19,    0,     1,     3),
    ("F20_Hartmann6",      F20,    0,     1,     6),
    ("F21_Shekel5",        F21,    0,    10,     4),
    ("F22_Shekel7",        F22,    0,    10,     4),
    ("F23_Shekel10",       F23,    0,    10,     4),
]


# ═══════════════════════════════════════════════════════════════════════
#  Pre-built Suites
# ═══════════════════════════════════════════════════════════════════════

def get_classical_suite(category="Classical"):
    """Return a BenchmarkSuite with all 23 classical functions (F1–F23)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CLASSICAL_FUNCTIONS:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_unimodal_suite(category="Unimodal"):
    """Return a BenchmarkSuite with unimodal functions (F1–F7)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CLASSICAL_FUNCTIONS[:7]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_multimodal_suite(category="Multimodal"):
    """Return a BenchmarkSuite with multimodal functions (F8–F13)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CLASSICAL_FUNCTIONS[7:13]:
        suite.add(name, func, lb, ub, dim)
    return suite


def get_fixeddim_suite(category="Fixed-Dimension"):
    """Return a BenchmarkSuite with fixed-dimension functions (F14–F23)."""
    suite = BenchmarkSuite(category)
    for name, func, lb, ub, dim in CLASSICAL_FUNCTIONS[13:]:
        suite.add(name, func, lb, ub, dim)
    return suite

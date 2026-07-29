"""
Constrained engineering design problems.

Each problem is declared as an :class:`EngineeringProblem` holding the raw
objective and the inequality constraints ``g(x) <= 0`` *separately*. That split
matters: it lets a constraint-handling strategy be chosen at run time instead of
being baked into the objective, and it lets tests check the feasibility of a
returned solution rather than only its penalised score.

For backward compatibility ``ENGINEERING_PROBLEMS`` is still exposed as a list
of ``(name, penalised_callable, dim, lb, ub)`` tuples, which is what
:func:`heurilab.engineering.runner.run_engineering_problems` consumes.

Every ``best_known`` value below is the optimum reported in the cited source and
is checked in ``tests/test_engineering.py``: an optimiser given a generous budget
must reach it, which is what catches a mistyped coefficient.
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

#: Static penalty coefficient used by :func:`penalised`.
PENALTY = 1e10


# ═════════════════════════════════════════════════════════════════════
#  Container
# ═════════════════════════════════════════════════════════════════════

@dataclass
class EngineeringProblem:
    """A constrained minimisation problem with a known published optimum."""
    name: str
    dim: int
    lb: List[float]
    ub: List[float]
    objective: Callable[[Sequence[float]], float]
    constraints: Callable[[Sequence[float]], List[float]]
    best_known: float
    reference: str
    n_constraints: int = 0
    notes: str = ""

    def __post_init__(self):
        if self.n_constraints == 0:
            lo = np.asarray(self.lb, dtype=float)
            hi = np.asarray(self.ub, dtype=float)
            self.n_constraints = len(self.constraints(lo + 0.5 * (hi - lo)))

    def violation(self, x) -> float:
        """Total constraint violation, ``sum(max(0, g_i(x)))``. Zero means feasible."""
        return float(sum(max(0.0, g) for g in self.constraints(x)))

    def is_feasible(self, x, tol: float = 1e-6) -> bool:
        return self.violation(x) <= tol

    def penalised(self, penalty: float = PENALTY) -> Callable:
        return _Penalised(self, penalty)

    def __repr__(self):
        return (f"EngineeringProblem('{self.name}', dim={self.dim}, "
                f"constraints={self.n_constraints}, best_known={self.best_known:g})")


class _Penalised:
    """Static-penalty wrapper: ``f(x) + P * sum(max(0, g_i(x))**2)``.

    A class rather than a closure so that it stays picklable, which is what
    allows engineering runs to be parallelised.
    """

    __slots__ = ("problem", "penalty")

    def __init__(self, problem: "EngineeringProblem", penalty: float = PENALTY):
        self.problem = problem
        self.penalty = penalty

    def __call__(self, x):
        value = float(self.problem.objective(x))
        pen = 0.0
        for g in self.problem.constraints(x):
            if g > 0:
                pen += self.penalty * g * g
        return value + pen

    def __repr__(self):
        return f"<penalised {self.problem.name}>"


def penalised(problem: EngineeringProblem, penalty: float = PENALTY) -> Callable:
    """Return a single-callable static-penalty form of ``problem``."""
    return _Penalised(problem, penalty)


# ═════════════════════════════════════════════════════════════════════
#  1. Pressure Vessel Design  (4 variables, 4 constraints)
# ═════════════════════════════════════════════════════════════════════

def _pressure_vessel_obj(x):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2
            + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3)


def _pressure_vessel_con(x):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return [
        -x1 + 0.0193 * x3,
        -x2 + 0.00954 * x3,
        -np.pi * x3 ** 2 * x4 - (4.0 / 3.0) * np.pi * x3 ** 3 + 1_296_000.0,
        x4 - 240.0,
    ]


# ═════════════════════════════════════════════════════════════════════
#  2. Welded Beam Design  (4 variables, 7 constraints)
# ═════════════════════════════════════════════════════════════════════

def _welded_beam_obj(x):
    h, l, t, b = x[0], x[1], x[2], x[3]
    return 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)


def _welded_beam_con(x):
    h, l, t, b = x[0], x[1], x[2], x[3]
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    tau_max, sigma_max, delta_max = 13600.0, 30000.0, 0.25

    M = P * (L + l / 2.0)
    R = np.sqrt(l ** 2 / 4.0 + ((h + t) / 2.0) ** 2)
    J = 2.0 * (np.sqrt(2.0) * h * l * (l ** 2 / 12.0 + ((h + t) / 2.0) ** 2))
    tau1 = P / (np.sqrt(2.0) * h * l)
    tau2 = M * R / J
    tau = np.sqrt(tau1 ** 2 + 2.0 * tau1 * tau2 * (l / (2.0 * R)) + tau2 ** 2)
    sigma = 6.0 * P * L / (b * t ** 2)
    delta = 4.0 * P * L ** 3 / (E * b * t ** 3)
    Pc = ((4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36.0) / L ** 2)
          * (1.0 - (t / (2.0 * L)) * np.sqrt(E / (4.0 * G))))
    return [
        tau - tau_max,
        sigma - sigma_max,
        h - b,
        0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + l) - 5.0,
        0.125 - h,
        delta - delta_max,
        P - Pc,
    ]


# ═════════════════════════════════════════════════════════════════════
#  3. Tension / Compression Spring  (3 variables, 4 constraints)
# ═════════════════════════════════════════════════════════════════════

def _spring_obj(x):
    d, D, N = x[0], x[1], x[2]
    return (N + 2.0) * D * d ** 2


def _spring_con(x):
    d, D, N = x[0], x[1], x[2]
    return [
        1.0 - (D ** 3 * N) / (71785.0 * d ** 4),
        (4.0 * D ** 2 - d * D) / (12566.0 * (D * d ** 3 - d ** 4))
        + 1.0 / (5108.0 * d ** 2) - 1.0,
        1.0 - 140.45 * d / (D ** 2 * N),
        (d + D) / 1.5 - 1.0,
    ]


# ═════════════════════════════════════════════════════════════════════
#  4. Speed Reducer / Gearbox  (7 variables, 11 constraints)
# ═════════════════════════════════════════════════════════════════════

def _speed_reducer_obj(x):
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    return (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
            + 7.4777 * (x6 ** 3 + x7 ** 3)
            + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))


def _speed_reducer_con(x):
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    return [
        27.0 / (x1 * x2 ** 2 * x3) - 1.0,
        397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1.0,
        1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1.0,
        1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1.0,
        np.sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110.0 * x6 ** 3) - 1.0,
        np.sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85.0 * x7 ** 3) - 1.0,
        x2 * x3 / 40.0 - 1.0,
        5.0 * x2 / x1 - 1.0,
        x1 / (12.0 * x2) - 1.0,
        (1.5 * x6 + 1.9) / x4 - 1.0,
        (1.1 * x7 + 1.9) / x5 - 1.0,
    ]


# ═════════════════════════════════════════════════════════════════════
#  5. Three-Bar Truss  (2 variables, 3 constraints)
# ═════════════════════════════════════════════════════════════════════

def _three_bar_obj(x):
    x1, x2 = x[0], x[1]
    return (2.0 * np.sqrt(2.0) * x1 + x2) * 100.0


def _three_bar_con(x):
    x1, x2 = x[0], x[1]
    P, sigma = 2.0, 2.0
    denom = np.sqrt(2.0) * x1 ** 2 + 2.0 * x1 * x2
    if denom <= 0:
        return [1e6, 1e6, 1e6]
    return [
        (np.sqrt(2.0) * x1 + x2) / denom * P - sigma,
        x2 / denom * P - sigma,
        1.0 / (x1 + np.sqrt(2.0) * x2) * P - sigma,
    ]


# ═════════════════════════════════════════════════════════════════════
#  6. Cantilever Beam  (5 variables, 1 constraint)
# ═════════════════════════════════════════════════════════════════════

def _cantilever_obj(x):
    return 0.0624 * (x[0] + x[1] + x[2] + x[3] + x[4])


def _cantilever_con(x):
    return [61.0 / x[0] ** 3 + 37.0 / x[1] ** 3 + 19.0 / x[2] ** 3
            + 7.0 / x[3] ** 3 + 1.0 / x[4] ** 3 - 1.0]


# ═════════════════════════════════════════════════════════════════════
#  7. Gear Train Design  (4 variables, bound-constrained only)
# ═════════════════════════════════════════════════════════════════════

def _gear_train_obj(x):
    # Teeth counts are integers; rounding keeps the objective faithful to the
    # discrete problem even when a continuous optimiser proposes fractions.
    x1, x2, x3, x4 = (round(float(v)) for v in x[:4])
    return (1.0 / 6.931 - (x3 * x2) / (x1 * x4)) ** 2


def _gear_train_con(x):
    return []


# ═════════════════════════════════════════════════════════════════════
#  8. I-Beam Vertical Deflection  (4 variables, 2 constraints)
# ═════════════════════════════════════════════════════════════════════

def _ibeam_obj(x):
    b, h, tw, tf = x[0], x[1], x[2], x[3]
    inertia = (tw * (h - 2.0 * tf) ** 3 / 12.0
               + b * tf ** 3 / 6.0
               + 2.0 * b * tf * ((h - tf) / 2.0) ** 2)
    return 5000.0 / inertia


def _ibeam_con(x):
    """Cross-sectional area <= 300 cm^2.

    This is the single-constraint variant used throughout the metaheuristic
    literature, whose optimum x* = (50, 80, 0.9, 2.3217) gives f = 0.0130741.
    The two-constraint variant additionally bounds bending stress (see
    :func:`ibeam_stress_constraint`); its optimum is a different, higher value,
    so the two must not be mixed.
    """
    b, h, tw, tf = x[0], x[1], x[2], x[3]
    return [2.0 * b * tf + tw * (h - 2.0 * tf) - 300.0]


def ibeam_stress_constraint(x):
    """Bending-stress constraint of the two-constraint I-beam variant."""
    b, h, tw, tf = x[0], x[1], x[2], x[3]
    term1 = (18.0 * h * 1e4
             / (tw * (h - 2.0 * tf) ** 3
                + 2.0 * b * tf * (4.0 * tf ** 2 + 3.0 * h * (h - 2.0 * tf))))
    term2 = 15.0 * b * 1e3 / ((h - 2.0 * tf) * tw ** 3 + 2.0 * tw * b ** 3)
    return term1 + term2 - 6.0


# ═════════════════════════════════════════════════════════════════════
#  9. Tubular Column  (2 variables, 6 constraints)
# ═════════════════════════════════════════════════════════════════════

def _tubular_obj(x):
    d, t = x[0], x[1]
    return 9.8 * d * t + 2.0 * d


def _tubular_con(x):
    d, t = x[0], x[1]
    P, sigma_y, E, L = 2500.0, 500.0, 0.85e6, 250.0
    return [
        P / (np.pi * d * t * sigma_y) - 1.0,
        8.0 * P * L ** 2 / (np.pi ** 3 * E * d * t * (d ** 2 + t ** 2)) - 1.0,
        2.0 / d - 1.0,
        d / 14.0 - 1.0,
        0.2 / t - 1.0,
        t / 8.0 - 1.0,
    ]


# ═════════════════════════════════════════════════════════════════════
#  10. Multi-Disc Clutch Brake  (5 variables, 8 constraints)
# ═════════════════════════════════════════════════════════════════════

def _clutch_vars(x):
    return (round(float(x[0])), round(float(x[1])), round(float(x[2])),
            float(x[3]), round(float(x[4])))


def _clutch_obj(x):
    ri, ro, t, F, Z = _clutch_vars(x)
    rho = 0.0000078
    return np.pi * (ro ** 2 - ri ** 2) * t * (Z + 1) * rho


def _clutch_con(x):
    ri, ro, t, F, Z = _clutch_vars(x)
    Mf, Ms, Iz, n = 3.0, 40.0, 55.0, 250.0
    Tmax, pmax, Vsrmax, delta_r = 15.0, 1.0, 10.0, 20.0
    mu, s, Lmax = 0.6, 1.5, 30.0

    denom = ro ** 2 - ri ** 2
    if denom <= 0:
        return [1e6] * 8
    Rsr = (2.0 / 3.0) * (ro ** 3 - ri ** 3) / denom          # mm
    prz = F / (np.pi * denom)                                  # N/mm^2 = MPa
    Vsr = np.pi * Rsr * n / (30.0 * 1000.0)                    # m/s
    Mh = (2.0 / 3.0) * mu * F * Z * (ro ** 3 - ri ** 3) / denom / 1000.0   # N.m
    omega = np.pi * n / 30.0                                   # rad/s
    T = Iz * omega / max(Mh + Mf, 1e-12)                        # s
    return [
        -ro + ri + delta_r,
        (Z + 1) * (t + 0.5) - Lmax,
        prz - pmax,
        prz * Vsr - pmax * Vsrmax,
        Vsr - Vsrmax,
        s * Ms - Mh,
        -T,
        T - Tmax,
    ]


# ═════════════════════════════════════════════════════════════════════
#  11. Corrugated Bulkhead  (4 variables, 6 constraints)
# ═════════════════════════════════════════════════════════════════════

def _bulkhead_obj(x):
    b, h, l, t = x[0], x[1], x[2], x[3]
    root = np.sqrt(max(l ** 2 - h ** 2, 0.0))
    denom = b + root
    if denom <= 1e-12:
        return 1e12
    return 5.885 * t * (b + l) / denom


def _bulkhead_con(x):
    b, h, l, t = x[0], x[1], x[2], x[3]
    if l < h:
        return [1e6] * 6
    root = np.sqrt(max(l ** 2 - h ** 2, 0.0))
    return [
        -b * t * h * (0.4 * b + l / 6.0) + 8.94 * (b + root),
        -t * h ** 2 * (0.2 * b + l / 12.0) + 2.2 * (8.94 * (b + root)) ** (4.0 / 3.0),
        0.0156 * b + 0.15 - t,
        0.0156 * l + 0.15 - t,
        1.05 - t,
        h - l,
    ]


# ═════════════════════════════════════════════════════════════════════
#  12. Himmelblau's Nonlinear Design  (5 variables, 6 constraints)
# ═════════════════════════════════════════════════════════════════════

def _himmelblau_obj(x):
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    return 5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141


def _himmelblau_con(x):
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    g = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
    h = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    k = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    return [-g, g - 92.0, 90.0 - h, h - 110.0, 20.0 - k, k - 25.0]


# ═════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════

PROBLEMS: List[EngineeringProblem] = [
    EngineeringProblem(
        "Pressure Vessel Design", 4,
        [0.0625, 0.0625, 10.0, 10.0], [6.1875, 6.1875, 200.0, 200.0],
        _pressure_vessel_obj, _pressure_vessel_con,
        best_known=5885.3327736,
        reference="Kannan & Kramer (1994) - continuous-thickness variant",
        notes="With thickness restricted to multiples of 0.0625 the best known is 6059.714.",
    ),
    EngineeringProblem(
        "Welded Beam Design", 4,
        [0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0],
        _welded_beam_obj, _welded_beam_con,
        best_known=1.724852,
        reference="Rao (1996); Coello (2000)",
    ),
    EngineeringProblem(
        "Tension Compression Spring", 3,
        [0.05, 0.25, 2.0], [2.0, 1.3, 15.0],
        _spring_obj, _spring_con,
        best_known=0.012665,
        reference="Belegundu (1982); Arora (1989)",
    ),
    EngineeringProblem(
        "Speed Reducer Design", 7,
        [2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0], [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5],
        _speed_reducer_obj, _speed_reducer_con,
        best_known=2996.348165,
        reference="Golinski (1970); Mezura-Montes & Coello (2005)",
        notes="11-constraint formulation. The often-quoted 2994.4245 belongs to "
              "a reduced 7-constraint variant.",
    ),
    EngineeringProblem(
        "Three-Bar Truss Design", 2,
        [1e-8, 1e-8], [1.0, 1.0],
        _three_bar_obj, _three_bar_con,
        best_known=263.895843,
        reference="Ray & Saini (2001)",
    ),
    EngineeringProblem(
        "Cantilever Beam Design", 5,
        [0.01] * 5, [100.0] * 5,
        _cantilever_obj, _cantilever_con,
        best_known=1.339956,
        reference="Chickermane & Gea (1996)",
    ),
    EngineeringProblem(
        "Gear Train Design", 4,
        [12.0] * 4, [60.0] * 4,
        _gear_train_obj, _gear_train_con,
        best_known=2.7009e-12,
        reference="Sandgren (1990) - integer teeth counts, bound-constrained",
    ),
    EngineeringProblem(
        "I-Beam Vertical Deflection", 4,
        [10.0, 10.0, 0.9, 0.9], [50.0, 80.0, 5.0, 5.0],
        _ibeam_obj, _ibeam_con,
        best_known=0.0130741,
        reference="Gold & Krishnamurty (1997) - single-constraint (area) variant",
        notes="x* = (50, 80, 0.9, 2.3217). Adding the bending-stress constraint "
              "(ibeam_stress_constraint) makes x* infeasible; that variant "
              "optimises to about 0.013564.",
    ),
    EngineeringProblem(
        "Tubular Column Design", 2,
        [2.0, 0.2], [14.0, 0.8],
        _tubular_obj, _tubular_con,
        best_known=26.4995,
        reference="Rao (1996); Hsu & Liu (2007)",
    ),
    EngineeringProblem(
        "Multi-Disc Clutch Brake", 5,
        [60.0, 90.0, 1.0, 600.0, 2.0], [80.0, 110.0, 3.0, 1000.0, 9.0],
        _clutch_obj, _clutch_con,
        best_known=0.235242,
        reference="Osyczka (2002); Deb & Srinivasan (2006)",
    ),
    EngineeringProblem(
        "Corrugated Bulkhead Design", 4,
        [0.0, 0.0, 0.0, 0.0], [100.0, 100.0, 100.0, 5.0],
        _bulkhead_obj, _bulkhead_con,
        best_known=6.842958,
        reference="Ravindran, Ragsdell & Reklaitis (2006)",
    ),
    EngineeringProblem(
        "Himmelblau Nonlinear Design", 5,
        [78.0, 33.0, 27.0, 27.0, 27.0], [102.0, 45.0, 45.0, 45.0, 45.0],
        _himmelblau_obj, _himmelblau_con,
        best_known=-30665.539,
        reference="Himmelblau (1972); problem G04 of the constrained suite",
    ),
]

#: Look-up by name.
PROBLEMS_BY_NAME = {p.name: p for p in PROBLEMS}

#: Legacy tuple form consumed by ``run_engineering_problems``:
#: ``(name, penalised_callable, dim, lb, ub)``.
ENGINEERING_PROBLEMS = [
    (p.name, p.penalised(), p.dim, list(p.lb), list(p.ub)) for p in PROBLEMS
]


def get_engineering_problems(names: Sequence[str] = None) -> List[EngineeringProblem]:
    """All engineering problems, or the named subset."""
    if names is None:
        return list(PROBLEMS)
    missing = [n for n in names if n not in PROBLEMS_BY_NAME]
    if missing:
        raise ValueError(f"Unknown engineering problem(s): {missing}. "
                         f"Available: {list(PROBLEMS_BY_NAME)}")
    return [PROBLEMS_BY_NAME[n] for n in names]


# Backward-compatible module-level callables (previously plain functions).
pressure_vessel = PROBLEMS_BY_NAME["Pressure Vessel Design"].penalised()
welded_beam = PROBLEMS_BY_NAME["Welded Beam Design"].penalised()
tension_compression_spring = PROBLEMS_BY_NAME["Tension Compression Spring"].penalised()

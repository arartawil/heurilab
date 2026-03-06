"""
Engineering design problem definitions with penalty-based constraint handling.
"""

import numpy as np

PENALTY = 1e10


def pressure_vessel(x):
    """Pressure Vessel Design (4 variables)."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    cost = (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2
            + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3)
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000
    g4 = x4 - 240
    penalty = 0
    for g in [g1, g2, g3, g4]:
        if g > 0:
            penalty += PENALTY * g**2
    return cost + penalty


def welded_beam(x):
    """Welded Beam Design (4 variables)."""
    h, l, t, b = x[0], x[1], x[2], x[3]
    cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
    M = 6000 * (14 + l / 2)
    R = np.sqrt(l**2 / 4 + ((h + t) / 2)**2)
    J = 2 * (np.sqrt(2) * h * l * (l**2 / 12 + ((h + t) / 2)**2))
    tau1 = 6000 / (np.sqrt(2) * h * l)
    tau2 = M * R / J if J != 0 else 1e30
    tau = np.sqrt(tau1**2 + 2 * tau1 * tau2 * l / (2 * R) + tau2**2)
    sigma = 504000 / (t**2 * b) if (t * b) != 0 else 1e30
    Pc = 64746.022 * (1 - 0.0282346 * t) * t * b**3 if b != 0 else 0
    delta = 2.1952 / (t**3 * b) if (t * b) != 0 else 1e30
    g1 = tau - 13600
    g2 = sigma - 30000
    g3 = h - b
    g4 = 0.10471 * h**2 + 0.04811 * t * b * (14 + l) - 5
    g5 = 0.125 - h
    g6 = delta - 0.25
    g7 = 6000 - Pc
    penalty = 0
    for g in [g1, g2, g3, g4, g5, g6, g7]:
        if g > 0:
            penalty += PENALTY * g**2
    return cost + penalty


def tension_compression_spring(x):
    """Tension/Compression Spring Design (3 variables)."""
    d, D, N = x[0], x[1], x[2]
    cost = (N + 2) * D * d**2
    g1 = 1 - D**3 * N / (71785 * d**4) if d != 0 else 1e30
    g2_num = 4 * D**2 - d * D
    g2_den = 12566 * (D * d**3 - d**4)
    g2 = g2_num / g2_den + 1 / (5108 * d**2) - 1 if (g2_den != 0 and d != 0) else 1e30
    g3 = 1 - 140.45 * d / (D**2 * N) if (D * N) != 0 else 1e30
    g4 = (d + D) / 1.5 - 1
    penalty = 0
    for g in [g1, g2, g3, g4]:
        if g > 0:
            penalty += PENALTY * g**2
    return cost + penalty


# Problem configs: (name, func, dim, lb, ub)
ENGINEERING_PROBLEMS = [
    ("Pressure Vessel Design", pressure_vessel, 4,
     [0.0625, 0.0625, 10.0, 10.0], [6.1875, 6.1875, 200.0, 200.0]),
    ("Welded Beam Design", welded_beam, 4,
     [0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0]),
    ("Tension Compression Spring", tension_compression_spring, 3,
     [0.05, 0.25, 2.0], [2.0, 1.3, 15.0]),
]

"""KOA — Kepler Optimization Algorithm (Abdel-Basset et al., 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class KOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Kepler's orbital mechanics parameters
            T_ratio = t / self.max_iter
            h = 2 * (1 - T_ratio)  # gravitational coefficient

            for i in range(self.pop_size):
                r = np.random.rand()

                # Orbital distance and velocity
                R_dist = np.linalg.norm(best - X[i]) + 1e-16
                M = fitness[i] / (best_fit + 1e-16)  # mass ratio (inversely proportional to fitness)

                if r < 0.5:
                    # Kepler's first law: elliptical orbit (exploitation)
                    # Semi-major axis decreases with iteration
                    a_orbit = h * np.abs(best - X[i])
                    e = np.random.rand()  # eccentricity [0,1)
                    theta = 2 * np.pi * np.random.rand(self.dim)

                    # Orbital position update
                    r_orbit = a_orbit * (1 - e ** 2) / (1 + e * np.cos(theta) + 1e-16)
                    new_X = best + r_orbit * np.sign(np.random.randn(self.dim))
                else:
                    # Kepler's third law: gravitational attraction (exploration)
                    j = np.random.randint(self.pop_size)
                    F_grav = M / (R_dist ** 2 + 1e-16)
                    F_grav = min(F_grav, 10)  # cap force
                    r1 = np.random.rand(self.dim)
                    new_X = X[i] + h * r1 * F_grav * (best - X[i]) + (1 - h) * np.random.randn(self.dim) * (X[j] - X[i])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence

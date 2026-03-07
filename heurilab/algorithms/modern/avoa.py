"""AVOA — African Vultures Optimization Algorithm (Abdollahzadeh et al., 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class AVOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        best1 = X[sorted_idx[0]].copy()
        best1_fit = fitness[sorted_idx[0]]
        best2 = X[sorted_idx[1]].copy()

        convergence = [best1_fit]

        for t in range(self.max_iter):
            # Starvation rate decreases linearly
            F = 2 * np.random.rand() * (1 - t / self.max_iter) - (1 - t / self.max_iter)
            P1 = 0.6  # probability to select best vulture
            P2 = 0.4
            P3 = 0.6

            for i in range(self.pop_size):
                # Select reference vulture
                R_i = best1.copy() if np.random.rand() < P1 else best2.copy()

                if abs(F) >= 1:
                    # Exploration phase
                    if np.random.rand() < P2:
                        # Random exploration
                        r1 = np.random.randint(self.pop_size)
                        D = np.abs(R_i - X[r1])
                        X[i] = R_i - D * F
                    else:
                        # Levy flight exploration
                        X[i] = R_i - F + np.random.rand(self.dim) * (self.ub - self.lb) * np.random.randn()
                else:
                    # Exploitation phase
                    if abs(F) >= 0.5:
                        if np.random.rand() < P3:
                            # Rotating flight
                            D = np.abs(F * R_i - X[i])
                            X[i] = D * np.cos(2 * np.pi * np.random.rand(self.dim)) - D * np.sin(2 * np.pi * np.random.rand(self.dim)) + R_i
                        else:
                            # Siege fight
                            d1 = R_i - X[i]
                            S1 = R_i * (np.random.rand(self.dim) * d1 / (2 * np.pi)) * np.cos(d1)
                            S2 = R_i * (np.random.rand(self.dim) * d1 / (2 * np.pi)) * np.sin(d1)
                            X[i] = R_i - (S1 + S2)
                    else:
                        # Aggressive competition
                        A1 = best1 - (best1 * X[i]) / (best1 - X[i] ** 2 + 1e-16) * F
                        A2 = best2 - (best2 * X[i]) / (best2 - X[i] ** 2 + 1e-16) * F
                        X[i] = (A1 + A2) / 2

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            sorted_idx = np.argsort(fitness)

            if fitness[sorted_idx[0]] < best1_fit:
                best1 = X[sorted_idx[0]].copy()
                best1_fit = fitness[sorted_idx[0]]
            best2 = X[sorted_idx[1]].copy()

            convergence.append(best1_fit)

        return best1, best1_fit, convergence

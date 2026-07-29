"""
ED — Enterprise Development-inspired metaheuristic

Reference:
    Dinh-Nhat Truong, Jui-Sheng Chou.
    "Metaheuristic algorithm inspired by enterprise development for global
    optimization and structural engineering problems with frequency constraints",
    Engineering Structures, 2024.
    https://doi.org/10.1016/j.engstruct.2024.118679
"""

import math
import numpy as np
from heurilab.algorithms.base import _Base


class ED(_Base):
    def optimize(self):
        pop = self._init_pop()                                       # Eq.(1)
        fitness = np.array([self._eval(pop[i]) for i in range(self.pop_size)])

        best_idx = int(np.argmin(fitness))
        best = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for g in range(1, self.max_iter + 1):
            ct = 1 - self.rng.random() * g / self.max_iter            # c(t)

            if self.rng.random() < 0.1:
                # --- Task phase — Eq.(2) ---
                worst_idx = int(np.argmax(fitness))
                sol = self.rng.uniform(self.lb, self.ub)
                f_sol = self._eval(sol)
                pop[worst_idx] = sol
                fitness[worst_idx] = f_sol
                if f_sol < best_fit:
                    best_fit = f_sol
                    best = sol.copy()
            else:
                # --- Select phase via c(t) — Eq.(9) ---
                a = min(math.ceil(3 * ct), 3)

                if a == 1:
                    # Structure phase — Eq.(3)
                    for i in range(self.pop_size):
                        idx3 = self.rng.choice(self.pop_size, 3, replace=False)
                        centre = (pop[idx3[0]] + pop[idx3[1]] + pop[idx3[2]]) / 3
                        sol = pop[i] + 2 * (self.rng.random(self.dim) - 0.5) * (best - centre)
                        sol = self._clip(sol)
                        f_sol = self._eval(sol)
                        if f_sol < best_fit:
                            best = sol.copy()
                            best_fit = f_sol

                elif a == 2:
                    # Technology phase — Eq.(5)
                    for i in range(self.pop_size):
                        h = self.rng.integers(self.pop_size)
                        sol = pop[i] + (self.rng.random(self.dim) * (best - pop[i])
                                        + self.rng.random(self.dim) * (best - pop[h]))
                        sol = self._clip(sol)
                        f_sol = self._eval(sol)
                        if f_sol <= fitness[i]:
                            pop[i] = sol
                            fitness[i] = f_sol
                            if f_sol < best_fit:
                                best = sol.copy()
                                best_fit = f_sol

                else:  # a == 3
                    # People phase — Eq.(6)
                    for i in range(self.pop_size):
                        change = self.rng.integers(self.dim)
                        nb = self.rng.choice(self.pop_size, 3, replace=False)
                        sol = pop[i].copy()
                        avg3 = (pop[nb[0], change] + pop[nb[1], change] + pop[nb[2], change]) / 3
                        sol[change] = pop[i, change] + (best[change] - avg3) * (self.rng.random() - 0.5) * 2
                        sol = self._clip(sol)
                        f_sol = self._eval(sol)
                        if f_sol <= fitness[i]:
                            pop[i] = sol
                            fitness[i] = f_sol
                            if f_sol < best_fit:
                                best = sol.copy()
                                best_fit = f_sol

            convergence.append(best_fit)

        return best, best_fit, convergence

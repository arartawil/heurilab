"""ALO — Ant Lion Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class ALO(_Base):
    def optimize(self):
        X = self._init_pop()  # Ants
        antlions = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
        al_fitness = np.array([self._eval(antlions[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(al_fitness)
        elite = antlions[best_idx].copy()
        elite_fit = al_fitness[best_idx]

        convergence = [elite_fit]

        for t in range(self.max_iter):
            ratio = t / self.max_iter
            I = 1 + 1e10 * ratio if ratio > 0.95 else (1 + 1e6 * ratio if ratio > 0.9 else 1 + 10 ** (2 * ratio))

            for i in range(self.pop_size):
                # Select antlion via roulette wheel
                probs = 1.0 / (al_fitness - np.min(al_fitness) + 1e-16)
                probs = probs / np.sum(probs)
                al_idx = np.random.choice(self.pop_size, p=probs)

                # Random walks around selected antlion and elite
                lb_al = antlions[al_idx] / I
                ub_al = antlions[al_idx] + (self.ub - self.lb) / I

                lb_el = elite / I
                ub_el = elite + (self.ub - self.lb) / I

                rw_al = np.random.uniform(lb_al, ub_al, self.dim)
                rw_el = np.random.uniform(lb_el, ub_el, self.dim)

                X[i] = self._clip((rw_al + rw_el) / 2.0)

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            # Replace antlions if ants are fitter
            for i in range(self.pop_size):
                if fitness[i] < al_fitness[i]:
                    antlions[i] = X[i].copy()
                    al_fitness[i] = fitness[i]

            min_idx = np.argmin(al_fitness)
            if al_fitness[min_idx] < elite_fit:
                elite = antlions[min_idx].copy()
                elite_fit = al_fitness[min_idx]

            convergence.append(elite_fit)

        return elite, elite_fit, convergence

"""
RIME — A physics-based optimization (Su, Zhao, Heidari, Liu, Zhang, Mafarja & Chen, 2023)

Reference:
    Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang,
    Majdi Mafarja, Huiling Chen.
    "RIME: A physics-based optimization", Neurocomputing, Elsevier, 2023.
    http://www.aliasgharheidari.com/RIME.html
"""

import numpy as np
from heurilab.algorithms.base import _Base


class RIME(_Base):
    def optimize(self):
        W = 5  # Soft-rime parameter (Section 4.3.1)

        # Initialise population and evaluate
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(1, self.max_iter + 1):
            # Eq.(3)-(5): RimeFactor
            rime_factor = ((self.rng.random() - 0.5) * 2
                           * np.cos(np.pi * t / (self.max_iter / 10))
                           * (1 - round(t * W / self.max_iter) / W))

            # Eq.(6): E
            E = np.sqrt(t / self.max_iter)

            new_X = X.copy()

            # Eq.(7): normalised rime rates (row-norm, equivalent to MATLAB normr)
            norm_val = np.linalg.norm(fitness)
            if norm_val == 0:
                normalised_rates = np.zeros(self.pop_size)
            else:
                normalised_rates = fitness / norm_val

            # Vectorised soft-rime search strategy — Eq.(3)
            r1 = self.rng.random((self.pop_size, self.dim))
            soft_mask = r1 < E
            rand_vals = self.rng.random((self.pop_size, self.dim))
            soft_update = best + rime_factor * ((self.ub - self.lb) * rand_vals + self.lb)
            new_X = np.where(soft_mask, soft_update, new_X)

            # Vectorised hard-rime puncture mechanism — Eq.(7)
            r2 = self.rng.random((self.pop_size, self.dim))
            hard_mask = r2 < normalised_rates[:, None]
            new_X = np.where(hard_mask, best, new_X)

            for i in range(self.pop_size):
                # Boundary absorption
                new_X[i] = self._clip(new_X[i])

                new_fit = self._eval(new_X[i])

                # Positive greedy selection
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    X[i] = new_X[i]
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best = X[i].copy()

            convergence.append(best_fit)

        return best, best_fit, convergence

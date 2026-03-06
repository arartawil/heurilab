"""SOS — Symbiotic Organisms Search"""

import numpy as np
from heurilab.algorithms.base import _Base


class SOS(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # ── Mutualism Phase ──
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                mutual_vector = (X[i] + X[j]) / 2
                BF1 = np.random.randint(1, 3)
                BF2 = np.random.randint(1, 3)

                new_Xi = X[i] + np.random.rand(self.dim) * (best - mutual_vector * BF1)
                new_Xj = X[j] + np.random.rand(self.dim) * (best - mutual_vector * BF2)
                new_Xi = self._clip(new_Xi)
                new_Xj = self._clip(new_Xj)

                fit_Xi = self._eval(new_Xi)
                fit_Xj = self._eval(new_Xj)

                if fit_Xi < fitness[i]:
                    X[i] = new_Xi
                    fitness[i] = fit_Xi
                if fit_Xj < fitness[j]:
                    X[j] = new_Xj
                    fitness[j] = fit_Xj

                # ── Commensalism Phase ──
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                new_Xi = X[i] + np.random.uniform(-1, 1, self.dim) * (best - X[j])
                new_Xi = self._clip(new_Xi)
                fit_Xi = self._eval(new_Xi)

                if fit_Xi < fitness[i]:
                    X[i] = new_Xi
                    fitness[i] = fit_Xi

                # ── Parasitism Phase ──
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                parasite = X[i].copy()
                n_modify = np.random.randint(1, self.dim + 1)
                dims_to_modify = np.random.choice(self.dim, n_modify, replace=False)
                parasite[dims_to_modify] = np.random.uniform(
                    self.lb[dims_to_modify], self.ub[dims_to_modify])
                fit_parasite = self._eval(parasite)

                if fit_parasite < fitness[j]:
                    X[j] = parasite
                    fitness[j] = fit_parasite

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence

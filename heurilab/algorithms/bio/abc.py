"""ABC — Artificial Bee Colony"""

import numpy as np
from heurilab.algorithms.base import _Base


class ABC(_Base):
    def optimize(self):
        limit = self.pop_size * self.dim  # Abandonment limit
        n_employed = self.pop_size // 2
        n_onlooker = self.pop_size - n_employed

        X = np.random.uniform(self.lb, self.ub, (n_employed, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(n_employed)])
        trial_counter = np.zeros(n_employed, dtype=int)

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # ── Employed Bees ──
            for i in range(n_employed):
                k = np.random.randint(n_employed)
                while k == i:
                    k = np.random.randint(n_employed)
                j = np.random.randint(self.dim)
                phi = np.random.uniform(-1, 1)

                new_X = X[i].copy()
                new_X[j] = X[i, j] + phi * (X[i, j] - X[k, j])
                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit <= fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit
                    trial_counter[i] = 0
                else:
                    trial_counter[i] += 1

            # ── Onlooker Bees ──
            fit_vals = 1 / (1 + fitness) if np.all(fitness >= 0) else 1 / (1 + np.abs(fitness))
            probs = fit_vals / np.sum(fit_vals)

            for _ in range(n_onlooker):
                i = np.random.choice(n_employed, p=probs)
                k = np.random.randint(n_employed)
                while k == i:
                    k = np.random.randint(n_employed)
                j = np.random.randint(self.dim)
                phi = np.random.uniform(-1, 1)

                new_X = X[i].copy()
                new_X[j] = X[i, j] + phi * (X[i, j] - X[k, j])
                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit <= fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit
                    trial_counter[i] = 0
                else:
                    trial_counter[i] += 1

            # ── Scout Bees ──
            for i in range(n_employed):
                if trial_counter[i] > limit:
                    X[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[i] = self._eval(X[i])
                    trial_counter[i] = 0

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence

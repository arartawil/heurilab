"""INFO — Weighted Mean of Vectors (Information-based Algorithm)"""

import numpy as np
from heurilab.algorithms.base import _Base


class INFO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        worst_idx = np.argmax(fitness)

        convergence = [best_fit]

        for t in range(self.max_iter):
            alpha = 2 * np.exp(-(4 * t / self.max_iter) ** 2)

            for i in range(self.pop_size):
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()

                # Select three random vectors
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)

                # Weighted mean direction
                eps = 1e-16
                w_a = 1.0 / (fitness[a] + eps)
                w_b = 1.0 / (fitness[b] + eps)
                w_c = 1.0 / (fitness[c] + eps)
                w_sum = w_a + w_b + w_c

                mean_vec = (w_a * X[a] + w_b * X[b] + w_c * X[c]) / w_sum

                if r1 < 0.5:
                    # Rule 1: Update via weighted mean and best
                    z = np.random.randint(self.dim)
                    new_X = X[i].copy()
                    new_X[z] = X[i, z] + alpha * r2 * (best[z] - np.abs(mean_vec[z]))
                else:
                    # Rule 2: Update using mean-based perturbation
                    new_X = mean_vec + alpha * r3 * (best - X[i])

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

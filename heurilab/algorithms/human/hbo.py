"""HBO — Heap-Based Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class HBO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Build min-heap ordering
            sorted_idx = np.argsort(fitness)

            for i in range(self.pop_size):
                rank = np.where(sorted_idx == i)[0][0]

                # Parent in heap
                parent = (rank - 1) // 2 if rank > 0 else 0
                parent_idx = sorted_idx[parent]

                # Left and right children in heap
                left = 2 * rank + 1
                right = 2 * rank + 2

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                if rank == 0:
                    # Root: small random perturbation
                    scale = (self.ub - self.lb) * 0.01 * (1 - t / self.max_iter)
                    new_X = X[i] + (2 * r1 - 1) * scale
                elif left < self.pop_size and right < self.pop_size:
                    # Has children: move toward parent and away from worst child
                    left_idx = sorted_idx[left]
                    right_idx = sorted_idx[right]
                    worst_child = left_idx if fitness[left_idx] > fitness[right_idx] else right_idx

                    new_X = X[i] + r1 * (X[parent_idx] - X[i]) + r2 * (X[i] - X[worst_child])
                else:
                    # Leaf: move toward parent
                    new_X = X[i] + r1 * (X[parent_idx] - X[i]) + r2 * (best - X[i])

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

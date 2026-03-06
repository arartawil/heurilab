"""BSO — Brain Storm Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class BSO(_Base):
    def optimize(self):
        p_replace = 0.2   # Probability of replacing cluster center
        p_one = 0.8       # Probability of selecting one cluster
        p_center = 0.4    # Probability of selecting center
        n_clusters = 5

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Simple K-means-like clustering
            centers = X[np.random.choice(self.pop_size, min(n_clusters, self.pop_size), replace=False)].copy()
            labels = np.zeros(self.pop_size, dtype=int)

            for _ in range(3):  # Few iterations of assignment
                for i in range(self.pop_size):
                    dists = np.linalg.norm(centers - X[i], axis=1)
                    labels[i] = np.argmin(dists)
                for c in range(len(centers)):
                    mask = labels == c
                    if np.any(mask):
                        centers[c] = np.mean(X[mask], axis=0)

            # Find best in each cluster
            cluster_bests = {}
            for c in range(len(centers)):
                mask = np.where(labels == c)[0]
                if len(mask) > 0:
                    cluster_bests[c] = mask[np.argmin(fitness[mask])]

            # Randomly replace a cluster center
            if np.random.rand() < p_replace and len(cluster_bests) > 0:
                c = np.random.choice(list(cluster_bests.keys()))
                centers[c] = np.random.uniform(self.lb, self.ub, self.dim)

            # Generate new individuals
            k = 20 * (1 - t / self.max_iter)  # Decreasing step size

            for i in range(self.pop_size):
                if np.random.rand() < p_one:
                    # Select one cluster
                    c = labels[i]
                    if np.random.rand() < p_center:
                        new_X = centers[c] + np.random.randn(self.dim) * k
                    else:
                        idx = np.random.choice(np.where(labels == c)[0])
                        new_X = X[idx] + np.random.randn(self.dim) * k
                else:
                    # Combine two clusters
                    available = list(cluster_bests.keys())
                    if len(available) >= 2:
                        c1, c2 = np.random.choice(available, 2, replace=False)
                        r = np.random.rand()
                        new_X = r * centers[c1] + (1 - r) * centers[c2] + np.random.randn(self.dim) * k
                    else:
                        new_X = X[i] + np.random.randn(self.dim) * k

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

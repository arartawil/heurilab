"""ICA — Imperialist Competitive Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class ICA(_Base):
    def optimize(self):
        n_imp = max(2, self.pop_size // 10)  # Number of imperialists
        zeta = 0.1  # Assimilation coefficient

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        best = X[sorted_idx[0]].copy()
        best_fit = fitness[sorted_idx[0]]

        # Split into imperialists and colonies
        imp_idx = sorted_idx[:n_imp].tolist()
        col_idx = sorted_idx[n_imp:].tolist()

        # Assign colonies to empires based on imperialist power
        empire_map = {}  # imperialist index -> list of colony indices
        for imp in imp_idx:
            empire_map[imp] = []

        if len(col_idx) > 0:
            imp_costs = fitness[imp_idx]
            probs = np.max(imp_costs) - imp_costs + 1e-16
            probs = probs / np.sum(probs)
            for col in col_idx:
                chosen = np.random.choice(imp_idx, p=probs)
                empire_map[chosen].append(col)

        convergence = [best_fit]

        for t in range(self.max_iter):
            for imp in list(empire_map.keys()):
                colonies = empire_map[imp]

                for col in colonies:
                    # Assimilation
                    X[col] += 2 * zeta * np.random.rand(self.dim) * (X[imp] - X[col])
                    # Revolution (small random perturbation)
                    if np.random.rand() < 0.1:
                        j = np.random.randint(self.dim)
                        X[col, j] = np.random.uniform(self.lb[j], self.ub[j])
                    X[col] = self._clip(X[col])
                    fitness[col] = self._eval(X[col])

                    # Swap if colony is better than imperialist
                    if fitness[col] < fitness[imp]:
                        X[imp], X[col] = X[col].copy(), X[imp].copy()
                        fitness[imp], fitness[col] = fitness[col], fitness[imp]

            # Competition: weakest empire loses a colony to strongest
            if len(empire_map) > 1:
                empire_powers = {}
                for imp, cols in empire_map.items():
                    col_cost = np.mean(fitness[cols]) if len(cols) > 0 else 0
                    empire_powers[imp] = fitness[imp] + 0.1 * col_cost

                weakest = max(empire_powers, key=empire_powers.get)
                strongest = min(empire_powers, key=empire_powers.get)

                if len(empire_map[weakest]) > 0:
                    transferred = empire_map[weakest].pop()
                    empire_map[strongest].append(transferred)

                # Remove empty empires
                if len(empire_map[weakest]) == 0 and weakest != strongest:
                    empire_map[strongest].append(weakest)
                    del empire_map[weakest]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence

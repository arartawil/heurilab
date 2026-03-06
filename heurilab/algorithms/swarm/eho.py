"""EHO — Elephant Herding Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class EHO(_Base):
    def optimize(self):
        n_clans = max(2, self.pop_size // 10)
        alpha_eho = 0.5  # Influence of matriarch
        beta_eho = 0.1   # Influence of clan center

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # Assign to clans
        clans = np.random.randint(0, n_clans, self.pop_size)

        convergence = [best_fit]

        for t in range(self.max_iter):
            # ── Clan updating operator ──
            for c in range(n_clans):
                members = np.where(clans == c)[0]
                if len(members) == 0:
                    continue

                # Find matriarch (best in clan)
                clan_fit = fitness[members]
                matriarch_idx = members[np.argmin(clan_fit)]
                clan_center = np.mean(X[members], axis=0)

                for idx in members:
                    if idx == matriarch_idx:
                        # Matriarch moves toward best overall
                        new_X = beta_eho * clan_center
                    else:
                        # Other elephants move toward matriarch
                        r = np.random.rand()
                        new_X = X[idx] + alpha_eho * r * (X[matriarch_idx] - X[idx])

                    new_X = self._clip(new_X)
                    new_fit = self._eval(new_X)

                    if new_fit < fitness[idx]:
                        X[idx] = new_X
                        fitness[idx] = new_fit

            # ── Separating operator: worst in each clan re-initialized ──
            for c in range(n_clans):
                members = np.where(clans == c)[0]
                if len(members) == 0:
                    continue
                worst_in_clan = members[np.argmax(fitness[members])]
                X[worst_in_clan] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[worst_in_clan] = self._eval(X[worst_in_clan])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence

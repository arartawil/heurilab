"""
Statistical tests: Wilcoxon rank-sum and Friedman with Nemenyi post-hoc.
"""

import numpy as np
from scipy.stats import ranksums, friedmanchisquare, rankdata
from typing import Dict, List, Tuple


def wilcoxon_test(proposed_vals: List[float], competitor_vals: List[float],
                  alpha: float = 0.05) -> Tuple[float, str]:
    """
    Wilcoxon rank-sum test between proposed and competitor.
    Returns (p_value, winner) where winner is 'proposed', 'competitor', or 'tie'.
    """
    if len(proposed_vals) < 2 or len(competitor_vals) < 2:
        return 1.0, "tie"
    stat, p_value = ranksums(proposed_vals, competitor_vals)
    if p_value < alpha:
        if np.mean(proposed_vals) < np.mean(competitor_vals):
            return p_value, "proposed"
        else:
            return p_value, "competitor"
    return p_value, "tie"


def friedman_test(all_data: Dict[str, Dict[str, List[float]]],
                  algo_names: List[str],
                  func_names: List[str]) -> Tuple[float, float, Dict[str, float]]:
    """
    Friedman test across all algorithms.
    all_data[func_name][algo_name] = list of fitness values.
    Returns (chi2, p_value, mean_ranks_dict).
    """
    n_funcs = len(func_names)
    n_algos = len(algo_names)

    if n_funcs < 3:
        mean_ranks = {name: 0.0 for name in algo_names}
        return 0.0, 1.0, mean_ranks

    # Rank algorithms per function
    rank_matrix = np.zeros((n_funcs, n_algos))
    for i, func in enumerate(func_names):
        means = []
        for algo in algo_names:
            vals = all_data.get(func, {}).get(algo, [0.0])
            means.append(np.mean(vals))
        rank_matrix[i, :] = rankdata(means)

    mean_ranks = {algo_names[j]: np.mean(rank_matrix[:, j]) for j in range(n_algos)}

    # Friedman test
    groups = [rank_matrix[:, j] for j in range(n_algos)]
    try:
        chi2, p_value = friedmanchisquare(*groups)
    except Exception:
        chi2, p_value = 0.0, 1.0

    return chi2, p_value, mean_ranks


def nemenyi_cd(n_algos: int, n_funcs: int, alpha: float = 0.05) -> float:
    """
    Critical difference for the Nemenyi post-hoc test (Demsar, 2006).

        CD = q_alpha * sqrt(k * (k + 1) / (6 * N))

    where ``k`` is the number of algorithms, ``N`` the number of benchmark
    functions, and ``q_alpha`` the Studentized range statistic at infinite
    degrees of freedom divided by sqrt(2).

    The tabulated values below cover the common case (alpha = 0.05,
    k <= 20). Outside that range the statistic is computed exactly from
    ``scipy.stats.studentized_range``; previously any k > 20 silently reused
    the k = 5 value, which understates the critical difference and reports
    differences as significant when they are not.
    """
    if n_funcs <= 0:
        raise ValueError("n_funcs must be positive")
    if n_algos < 2:
        # No pair exists, so no difference can be significant. Returning 0.0
        # keeps single-algorithm reports working (nemenyi_pairwise then only
        # compares an algorithm with itself, diff 0.0, which is not > 0.0).
        return 0.0

    q_values_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
    }

    q = None
    if abs(alpha - 0.05) < 1e-12:
        q = q_values_005.get(n_algos)
    if q is None:
        try:
            from scipy.stats import studentized_range
            q = float(studentized_range.ppf(1 - alpha, n_algos, np.inf) / np.sqrt(2))
        except Exception:
            # Last resort: hold the largest tabulated value rather than
            # falling back to a smaller one (conservative, never anti-conservative).
            q = q_values_005[max(q_values_005)]

    return float(q * np.sqrt(n_algos * (n_algos + 1) / (6 * n_funcs)))


def nemenyi_pairwise(mean_ranks: Dict[str, float], n_funcs: int,
                     algo_names: List[str]) -> Dict[Tuple[str, str], Tuple[float, bool]]:
    """
    Nemenyi post-hoc pairwise comparison.
    Returns dict[(algo_i, algo_j)] = (rank_diff, is_significant).
    """
    n_algos = len(algo_names)
    cd = nemenyi_cd(n_algos, n_funcs)
    results = {}
    for i in range(n_algos):
        for j in range(n_algos):
            diff = abs(mean_ranks[algo_names[i]] - mean_ranks[algo_names[j]])
            results[(algo_names[i], algo_names[j])] = (round(diff, 4), diff > cd)
    return results

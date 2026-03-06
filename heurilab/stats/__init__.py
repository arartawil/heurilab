"""
Stats — statistical tests for algorithm comparison.
"""

from heurilab.stats.tests import (
    wilcoxon_test, friedman_test, nemenyi_cd, nemenyi_pairwise,
)

__all__ = ["wilcoxon_test", "friedman_test", "nemenyi_cd", "nemenyi_pairwise"]

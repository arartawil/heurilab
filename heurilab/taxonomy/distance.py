"""
Rogers-Tanimoto distance between binary structural feature vectors.

The metric doubles the weight of disagreements relative to agreements, which is
what makes it discriminating on short binary vectors where most algorithms share
a majority of features:

.. math::

    D_{RT}(F_i, F_j) = \\frac{b + c}{a + d + b + c}
                     + \\frac{b + c}{a + d + 2(b + c)}

where, comparing two 19-bit vectors,

* ``a`` = positions where both have the feature,
* ``b`` = positions where only the first has it,
* ``c`` = positions where only the second has it,
* ``d`` = positions where neither has it.

Note this is the two-term form given as Eq. (2) of the source paper, not the
single-term Rogers-Tanimoto coefficient found in some references. It is
reproduced here so distances are comparable with the published analysis.
"""

from typing import Sequence

import numpy as np


def rogers_tanimoto(f_i: Sequence[int], f_j: Sequence[int]) -> float:
    """
    Distance between two binary feature vectors.

    Returns 0.0 for identical vectors. The maximum, reached when the vectors
    disagree everywhere, is 1.5.

    Raises
    ------
    ValueError
        If the vectors differ in length or are not binary.
    """
    a_vec = np.asarray(f_i, dtype=int)
    b_vec = np.asarray(f_j, dtype=int)
    if a_vec.shape != b_vec.shape:
        raise ValueError(f"vector length mismatch: {a_vec.shape} vs {b_vec.shape}")
    if a_vec.ndim != 1:
        raise ValueError("feature vectors must be one-dimensional")
    if not np.isin(a_vec, (0, 1)).all() or not np.isin(b_vec, (0, 1)).all():
        raise ValueError("feature vectors must be binary")

    a = int(np.sum((a_vec == 1) & (b_vec == 1)))
    b = int(np.sum((a_vec == 1) & (b_vec == 0)))
    c = int(np.sum((a_vec == 0) & (b_vec == 1)))
    d = int(np.sum((a_vec == 0) & (b_vec == 0)))

    disagree = b + c
    if disagree == 0:
        return 0.0
    total = a + d + disagree
    return disagree / total + disagree / (a + d + 2 * disagree)


def distance_matrix(vectors: Sequence[Sequence[int]]) -> np.ndarray:
    """Symmetric pairwise Rogers-Tanimoto matrix with a zero diagonal."""
    mat = np.asarray(vectors, dtype=int)
    n = mat.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = rogers_tanimoto(mat[i], mat[j])
            out[i, j] = out[j, i] = d
    return out


def condensed_distances(vectors: Sequence[Sequence[int]]) -> np.ndarray:
    """Upper-triangle distances in the condensed form ``scipy.linkage`` expects."""
    mat = np.asarray(vectors, dtype=int)
    n = mat.shape[0]
    return np.array([rogers_tanimoto(mat[i], mat[j])
                     for i in range(n) for j in range(i + 1, n)], dtype=float)


def nearest(target: Sequence[int],
            vectors: Sequence[Sequence[int]],
            names: Sequence[str],
            k: int = 5,
            exclude: str = None):
    """
    The ``k`` structurally closest entries to ``target``.

    Returns a list of ``(name, distance)`` sorted nearest first. ``exclude``
    drops one name, which is how an algorithm already in the reference set is
    compared against everything except itself.
    """
    scored = [(nm, rogers_tanimoto(target, vec))
              for nm, vec in zip(names, vectors) if nm != exclude]
    scored.sort(key=lambda pair: (pair[1], pair[0]))
    return scored[:k]

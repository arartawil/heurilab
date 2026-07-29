"""
Hierarchical clustering and dimensionality reduction over structural vectors.

Follows the source study: UPGMA (average linkage) over the Rogers-Tanimoto
distance matrix, with the number of clusters chosen to maximise the silhouette
score, plus PCA on the raw binary feature matrix to visualise the space.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from heurilab.taxonomy.distance import condensed_distances, distance_matrix


@dataclass
class Taxonomy:
    """Result of clustering a set of structural feature vectors."""
    names: List[str]
    matrix: np.ndarray                       # (n, 19) binary features
    distances: np.ndarray                    # (n, n) Rogers-Tanimoto
    linkage: np.ndarray                      # scipy linkage matrix
    labels: np.ndarray                       # cluster id per algorithm
    n_clusters: int
    silhouette: float
    pca_coords: np.ndarray = field(default=None)      # (n, 2)
    pca_loadings: np.ndarray = field(default=None)    # (19, 2)
    pca_explained: np.ndarray = field(default=None)   # (2,)

    def clusters(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for name, label in zip(self.names, self.labels):
            out.setdefault(int(label), []).append(name)
        return dict(sorted(out.items()))

    def identical_pairs(self) -> List[Tuple[str, str]]:
        """Pairs at distance exactly zero - the same procedure under two names."""
        n = len(self.names)
        return [(self.names[i], self.names[j])
                for i in range(n) for j in range(i + 1, n)
                if self.distances[i, j] == 0.0]

    def pairs_below(self, threshold: float) -> List[Tuple[str, str, float]]:
        n = len(self.names)
        out = [(self.names[i], self.names[j], float(self.distances[i, j]))
               for i in range(n) for j in range(i + 1, n)
               if self.distances[i, j] < threshold]
        return sorted(out, key=lambda t: t[2])

    def summary(self) -> Dict[str, float]:
        n = len(self.names)
        iu = np.triu_indices(n, k=1)
        d = self.distances[iu]
        return {
            "n_algorithms": n,
            "n_pairs": int(d.size),
            "n_clusters": self.n_clusters,
            "silhouette": float(self.silhouette),
            "mean_distance": float(np.mean(d)),
            "median_distance": float(np.median(d)),
            "min_distance": float(np.min(d)),
            "max_distance": float(np.max(d)),
            "q1": float(np.percentile(d, 25)),
            "q3": float(np.percentile(d, 75)),
            "iqr": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "distinct_vectors": len({tuple(row) for row in self.matrix}),
        }

    def __repr__(self):
        return (f"Taxonomy(n={len(self.names)}, clusters={self.n_clusters}, "
                f"silhouette={self.silhouette:.3f})")


def _silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score from a precomputed distance matrix."""
    n = len(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= n:
        return -1.0
    scores = []
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = distances[i, same].mean()
        b = min(distances[i, labels == other].mean()
                for other in unique if other != labels[i])
        denom = max(a, b)
        scores.append(0.0 if denom == 0 else (b - a) / denom)
    return float(np.mean(scores))


def build_taxonomy(names: Sequence[str],
                   vectors: Sequence[Sequence[int]],
                   k_range: Tuple[int, int] = (2, 100)) -> Taxonomy:
    """
    Cluster structural vectors with UPGMA and pick ``k`` by silhouette.

    Parameters
    ----------
    names : sequence of str
    vectors : sequence of 19-bit sequences
    k_range : (int, int)
        Inclusive range of cluster counts to search, as in the source study.

    Returns
    -------
    Taxonomy
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    mat = np.asarray(vectors, dtype=int)
    if mat.shape[0] < 3:
        raise ValueError("clustering needs at least three algorithms")

    dist = distance_matrix(mat)
    link = linkage(condensed_distances(mat), method="average")   # UPGMA

    lo, hi = k_range
    hi = min(hi, mat.shape[0] - 1)
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(max(2, lo), hi + 1):
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        score = _silhouette(dist, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_labels is None:
        best_labels = fcluster(link, t=2, criterion="maxclust")
        best_k, best_score = 2, _silhouette(dist, best_labels)

    tax = Taxonomy(names=list(names), matrix=mat, distances=dist, linkage=link,
                   labels=best_labels, n_clusters=best_k, silhouette=best_score)
    tax.pca_coords, tax.pca_loadings, tax.pca_explained = pca(mat)
    return tax


def pca(matrix: Sequence[Sequence[int]], n_components: int = 2):
    """
    Principal components of the binary feature matrix.

    Returns ``(coords, loadings, explained_variance_ratio)``. Implemented with a
    plain SVD so the module carries no scikit-learn dependency.
    """
    mat = np.asarray(matrix, dtype=float)
    centred = mat - mat.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    total = float(np.sum(s ** 2)) or 1.0
    k = min(n_components, vt.shape[0])
    coords = u[:, :k] * s[:k]
    loadings = vt[:k].T
    explained = (s[:k] ** 2) / total
    return coords, loadings, explained


def criterion_frequencies(matrix: Sequence[Sequence[int]]) -> np.ndarray:
    """Fraction of algorithms satisfying each criterion.

    Values at 0.0 or 1.0 mark criteria that carry no information for this
    particular reference set and therefore cannot separate anything.
    """
    return np.asarray(matrix, dtype=float).mean(axis=0)

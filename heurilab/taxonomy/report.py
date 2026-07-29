"""
Reports and figures for the taxonomic analysis.

Produces the four figures of the source study - dendrogram, distance heatmap,
PCA biplot and criterion-loading bars - plus markdown reports suitable for
dropping into a manuscript.
"""

import os
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heurilab.taxonomy.criteria import (
    CRITERIA, CRITERION_IDS, DISTINCTION_THRESHOLD, REDUNDANCY_THRESHOLD, SOURCE,
)


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ═════════════════════════════════════════════════════════════════════
#  Figures
# ═════════════════════════════════════════════════════════════════════

def plot_dendrogram(taxonomy, output_dir: str, filename: str = "dendrogram.png"):
    """UPGMA dendrogram, cut at the selected number of clusters."""
    from scipy.cluster.hierarchy import dendrogram
    _ensure(output_dir)
    height = max(6.0, 0.18 * len(taxonomy.names))
    fig, ax = plt.subplots(figsize=(11, height))
    dendrogram(taxonomy.linkage, labels=taxonomy.names, orientation="right",
               ax=ax, color_threshold=None, leaf_font_size=7)
    ax.axvline(REDUNDANCY_THRESHOLD, color="crimson", ls="--", lw=1.2,
               label=f"redundancy threshold ({REDUNDANCY_THRESHOLD})")
    ax.set_xlabel("Rogers-Tanimoto distance")
    ax.set_title(f"Structural taxonomy - {len(taxonomy.names)} algorithms, "
                 f"{taxonomy.n_clusters} clusters "
                 f"(silhouette {taxonomy.silhouette:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distance_heatmap(taxonomy, output_dir: str, filename: str = "distance_heatmap.png"):
    """Pairwise distance matrix, ordered by cluster."""
    _ensure(output_dir)
    order = np.argsort(taxonomy.labels, kind="stable")
    dist = taxonomy.distances[np.ix_(order, order)]
    names = [taxonomy.names[i] for i in order]
    size = max(8.0, 0.16 * len(names))
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(dist, cmap="RdYlBu_r", vmin=0.0, vmax=float(dist.max()))
    fig.colorbar(im, ax=ax, shrink=0.75, label="Rogers-Tanimoto distance")
    step = 1 if len(names) <= 60 else 2
    ax.set_xticks(range(0, len(names), step))
    ax.set_xticklabels(names[::step], rotation=90, fontsize=5)
    ax.set_yticks(range(0, len(names), step))
    ax.set_yticklabels(names[::step], fontsize=5)
    ax.set_title("Pairwise structural distance (dark blue = near-identical)")
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pca(taxonomy, output_dir: str, filename: str = "pca_biplot.png",
             highlight: Optional[str] = None):
    """PCA scatter of the binary feature matrix, coloured by cluster."""
    _ensure(output_dir)
    coords, explained = taxonomy.pca_coords, taxonomy.pca_explained
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in np.unique(taxonomy.labels):
        mask = taxonomy.labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=42, alpha=0.75,
                   label=f"cluster {label} (n={int(mask.sum())})")
    for i, nm in enumerate(taxonomy.names):
        weight = "bold" if nm == highlight else "normal"
        colour = "crimson" if nm == highlight else "0.35"
        ax.annotate(nm, (coords[i, 0], coords[i, 1]), fontsize=5.5,
                    alpha=0.9, color=colour, fontweight=weight,
                    xytext=(2, 2), textcoords="offset points")
    if highlight and highlight in taxonomy.names:
        idx = taxonomy.names.index(highlight)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=220, facecolors="none",
                   edgecolors="crimson", linewidths=2.0, zorder=5)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.set_title("Algorithms in structural feature space")
    # A registry of this size resolves into dozens of clusters, most of them
    # singletons. Listing every one produces a legend taller than the figure and
    # tells the reader nothing, so only the populated clusters are named.
    handles, labels = ax.get_legend_handles_labels()
    sizes = [int((taxonomy.labels == lab).sum()) for lab in np.unique(taxonomy.labels)]
    keep = [i for i, n in enumerate(sizes) if n > 1][:12]
    if keep and len(keep) < len(handles):
        ax.legend([handles[i] for i in keep], [labels[i] for i in keep],
                  fontsize=7, loc="best", title=f"{len(sizes) - len(keep)} further "
                  f"clusters hold one algorithm each", title_fontsize=7)
    elif handles:
        ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_criterion_loadings(taxonomy, output_dir: str,
                            filename: str = "criterion_loadings.png"):
    """Criterion weights on PC1 and PC2, and how often each criterion is met."""
    _ensure(output_dir)
    loadings = taxonomy.pca_loadings
    freq = taxonomy.matrix.mean(axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, comp, title in ((axes[0], 0, "PC1 loadings"),
                            (axes[1], 1, "PC2 loadings")):
        vals = loadings[:, comp]
        ax.bar(CRITERION_IDS, vals,
               color=["tab:blue" if v >= 0 else "tab:red" for v in vals])
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_ylabel("loading")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[2].bar(CRITERION_IDS, freq, color="tab:grey")
    axes[2].axhline(0.0, color="crimson", lw=0.8, ls="--")
    axes[2].axhline(1.0, color="crimson", lw=0.8, ls="--")
    axes[2].set_ylabel("fraction meeting it")
    axes[2].set_title("Criterion prevalence - values at 0 or 1 cannot separate anything")
    axes[2].grid(axis="y", alpha=0.25)
    plt.setp(axes[2].get_xticklabels(), rotation=45)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
#  Markdown reports
# ═════════════════════════════════════════════════════════════════════

def taxonomy_markdown(taxonomy, vectors: Dict = None) -> str:
    """Full taxonomy write-up as markdown."""
    s = taxonomy.summary()
    lines = [
        "# Structural Taxonomy of the Algorithm Set", "",
        f"Criteria framework: {SOURCE}", "",
        "## Summary", "",
        "| metric | value |", "|---|---|",
        f"| algorithms | {s['n_algorithms']} |",
        f"| pairwise comparisons | {s['n_pairs']} |",
        f"| distinct feature vectors | {s['distinct_vectors']} |",
        f"| clusters (silhouette-optimal) | {s['n_clusters']} |",
        f"| silhouette score | {s['silhouette']:.3f} |",
        f"| mean distance | {s['mean_distance']:.4f} |",
        f"| median distance | {s['median_distance']:.4f} |",
        f"| Q1 / Q3 / IQR | {s['q1']:.3f} / {s['q3']:.3f} / {s['iqr']:.3f} |",
        f"| min / max | {s['min_distance']:.4f} / {s['max_distance']:.4f} |",
        "",
    ]

    identical = taxonomy.identical_pairs()
    lines += ["## Structurally identical pairs (distance = 0)", ""]
    if identical:
        lines += [f"**{len(identical)} pair(s)** share an identical feature vector — "
                  "the same procedure under two names.", "",
                  "| A | B |", "|---|---|"]
        lines += [f"| {a} | {b} |" for a, b in identical]
    else:
        lines.append("None.")
    lines.append("")

    redundant = [p for p in taxonomy.pairs_below(REDUNDANCY_THRESHOLD)
                 if p[2] > 0.0]
    lines += [f"## Pairs below the redundancy threshold ({REDUNDANCY_THRESHOLD})", ""]
    if redundant:
        lines += ["| A | B | distance |", "|---|---|---|"]
        lines += [f"| {a} | {b} | {d:.4f} |" for a, b, d in redundant]
    else:
        lines.append("None beyond the identical pairs above.")
    lines.append("")

    lines += ["## Clusters", ""]
    for label, members in taxonomy.clusters().items():
        lines.append(f"**Cluster {label}** ({len(members)}): {', '.join(sorted(members))}")
        lines.append("")

    freq = taxonomy.matrix.mean(axis=0)
    lines += ["## Criterion prevalence", "",
              "A criterion at 0.00 or 1.00 is constant across this set and "
              "therefore contributes nothing to the distances.", "",
              "| criterion | name | fraction |", "|---|---|---|"]
    for c, f in zip(CRITERIA, freq):
        flag = " ⚠ constant" if f in (0.0, 1.0) else ""
        lines.append(f"| {c.id} | {c.name} | {f:.2f}{flag} |")
    lines.append("")

    constant = [c.id for c, f in zip(CRITERIA, freq) if f in (0.0, 1.0)]
    lines += ["## How to read these numbers", ""]
    if constant:
        lines += [
            f"**{len(constant)} of 19 criteria are constant across this set "
            f"({', '.join(constant)}).** A constant criterion contributes nothing to "
            "any distance, so the effective vector length here is "
            f"{19 - len(constant)} bits, not 19. That compresses every distance and "
            "makes exact structural identity easier to reach than it would be on a "
            "set spanning discrete, single-point and multi-neighbourhood methods. "
            "Identity counts from this report are therefore **not** directly "
            "comparable with the source study's figures over its 145-algorithm set.", "",
            "The constant criteria are a real property of the library, not a "
            "measurement failure: every algorithm here is continuous, "
            "uniformly initialised and single-neighbourhood.", "",
        ]
    else:
        lines += ["All 19 criteria vary across this set, so distances use the "
                  "full vector length.", ""]
    lines += [
        "Detection is heuristic. Runtime-observed criteria (C1, C2, C4, C5, C6, "
        "C7, C10, C13) are the most reliable; source-matched criteria (C8, C9, "
        "C11, C12, C14-C19) can misfire on unusual implementations. Before "
        "publishing any pair as identical, read both implementations.", "",
    ]

    if vectors:
        lines += ["## Feature vectors", "",
                  "| algorithm | " + " | ".join(CRITERION_IDS) + " |",
                  "|---" * (len(CRITERION_IDS) + 1) + "|"]
        for nm in taxonomy.names:
            bits = vectors[nm].as_string()
            lines.append(f"| {nm} | " + " | ".join(bits) + " |")
        lines.append("")

    return "\n".join(lines)


def save_taxonomy_report(taxonomy, output_dir: str, vectors: Dict = None,
                         plots: bool = True) -> str:
    """Write the markdown taxonomy plus all four figures."""
    _ensure(output_dir)
    path = os.path.join(output_dir, "taxonomy_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(taxonomy_markdown(taxonomy, vectors))
    if plots:
        plot_dendrogram(taxonomy, output_dir)
        plot_distance_heatmap(taxonomy, output_dir)
        plot_pca(taxonomy, output_dir)
        plot_criterion_loadings(taxonomy, output_dir)
    return path


def save_novelty_report(report, output_dir: str, plots: bool = True) -> str:
    """Write a single algorithm's novelty assessment, with its position plotted."""
    _ensure(output_dir)
    path = os.path.join(output_dir, f"novelty_{report.name}.md")

    lines = [
        f"# Structural novelty assessment — {report.name}", "",
        f"Criteria framework: {SOURCE}", "",
        f"**Verdict: {report.verdict.upper()}** — {report.explanation()}", "",
        f"- Feature vector: `{report.vector.as_string()}`",
        f"- Reference set: {report.reference_label} ({report.reference_size} algorithms)",
        f"- Nearest known algorithm: **{report.nearest_name}** "
        f"(distance {report.nearest_distance:.4f})", "",
        "## Thresholds", "",
        "| range | meaning |", "|---|---|",
        f"| < {REDUNDANCY_THRESHOLD:.3f} | structurally redundant |",
        f"| {REDUNDANCY_THRESHOLD:.3f} – {DISTINCTION_THRESHOLD:.3f} | normal difference |",
        f"| > {DISTINCTION_THRESHOLD:.3f} | fundamentally different |", "",
        "## Closest known algorithms", "",
        "| rank | algorithm | distance | differing criteria |", "|---|---|---|---|",
    ]
    for rank, (nm, dist) in enumerate(report.neighbours, 1):
        diff = report.differing_criteria.get(nm, [])
        lines.append(f"| {rank} | {nm} | {dist:.4f} | "
                     f"{', '.join(diff) if diff else '*none — identical*'} |")

    lines += ["", "## Detected feature vector", "",
              "| criterion | name | value | confidence | evidence |",
              "|---|---|---|---|---|"]
    for c in CRITERIA:
        lines.append(f"| {c.id} | {c.name} | {report.vector.bits[c.id]} | "
                     f"{report.vector.confidence[c.id]:.2f} | "
                     f"{report.vector.evidence[c.id]} |")

    review = report.vector.low_confidence()
    if review:
        lines += ["", "## Criteria to review by hand", "",
                  "Correct any that are wrong with `overrides={...}` and re-run "
                  "before quoting the verdict.", "",
                  ", ".join(review)]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    if plots:
        try:
            from heurilab.taxonomy import reference_vectors, build_taxonomy
            refs = dict(reference_vectors())
            refs[report.name] = report.vector
            tax = build_taxonomy(list(refs), [v.as_array() for v in refs.values()])
            plot_pca(tax, output_dir, filename=f"novelty_{report.name}_pca.png",
                     highlight=report.name)
            plot_dendrogram(tax, output_dir,
                            filename=f"novelty_{report.name}_dendrogram.png")
        except Exception:                      # figures are a nicety, not the report
            pass
    return path

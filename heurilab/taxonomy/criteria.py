"""
The nineteen metaphor-free structural criteria.

Implements the criteria set of:

    Soto Calvo, M. & Lee, H. S. (2026). "Systematic taxonomic framework of
    metaheuristic algorithms using hierarchical clustering and structural
    criteria: how novel is the novelty?" *Artificial Intelligence Review*,
    59(61). https://doi.org/10.1007/s10462-025-11456-8

The criteria describe *how an algorithm processes and evolves solutions*, not
what it is named after. Two algorithms with identical vectors are structurally
the same search procedure however different their metaphors.

The criteria are grouped into five families:

===========  ============================================
C1           paradigm type
C2-C4        population initialization
C5-C6        solution space representation
C7-C15       solution generation and selection
C16-C19      path selection
===========  ============================================
"""

from dataclasses import dataclass
from typing import List

#: Full citation for the criteria set. Reproduced in every generated report.
SOURCE = (
    "Soto Calvo, M. & Lee, H. S. (2026). Systematic taxonomic framework of "
    "metaheuristic algorithms using hierarchical clustering and structural "
    "criteria: how novel is the novelty? Artificial Intelligence Review, 59(61). "
    "https://doi.org/10.1007/s10462-025-11456-8"
)


@dataclass(frozen=True)
class Criterion:
    """One binary structural criterion."""
    id: str
    family: str
    name: str
    description: str
    positive_examples: str
    negative_examples: str
    detection: str          # "runtime", "source", or "manual"

    def __repr__(self):
        return f"{self.id}({self.name})"


CRITERIA: List[Criterion] = [
    Criterion(
        "C1", "paradigm", "population-based",
        "Maintains and evolves multiple candidate solutions simultaneously "
        "rather than a single incumbent.",
        "GA, PSO, DE", "Simulated Annealing, Hill Climbing, Tabu Search",
        "runtime"),

    Criterion(
        "C2", "initialization", "uniform random initialization",
        "Initial solutions drawn independently and uniformly across the whole "
        "feasible box, giving unbiased coverage.",
        "DE, PSO", "Evolution Strategies (Gaussian around a point)",
        "runtime"),
    Criterion(
        "C3", "initialization", "strategic region oversampling",
        "Initial solutions concentrated in regions judged more likely to hold "
        "the optimum, via heuristic or preliminary analysis.",
        "some ABC variants, Electrical Storm Optimization", "PSO, DE",
        "manual"),
    Criterion(
        "C4", "initialization", "non-uniform initialization",
        "Initial solutions drawn from a shaped distribution (Gaussian, Levy, "
        "Cauchy, Beta) rather than uniform.",
        "Evolution Strategies, GWO adaptive coefficients, WOA spiral",
        "DE, PSO", "runtime"),

    Criterion(
        "C5", "representation", "continuous solution space",
        "Solutions are real-valued vectors in a product of intervals.",
        "DE, PSO, ES", "classical GA on bit strings, Ant Colony Optimization",
        "runtime"),
    Criterion(
        "C6", "representation", "discrete solution space",
        "Solutions are binary strings, integer vectors or permutations.",
        "classical GA, ACO", "DE, PSO", "runtime"),

    Criterion(
        "C7", "generation", "stochastic perturbation",
        "New candidates are produced by adding controlled randomness to "
        "existing ones.",
        "GA mutation, DE scaling factors",
        "Hill Climbing, Nelder-Mead, pure gradient methods", "runtime"),
    Criterion(
        "C8", "generation", "deterministic perturbation",
        "Solutions modified by a predetermined transformation rule rather than "
        "a random draw.",
        "RUN (Runge-Kutta steps), Nelder-Mead reflection/expansion/contraction",
        "GA, PSO", "source"),
    Criterion(
        "C9", "generation", "direct solution combination",
        "New candidates are built by combining components of two or more "
        "existing solutions.",
        "GA crossover, DE trial vectors",
        "PSO (guides velocity, does not merge vectors), SA, Hill Climbing",
        "source"),
    Criterion(
        "C10", "generation", "solution reinitialization",
        "Stagnating or converged members are restarted from fresh positions "
        "during the run.",
        "Shuffled Complex Evolution, ABC scout bees, Electrical Storm Optimization",
        "standard DE, standard GA", "runtime"),
    Criterion(
        "C11", "selection", "fitness-absolute selection",
        "Selection probability is a direct function of the raw objective value.",
        "GA roulette wheel, fitness-proportional PSO variants",
        "DE tournament, rank-based ES", "source"),
    Criterion(
        "C12", "selection", "ranking-based selection",
        "Selection depends only on relative ordering, not on fitness magnitude.",
        "DE tournament selection, rank-based ES",
        "GA roulette wheel, fitness-proportional PSO", "source"),
    Criterion(
        "C13", "selection", "elitist selection",
        "The best solutions are explicitly preserved so quality never regresses.",
        "GWO (alpha/beta/delta), PSO personal and global bests",
        "standard Simulated Annealing, basic GA", "runtime"),
    Criterion(
        "C14", "generation", "adaptive diversity control",
        "Parameters or operators are adjusted in response to the measured state "
        "of the population, not merely to the iteration counter.",
        "Marine Predators Algorithm, adaptive-inertia PSO variants",
        "standard DE, basic GA, SA", "source"),
    Criterion(
        "C15", "generation", "population migration",
        "The population is split into subpopulations that evolve separately and "
        "periodically exchange members.",
        "Biogeography-Based Optimization, island-model GAs",
        "standard GA, standard DE", "source"),

    Criterion(
        "C16", "path", "single neighbourhood",
        "One unified neighbourhood structure throughout, rather than "
        "systematically alternating between several.",
        "PSO, DE",
        "Variable Neighbourhood Search, Large Neighbourhood Search, Iterated Local Search",
        "source"),
    Criterion(
        "C17", "path", "multi-agent coordination",
        "Individuals share information and influence each other's trajectories, "
        "producing emergent collective behaviour.",
        "PSO (personal/global best), ACO (pheromone)",
        "GA, DE, Evolution Strategies", "source"),
    Criterion(
        "C18", "path", "role-differentiated subpopulations",
        "Distinct functional roles are assigned to different parts of the "
        "population (explorers vs exploiters, leaders vs followers).",
        "GWO (alpha/beta/delta), Marine Predators Algorithm",
        "PSO, GA, DE", "source"),
    Criterion(
        "C19", "path", "history-based optimization",
        "An explicit memory of past solutions, trajectories or performance "
        "guides later decisions.",
        "Tabu Search, SHADE/L-SHADE success histories, Coral Reef Optimization",
        "GA, DE, SA", "source"),
]

assert len(CRITERIA) == 19, "the framework defines exactly nineteen criteria"

#: Ordered criterion ids, the canonical bit order of a feature vector.
CRITERION_IDS = [c.id for c in CRITERIA]

#: Look-up by id.
CRITERIA_BY_ID = {c.id: c for c in CRITERIA}

#: Criteria grouped by family, in the order the source paper presents them.
FAMILIES = {}
for _c in CRITERIA:
    FAMILIES.setdefault(_c.family, []).append(_c.id)


# ── Interpretation thresholds ────────────────────────────────────────
#
# From the source study's distance analysis over 145 algorithms: the pairwise
# Rogers-Tanimoto distances have a 95% confidence interval of (0.040, 0.680),
# which partitions the range into three regimes.

#: Below this, two algorithms are structurally the same procedure.
REDUNDANCY_THRESHOLD = 0.040

#: Above this, an algorithm is a fundamentally different search approach.
DISTINCTION_THRESHOLD = 0.680

VERDICTS = {
    "identical": "Structurally identical - the same procedure under another name.",
    "redundant": "Below the redundancy threshold - a variant, not a new algorithm.",
    "incremental": "Within the normal range but close to a known algorithm.",
    "normal": "Within the expected range of algorithmic difference.",
    "distinct": "Above the distinction threshold - a fundamentally different approach.",
}


def describe(criterion_id: str) -> str:
    """Human-readable description of one criterion."""
    c = CRITERIA_BY_ID[criterion_id]
    return (f"{c.id} - {c.name} ({c.family})\n"
            f"  {c.description}\n"
            f"  yes: {c.positive_examples}\n"
            f"  no : {c.negative_examples}\n"
            f"  detected by: {c.detection}")

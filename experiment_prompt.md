# Prompt: Metaheuristic Experiment Infrastructure — CSV, Plots & Statistical Analysis

I have a metaheuristic optimization research project. I need to build the full experiment infrastructure: runner, CSV outputs, plots, and statistical Excel files.

## Algorithm Interface

All algorithms share the same interface:

```python
class Algorithm:
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func): ...
    def optimize(self) -> (best_solution, best_fitness, convergence_list): ...
```

I will provide N algorithm classes. Each returns a convergence list of length `max_iter+1` (one value per iteration, index 0 = initial fitness).

## Benchmark Suites

I will provide benchmark functions organized into categories (e.g. Classical, CEC2017, CEC2022). Each benchmark has: name, objective function, lower bound, upper bound, dimension.

## Experiment Settings

- Population size, max iterations, dimension, and number of independent runs are configurable
- Default: Pop=50, MaxIter=300, Dim=30, Runs=30

---

## OUTPUT 1: Experiment Runner → 3 CSV Files

For each algorithm × function × run, call `optimize()` and record results.

**results.csv** — one row per algorithm-function combo:

```
Benchmark, Algorithm, Mean, Std, Best, Worst, Median
```

Mean/Std/Best/Worst/Median computed over all runs' BestFitness values.

**raw_runs.csv** — one row per individual run:

```
Benchmark, Algorithm, Run, BestFitness, Time_s, Conv_0, Conv_1, ..., Conv_{MaxIter}
```

Conv_i = best fitness at iteration i for that run.

**convergence.csv** — mean convergence per algorithm-function pair (averaged over all runs):

```
Benchmark, Algorithm, Iter_0, Iter_1, ..., Iter_{MaxIter}
```

Flush CSVs after each algorithm-function combo completes. Pad or trim convergence to exactly MaxIter+1 entries.

---

## OUTPUT 2: Convergence Curve Plots (one PNG per function)

- Read from convergence.csv or compute mean from raw_runs.csv
- All algorithms on same plot
- Assign each algorithm a unique color and marker
- The proposed algorithm (first in the list): thicker line (linewidth=2.5), higher zorder, plotted last so it appears on top
- Y-axis: `symlog` scale (linthresh=1e-10)
- Markers every ~MaxIter/10 iterations
- Figure size 10×6 inches, 150 DPI
- Labels: x="Iteration", y="Fitness (log scale)", title=benchmark name
- Legend: upper right, fontsize=7, ncol=2

---

## OUTPUT 3: Box Plots (one PNG per function)

- Read from raw_runs.csv (BestFitness column, 30 values per algorithm)
- All algorithms side-by-side using `boxplot(patch_artist=True)`
- Same color per algorithm as convergence plots
- The proposed algorithm box highlighted: higher alpha, thicker edge, distinct edge color
- showmeans=True (diamond marker, black fill)
- `symlog` y-scale when max/min ratio > 1000
- Figure size 14×6 inches, 150 DPI
- x-labels rotated 30°

---

## OUTPUT 4: Excel Files (3 workbooks, styled with openpyxl)

### Results.xlsx

- Sheets per benchmark category + "All Functions" sheet
- Rows = functions, Columns = Mean±Std per algorithm
- Best mean per row highlighted in green fill
- Header row: blue fill, white bold text
- The proposed algorithm column: light blue fill
- Final "Ranking" sheet: average rank per algorithm across all functions, win count (how many functions each algo had the best mean)

### Wilcoxon.xlsx

- Wilcoxon rank-sum test: proposed algorithm vs each competitor (α=0.05)
- Uses raw_runs.csv BestFitness values (30 values per algo per function)
- Sheets per category + "Summary"
- Per function: p-value, winner
- Color-coded: green = proposed wins (significantly better), red = proposed loses, yellow = no significant difference
- Summary sheet: Win/Tie/Loss counts per competitor

### Friedman.xlsx

- Friedman test across all algorithms
- Sheet 1: mean rank per algorithm per category + overall
- Sheet 2: Nemenyi post-hoc pairwise comparison matrix (N×N)
- Significant differences highlighted
- Friedman chi-square statistic and p-value displayed

---

## OUTPUT 5: Engineering Design Problems (proposed algorithm only)

3 constrained optimization problems using penalty function approach (penalty=1e10):

- Pressure Vessel Design (4 variables)
- Welded Beam Design (4 variables)
- Tension/Compression Spring Design (3 variables)

Settings: Pop=50, MaxIter=500, Runs=30

Output per problem:

- Styled Excel: sheets for Summary (Best/Mean/Worst/Std), Optimal Variables, All 30 Runs, Convergence data
- Convergence curve PNG
- Box plot PNG

---

## Dependencies

numpy, scipy (ranksums, friedmanchisquare, rankdata), matplotlib, openpyxl

## Output Folder Structure

```
output/
├── CSV Data/           # results.csv, raw_runs.csv, convergence.csv
├── Convergence Curves/ # One PNG per benchmark function
├── Box Plots/          # One PNG per benchmark function
└── Excel Files/        # Results.xlsx, Wilcoxon.xlsx, Friedman.xlsx, Engineering.xlsx
```

## Requirements

The script should:

1. Accept a list of `(name, AlgorithmClass)` tuples and a list of benchmark configs
2. The first algorithm in the list is treated as the "proposed" algorithm (highlighted in plots, tested against others in Wilcoxon)
3. Generate all outputs automatically

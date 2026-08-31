# Changelog

All notable changes to HeuriLab are recorded here.
This project follows [Semantic Versioning](https://semver.org/).

## [2.4.0] — 2026-08-31

### Correct, verified CEC 2017 and CEC 2022 suites

HeuriLab's two routes to the CEC benchmarks both produced numbers that were
**not** the official ones. This release adds a verified implementation, keeps
the old routes working, and makes both of them say so.

#### What was wrong

**`heurilab.core.cec2017` was never official CEC 2017.** It draws its shift
vectors at runtime from `numpy.random.RandomState(seed).uniform(-80, 80)` and
applies **no rotation matrices at all**. Rotation is what makes the CEC
functions non-separable, so without it the suite is substantially easier and
coordinate-wise algorithms score far better than they should. The hybrids use
no shuffle permutations and the compositions use neither the official
sub-function shifts nor their rotations.

**`get_cec2017_opfunu_suite()` was not a fix either.** `opfunu` ships the
organisers' data files — `shift_data_*.txt`, `M_*_D*.txt`, `shuffle_data_*.txt`
are byte-identical to the official `input_data/` — but its *implementations* of
the functions disagree with the organisers' reference C. Checked at 1000 random
points per function against the compiled reference, `opfunu` 1.0.1 agrees with
official CEC 2017 only on **F1**, and with official CEC 2022 only on **F2**.
Representative defects:

- `opfunu.utils.operator.zakharov_func` computes `sum(0.5 * x)`; the official
  definition is `sum(0.5 * i * x_i)` with a 1-based `i`. That breaks CEC 2017
  F3, CEC 2022 F1, and every hybrid and composition built on Zakharov.
- **CEC 2022 F9–F12** z-transform *every* sub-function with `f_shift[0]` while
  weighting sub-function *i* with `f_shift[i]`. The reference uses `&Os[i*nx]`
  for both, so each sub-landscape is centred on its own optimum; with
  `f_shift[0]` everywhere they collapse onto one point and the composition
  stops being a composition. (The rotation-matrix slice
  `f_matrix[i*ndim:(i+1)*ndim]` is correct and matches `&Mr[i*nx*nx]`; it is
  the *rotation flags* that are wrong — the reference leaves F9's fifth term
  and F10's first term unrotated, `opfunu` rotates both.)
- CEC 2022 F9's bent-cigar term is scaled by `1e-6` instead of `10000/1e30`,
  and F11's five sub-function lambdas are in the wrong order entirely.

### Added

- **`heurilab.core.cec_official`** — a faithful NumPy port of the organisers'
  reference C (`cec17_test_func.cpp` from P-N-Suganthan/CEC2017-BoundContrained
  and `cec22_test_func.cpp` from P-N-Suganthan/2022-SO-BO), evaluated over
  `opfunu`'s official data files. Exposes `OfficialCEC(year, func_num, ndim)`,
  `official_bias()` and `supported_dimensions()`.

  The reference's own quirks are reproduced deliberately, because the published
  competition results were produced with them: `schaffer_F7_func` reads the
  unrotated buffer `y` instead of the rotated `z` it just computed (so rotation
  has no effect on CEC 2017 F6 / CEC 2022 F3, and inside a hybrid it reads the
  head of the permuted vector rather than its own sub-block);
  `step_rastrigin_func`'s rounding is dead code overwritten by the following
  `sr_func` call; and the composition weight sentinel is the finite `1e99`, not
  an infinity.

- **`heurilab.core.cec2022_fixed`** — official CEC 2022 F1–F12.
  `get_cec2022_suite(ndim)` returns a `BenchmarkSuite` with official numbering.
  `CEC2022F1` … `CEC2022F12` subclass their `opfunu` counterparts (so
  `isinstance`, `lb`/`ub`, `dim_supported` and `n_fe` keep working) and override
  `evaluate()`.

- **`heurilab.core.cec2017_fixed`** — official CEC 2017 F1, F3–F30.
  `get_cec2017_official_suite(ndim)`, plus `CEC2017F1` … `CEC2017F30`
  subclassing `opfunu`'s classes, under **official** numbering (`opfunu` packs
  the suite into a contiguous `F1..F29`, so its `F3` is the official F4).

- **`tests/test_cec_validity.py`** — for every CEC 2017 and CEC 2022 function at
  D = 10, 20, 30 and 50, four hard assertions: `f(x*)` equals the bias exactly;
  `scipy.optimize.differential_evolution` inside the box never dips below it;
  200,000 uniform samples never dip below it; and the value matches the
  organisers' reference C at 1000 identical points.
  `tests/cec_reference.py` downloads and compiles that reference on demand
  (`pip install ziglang` is enough — no system toolchain needed);
  `HEURILAB_CEC_REQUIRE_REFERENCE=1` turns "could not build it" from a skip
  into a failure.

- `pytest-xdist` and `ziglang` added to the `dev` extra. The validity matrix is
  worth running as `pytest tests/test_cec_validity.py -n auto`.

### Changed

- `heurilab.core.cec2017`'s suite constructors now emit a `UserWarning` stating
  that the suite is CEC2017-*inspired*, has no official rotation data, and is
  not comparable with published CEC 2017 results. Pass `official=False` to
  silence it if that is deliberately what you want. **Nothing was deleted** —
  the functions, their values and their names are unchanged, so earlier results
  remain reproducible.
- `get_cec2017_opfunu_suite()` and `get_cec2022_opfunu_suite()` warn likewise
  and point at the corrected routes; `official=False` silences them.
- `heurilab.analyzer`'s CEC2017 diagnostic battery documents that it is the
  inspired suite. The Enhancement Advisor's scores are unaffected — they are
  only ever compared against other HeuriLab runs on the same battery.

### Fixed

- **`pip install heurilab[cec]` produced an unusable install.** `opfunu` 1.0.x
  does `import pkg_resources` without declaring `setuptools`; Python ≥ 3.12
  venvs no longer preinstall it; and `setuptools` ≥ 81 removed `pkg_resources`
  outright. On a clean environment `opfunu` therefore installed successfully and
  then failed to import. The `cec` and `all` extras now pin `setuptools<81`.
  This affected 2.3.0 too — anyone hitting it can also just run
  `pip install 'setuptools<81'`.
- When `opfunu` is present but unimportable, HeuriLab said "opfunu is not
  installed" and told you to install it. It now reports the real cause and the
  actual fix.

### Known limitations (properties of the official data, not of this port)

- **CEC 2022 is defined only at D = 2, 10 and 20**, so the requested D = 30 and
  D = 50 do not exist for it. F6–F8 (the hybrids) are additionally undefined at
  D = 2.
- **CEC 2017 ships no D = 20 data** for F11–F19, F29 and F30; the competition
  ran at D = 10, 30, 50 and 100. `get_cec2017_official_suite(ndim=20)` yields 18
  of the 29 functions and reports what it skipped.
- **CEC 2017 F9's optimum is not at its shift vector.** The reference Levy uses
  `w = 1 + (z - 1)/4`, so it bottoms out at `z = 1`; `x*` is `o + M⁻¹·1` and
  `f(o)` is roughly `900 + 0.09·D`. `OfficialCEC.x_global` accounts for this.
- **CEC 2017 F10 at D = 50 returns `1000 + 1.8e-11` at its own optimum**, because
  `4.189828872724338e+2 × 50` does not cancel the 50 summed Schwefel terms to
  the last bit. The reference C returns the identical value; the test asserts
  the match against the C rather than waiving the check.

## [2.3.0]

- Evaluation-budget fairness (`max_fes`, per-algorithm calibration).
- Behavioural analysis figures and measured control laws.
- Structural novelty auditing against 19 metaphor-free criteria.

# Repo-Wide Code Review - 2026-04-17

## Scope

This review was done against the root workspace `AGENTS.md` and the repo-local guidance in `docs/reference/package-structure.md`.

Review method:

- repo-wide inventory over all tracked Python source and test files
- static scans across the full tree for legacy markers, duplication candidates, and terminology drift
- manual line-by-line review of the full visualization package and the highest-risk large modules/tests

The root `AGENTS.md` expectations used as the review standard were:

- keep modules small, cohesive, and explicit
- prefer required config plus fail-fast errors over hidden fallbacks
- avoid stale compatibility scaffolding where it is no longer needed
- keep examples, tests, and docs aligned
- keep terminology consistent around `setup`, `project`, and `step`

Repo inventory snapshot at review time:

- 165 Python files
- largest production files:
  - `openamundsen_da/methods/viz/plots/result_overview.py` - 1748 LOC
  - `openamundsen_da/methods/pf/plot_weights.py` - 1392 LOC
  - `openamundsen_da/pipeline/project.py` - 1293 LOC
  - `openamundsen_da/methods/viz/maps/panel_renderers.py` - 1189 LOC
  - `openamundsen_da/methods/viz/plots/project_ensemble.py` - 1169 LOC
- largest tests:
  - `tests/unit/test_project_maps.py` - 2463 LOC
  - `tests/unit/test_plot_result_overview.py` - 1175 LOC
  - `tests/unit/test_plot_weights.py` - 949 LOC
  - `tests/unit/test_benchmark_rendering.py` - 905 LOC

## Findings

### 1. High: visualization ownership is still fragmented outside `methods.viz`

The repo now has a good `methods.viz.{plots,maps}` structure, but figure-producing code is still spread across multiple non-viz packages:

- `openamundsen_da/methods/pf/plot_weights.py:1-1392`
- `openamundsen_da/methods/pf/plot_ess_timeline.py:1-428`
- `openamundsen_da/methods/pf/plot_station_diagnostics.py:1-100`
- `openamundsen_da/observer/plot_scf_summary.py:1-171`
- `openamundsen_da/subdomain/plot.py:1-166`
- `openamundsen_da/benchmark/render/plots/core.py:1-797`

Why this matters:

- visual design tokens and plot behavior are still split across `methods/viz`, `methods/pf`, `observer`, `subdomain`, and `benchmark`
- shared helpers such as `save_figure_png`, `set_matplotlib_text_black`, and DA palette styles are already imported across those boundaries, which is a strong signal that ownership is still not clean
- it increases the chance of style drift and duplicated layout helpers

Recommendation:

- adopt one explicit rule: if a module's primary job is to render figures, it should live under `openamundsen_da.methods.viz`
- move PF/observer/subdomain plotting CLIs to `methods/viz/plots/` in themed subpackages, for example:
  - `methods/viz/plots/assimilation/weights.py`
  - `methods/viz/plots/assimilation/ess_timeline.py`
  - `methods/viz/plots/assimilation/station_diagnostics.py`
  - `methods/viz/plots/observer/scf_summary.py`
  - `methods/viz/plots/subdomain/overview.py`
- keep benchmark rendering under `benchmark` only if the package wants benchmark-specific ownership; otherwise fold benchmark plot rendering into `methods.viz.plots.benchmark`

### 2. High: several modules are still monoliths and conflict with the repo guidance to keep modules small and cohesive

The worst examples are:

- `openamundsen_da/methods/viz/plots/result_overview.py`
  - 55 top-level defs
  - `plot_result_overview()` alone is 476 lines at `1074-1549`
  - `cli_main()` is 193 lines at `1552-1744`
- `openamundsen_da/methods/viz/plots/project_ensemble.py`
  - `plot_setup_forcing()` is 234 lines at `525-758`
  - `plot_setup_results()` is 271 lines at `764-1034`
- `openamundsen_da/methods/viz/maps/panel_renderers.py`
  - `render_static_panel()` is 123 lines at `800-922`
  - `render_model_panel()` is 106 lines at `925-1030`
  - `render_observation_panel()` is 102 lines at `1033-1134`
- `openamundsen_da/methods/pf/plot_weights.py`
  - 1392 LOC
  - large constant block and multiple plotting responsibilities mixed into one file
- `openamundsen_da/pipeline/project.py`
  - 1293 LOC
  - orchestration, logging, plotting triggers, diagnostics, and cleanup logic remain tightly packed

Why this matters:

- these files are hard to review safely
- testing pressure moves toward private helper testing instead of public behavior
- local changes are more likely to create accidental cross-feature regressions

Recommendation:

- split `result_overview.py` into:
  - `data_loading.py`
  - `panel_specs.py`
  - `legends.py`
  - `render.py`
  - `cli.py`
- split `project_ensemble.py` into:
  - forcing plots
  - result plots
  - label-axis helpers
  - station metadata/data loading
- split `maps/panel_renderers.py` by panel family:
  - `overview_renderer.py`
  - `static_renderer.py`
  - `model_renderer.py`
  - `observation_renderer.py`
  - `legend_renderer.py`
- split `pipeline/project.py` into orchestration plus post-run hooks and plotting/diagnostics tasks

### 3. High: the test suite is too tightly coupled to private visualization internals

The clearest example is `tests/unit/test_project_maps.py:18-51`, which imports private helpers directly from `openamundsen_da.methods.viz.maps.render`, including `_apply_map_axis_style`, `_draw_scale_bar`, `_overview_extent`, `_pack_horizontal_legend_rows`, and other underscore-prefixed helpers.

That coupling is reinforced by `openamundsen_da/methods/viz/maps/render.py:432-515`, which exposes a very large `__all__` dominated by private underscore names. This is a strong sign that test support has become part of the module surface.

Additional examples:

- `tests/unit/test_plot_weights.py` directly asserts many private helpers and constants such as `_fraction_axis_label`, `_member_ticks`, `_best_figure_legend_ncol`, `_expand_xlim`, and `_STANDALONE_*`
- `tests/unit/test_plot_result_overview.py` reaches into private legend handlers and helper loaders such as `_build_result_overview_legend_handles()` and `_result_overview_legend_handler_map()`
- several tests still encode migration-era expectations such as `tests/unit/test_project_maps.py:1549` (`tighter than legacy spacing`)

Why this matters:

- refactors remain artificially expensive because private cleanup breaks tests immediately
- `render.py` and similar modules are forced to preserve implementation details instead of only behavior
- tests become harder to read than the code they are supposed to protect

Recommendation:

- reduce private exports in `maps/render.py` and test through public modules/submodules instead
- split `tests/unit/test_project_maps.py` into behavior-focused suites matching the code layout:
  - `test_maps_config.py`
  - `test_maps_data.py`
  - `test_maps_layout.py`
  - `test_maps_annotations.py`
  - `test_maps_panels.py`
  - `test_maps_cli.py`
- keep a smaller number of direct helper tests for mathematically isolated pure functions only
- move layout look-and-feel assertions away from private constants and toward image or artifact regression where practical

### 4. Medium: there is still a lot of legacy-compatibility scaffolding for an unreleased codebase

Confirmed examples:

- `openamundsen_da/pipeline/project.py:708-711` removes `project_dir/plots`
- `openamundsen_da/methods/viz/maps/runner.py:157-185` removes legacy map output trees and family directories
- `openamundsen_da/benchmark/render/plots/core.py:190-211` removes old benchmark plot locations
- `openamundsen_da/util/run_mode.py:3-10, 29-39` still reads nested legacy `data_assimilation.run_mode`
- `openamundsen_da/methods/h_of_x/model_scf.py:129-131` keeps `load_hofx_from_setup()` as an alias
- `openamundsen_da/methods/wet_snow/area.py:870-872, 1108` keeps compatibility wrappers/aliases
- `openamundsen_da/methods/pf/assimilate_fraction.py:141-146, 345-352` still supports legacy likelihood layout and wraps an older SCF-specific entrypoint
- `openamundsen_da/methods/pf/resample.py:356, 414` carries CLI and alias compatibility notes

I did not find a clearly dead production module that can be deleted with high confidence today, but I did find several compatibility layers that are likely stale if backward compatibility is no longer a real product requirement.

Recommendation:

- make a deliberate compatibility policy decision instead of keeping ad hoc compatibility forever
- if the project is still pre-release, remove transitional aliases and legacy output cleanup in one planned pass
- then prune the matching tests:
  - `tests/unit/test_run_mode.py:42`
  - `tests/unit/test_benchmark_rendering.py:60`
  - legacy cleanup assertions in benchmark/maps tests

### 5. Medium: there are still good opportunities to share functionality instead of duplicating it

#### 5a. Station metadata loading is split between map and plot stacks

Two different loaders exist:

- `openamundsen_da/methods/viz/maps/data.py:149-198`
- `openamundsen_da/methods/viz/plots/ensemble_meta.py:14-36`

They solve related problems but with different assumptions:

- map rendering loads station metadata from setup-level meteo config and can transform CRS
- plot rendering loads `stations.csv` from step ensembles

Recommendation:

- extract a neutral station metadata helper, for example under `openamundsen_da.util.station_metadata` or `openamundsen_da.methods.viz.station_meta`
- keep source-specific path discovery separate, but share normalization, CRS handling, required columns, and error semantics

#### 5b. Assimilation label-axis helpers are duplicated across the plotting stack

Confirmed duplicates:

- `openamundsen_da/methods/viz/plots/result_overview.py:661-713`
- `openamundsen_da/methods/viz/plots/project_ensemble.py:244-291`
- `openamundsen_da/methods/pf/plot_ess_timeline.py:141-155`
- `openamundsen_da/benchmark/render/plots/core.py:405-425`

These helpers all create top label axes for assimilation event numbering with near-identical logic around visible dates and axis positioning.

Recommendation:

- move this into one shared helper in `methods.viz.common` or `methods.viz.plots.common`
- parameterize only the label style and date-centering behavior

#### 5c. Fraction-series stitching still duplicates loops

`openamundsen_da/methods/viz/fraction_series.py:97-126` duplicates the same step/member traversal in both `load_member_series()` and `load_named_member_series()`.

Recommendation:

- factor out one internal stitcher returning `dict[str, pd.Series]`
- derive the list-returning function from that mapping

### 6. Medium: source-file hygiene is inconsistent and includes signs of stale editor or encoding history

Confirmed issues:

- many tracked Python files still carry a UTF-8 BOM, including:
  - `openamundsen_da/methods/pf/plot_weights.py`
  - `openamundsen_da/pipeline/project.py`
  - `openamundsen_da/util/da_events.py`
  - `openamundsen_da/core/config.py`
  - several observer/PF/core modules
- `openamundsen_da/pipeline/project.py:724` contains mojibake: `Initializing prior ensemble for step {} â€¦`
- `openamundsen_da/observer/plot_scf_summary.py:104-111` contains Windows-style path examples with invalid escape sequences such as `path\to\...`, which emits a Python warning

Recommendation:

- do one hygiene pass to remove BOMs from tracked sources
- fix mojibake and non-ASCII accidents where there is no good reason for them
- convert the `plot_scf_summary.py` docstring examples to raw strings or POSIX-style example paths

### 7. Low: docs and terminology are mostly aligned, but there are still some stale or drifting areas

Confirmed examples:

- `docs/README.md:91-97, 164` still contains TODO placeholders in the reference section
- `tests/README.md:98` still uses `scenario` phrasing (`Integration scenario behavior`)
- tutorial prose still uses `season` in natural language for project cleanup and analysis, for example in `docs/Tutorial/07-results-and-diagnostics.md:570-583`

This is not a production code problem, but it does matter because the root `AGENTS.md` explicitly tries to keep configuration terminology strict.

Recommendation:

- keep using natural-language `season` where the text really means climate season
- avoid using `scenario` or `season` in docs/tests when the text actually means a config object or directory unit
- finish the docs TODO placeholders or remove them from the published structure description

## Code That Looks Stale Or Transitional

I would not call these dead with full confidence yet, but they are the first candidates to re-evaluate:

- `openamundsen_da/methods/h_of_x/model_scf.py:129-131` - compatibility alias
- `openamundsen_da/methods/wet_snow/area.py:870-872` - compatibility wrapper
- `openamundsen_da/methods/pf/resample.py:356, 414` - compatibility CLI argument and alias
- `openamundsen_da/util/run_mode.py:34-36` - legacy nested key fallback
- legacy cleanup helpers in:
  - `pipeline/project.py`
  - `methods/viz/maps/runner.py`
  - `benchmark/render/plots/core.py`

My recommendation is to decide explicitly whether openAMUNDSEN-DA still wants these transitional contracts. If not, remove them together with their tests rather than carrying them indefinitely.

## Tests That Look Outdated Or Too Expensive

The biggest candidates for pruning or reshaping are not "bad tests", but tests that are now protecting implementation details more than behavior:

- `tests/unit/test_project_maps.py`
  - too much direct testing of private helpers/constants
  - still contains migration-era assertions like "tighter than legacy spacing"
- `tests/unit/test_plot_weights.py`
  - very high private-helper coupling
- `tests/unit/test_plot_result_overview.py`
  - mixes loader, legend, and rendering behavior into one very large file
- tests explicitly protecting legacy compatibility may be removable once the corresponding compatibility code is dropped

Recommendation:

- keep the image-regression baselines and public-CLI tests
- split large unit files to match current code structure
- reduce direct imports of underscore-prefixed helpers

## What Already Looks Good

The repo is not in bad shape overall. Several things are already moving in the right direction:

- `methods.viz` now has a real `plots` and `maps` structure
- DA style tokens are already shared through `methods.viz.theme`
- figure save/text-color helpers are already centralized in `methods.viz.common`
- the repo generally respects `setup` / `project` / `step` semantics in production code
- no obvious mass of unused imports or unreachable entire subsystems surfaced from the repo-wide scan

## Suggested Refactor Order

### Batch 1: ownership and structure

- move remaining plotting CLIs out of `methods/pf`, `observer`, and `subdomain` into `methods/viz`
- keep old entrypoints only if there is a real external compatibility requirement

### Batch 2: split the current monoliths

- `methods/viz/plots/result_overview.py`
- `methods/viz/plots/project_ensemble.py`
- `methods/viz/maps/panel_renderers.py`
- `methods/pf/plot_weights.py`
- `pipeline/project.py`

### Batch 3: retire compatibility debt

- remove legacy output cleanup if no longer needed
- remove compatibility aliases/wrappers after updating tests/callers
- simplify `run_mode` config ownership to the top-level key only

### Batch 4: test cleanup

- break up `test_project_maps.py`, `test_plot_result_overview.py`, and `test_plot_weights.py`
- shift from private-helper assertions toward public behavior and regression artifacts

### Batch 5: hygiene

- remove BOMs
- fix mojibake and invalid escape sequences
- clean docs TODO placeholders and terminology drift

## Bottom Line

The main issue is no longer obvious broken code. The main issue is structural drag:

- plotting ownership is still fragmented
- a handful of files are too large
- compatibility code is lingering
- tests are too coupled to internals

If the project wants the next big improvement, the highest-value move is:

1. finish centralizing all plotting under `methods.viz`
2. split the large visualization/orchestration modules
3. deliberately remove transitional compatibility and the tests that only exist to protect it

# openamundsen_da vs MuSA (MuSA v1.0, Alonso-Gonzalez et al. 2022)

This report summarizes similarities and differences between the openamundsen_da data-assimilation toolkit in this repo and the Multiple Snow Data Assimilation System (MuSA) cloned under `MuSA/`.

## High-level takeaways
- Both frameworks are Python-based ensemble DA toolkits for snow modeling that can ingest gridded remote-sensing products and generate posterior ensembles. Both expose particle-filter style steps and target research/operational use.
- openamundsen_da is tightly coupled to the openAMUNDSEN multilayer snow model and focuses on SCF (MODIS) and optional Sentinel-1 wet-snow DA with a streamlined, Docker-first, season-pipeline workflow and reproducible project/season folder layout.
- MuSA is a general DA laboratory for snowpack reanalyses around FSM2/Snow17/degree-index models, supports many observables (SWE, HS, LST, fSCA, albedo, turbulent fluxes), and offers a broad menu of filters/smoothers (PF/EnKF/IEnKF/ES/IES/PBS/PIES/IES-MCMC) with spatial localization and HPC/MPI options.

## Side-by-side overview
| Aspect | openamundsen_da | MuSA |
| --- | --- | --- |
| Core model(s) | openAMUNDSEN only; project/season/step YAML merged into OA config (`openamundsen_da/core/config.py`). | FSM2 (primary), Snow17, degree-index; other models pluggable via `modules/*_tools.py` (see `config.py`). |
| Target domain | Distributed catchment/ROI runs with AOI masking; assumes project folder layout with meteo stations, steps, obs. | Point-scale or distributed grids; requires all inputs/obs share identical netCDF grid. Spatial propagation/localization available. |
| Obs supported | SCF (MODIS MOD10A1) and wet-snow (Sentinel-1) fractions; SCF H(x) via depth-threshold or logistic on HS/SWE; Gaussian likelihood with optional binomial/error floors. | SWE, snow depth, land/surface temperature, fSCA, albedo, sensible/latent heat fluxes (joint assimilation allowed). |
| DA algorithms | Particle filter weighting + systematic resampling + optional rejuvenation; Gaussian likelihood; ESS-based resampling skip; no Kalman smoothers. | PF, EnKF, IEnKF, ES, IES, PBS, PIES, IES-MCMC; multiple resampling schemes (bootstrap, residual, stratified, systematic, redraw Gaussian). |
| Pipeline orchestration | CLI wrappers for prior forcing, ensemble launch, obs preprocessing, assimilation, resampling, rejuvenation, plotting; full season driver `pipeline/season.py` using step directories; Docker/Compose entrypoints; ROI/glacier masking helpers. | Single main script driven by `config.py`; optional `pre_main_spatial.py` for spatial prior sampling; MPI or multiprocessing; PBS/Slurm launcher scripts. |
| Inputs | Meteo station CSVs -> OA forcing; ROI vector (`env/roi.gpkg`); obs CSVs per date (`obs_scf_*` / `obs_wet_snow_*`); YAML configs (`project.yml`, `season.yml`, step YAMLs). | NetCDF forcing (hourly) and obs on identical grid; optional mask/dem; ensemble perturbation settings in `constants.py` and `config.py`; obs schedule `dates_obs`. |
| Outputs | Per-member OA results dirs plus assimilation weights CSVs (`weights_scf_*.csv`), resampled posterior ensembles, plots (forcing, results, SCF/wet-snow envelopes, ESS/weights). | Pickled, blosc-compressed per-cell dictionaries with DA_Results, OL_Sim, mean/std prior/post; optional ensemble dumps; supports restart/real-time outputs. |
| Parallelism | Python multiprocessing for ensemble launch; Docker resource limits; max workers clamp to CPU count; no MPI. | Multiprocessing or MPI; PBS/Slurm array compatible; configurable work chunking and numpy thread limits. |
| Licensing | MIT-style (inherits openAMUNDSEN’s GPL? check project); container-focused distribution. | GPL-3.0 with required citation (Alonso-Gonzalez et al. 2022). |

## Similarities
- Ensemble-based DA around snow models; both implement particle filters with systematic/bootstrapping-style resampling.
- Python codebases with configurable perturbations to meteo/parameters; support for parallel execution and per-member outputs.
- Can assimilate fractional snow cover (fSCA/SCF) and operate in distributed mode over a spatial grid/ROI.
- Provide reproducible examples and scripted runners; encourage conda environments.

## Key differences
### Scope and model coupling
- openamundsen_da is model-specific to openAMUNDSEN; MuSA is model-agnostic within its supported set (FSM2/Snow17/dIm) and is designed as a DA research sandbox.
- MuSA supports point-scale, distributed, and spatially localized propagation; openamundsen_da expects a project/season/step directory layout with predefined ROI vectors and assimilation dates tied to step boundaries.

### Observations and operators
- openamundsen_da focuses on SCF (MODIS MOD10A1) and optional Sentinel-1 wet-snow fractions; H(x) is limited to depth-threshold or logistic SCF mapping using modeled HS/SWE; wet-snow operator derives liquid-water fraction from OA outputs.
- MuSA supports a wider set of observables (SWE/HS/LST/fSCA/albedo/fluxes) with joint assimilation; observation errors can be spatially/temporally dynamic; supports Gaspari-Cohn localization for spatial propagation.

### DA algorithms and resampling
- openamundsen_da: single-step Gaussian likelihood + weight normalization; resampling is systematic with ESS thresholds; rejuvenation perturbs meteo forcing between steps.
- MuSA: broad menu of filters and smoothers (including iterative/ensemble Kalman variants and PBS/PIES/IES-MCMC); multiple resampling strategies; supports redraw-from-Gaussian posterior; configurable perturbation strategies (logitnormal, lognormal, etc.).

### Workflows and user interface
- openamundsen_da provides granular CLIs for forcing generation, ensemble launch, obs preprocessing, assimilation, resampling, rejuvenation, plotting, and a high-level season pipeline; strongly Docker/Compose oriented with `.env` driving mounts and resources.
- MuSA relies on a monolithic `config.py` plus `main.py` (and `pre_main_spatial.py` when spatial propagation is enabled); HPC-friendly launcher scripts (`run_PBS.sh`, `run_slurm.sh`); assumes user-prepared netCDF forcing/obs grids.

### Inputs/geometry assumptions
- openamundsen_da uses meteo station CSVs converted to OA forcing internally and supports glacier-masked ROIs; observation ingestion hinges on per-step CSVs produced by preprocessing CLIs (MODIS/Sentinel-1) with geometry handled via AOI vectors.
- MuSA requires obs and forcing on an identical netCDF grid (hourly forcing mandatory); optional mask and DEM netCDF; parallelization divides grid cells across processes/arrays.

### Outputs and diagnostics
- openamundsen_da writes per-member OA result trees plus CSV weights/ESS and curated plots (forcing/results/fraction timeseries, weights, ESS timeline). States can be chained across steps (`state_pointer.json`), enabling warm starts and rejuvenation.
- MuSA writes blosc-compressed pickle dictionaries per cell containing prior/post mean/std and observed values; optional full ensembles; focuses on numerical outputs over plotting (users post-process externally).

### Extensibility
- openamundsen_da exposes H(x) SCF parameters in `project.yml` and likelihood/resampling/rejuvenation blocks; adding new observables requires new operators and obs preprocessing scripts.
- MuSA exposes most DA knobs in `config.py`/`constants.py` (perturbations, obs errors, localization, MCMC, smoothing); additional observables or models plug in via `modules/*_tools.py` with minimal pipeline changes.

## Practical guidance
- Choose **openamundsen_da** when you need an end-to-end, reproducible SCF (and wet-snow) assimilation workflow around openAMUNDSEN with built-in obs preprocessing, Dockerized execution, and season-level orchestration.
- Choose **MuSA** when you need algorithmic flexibility (Kalman/smoother/MCMC), joint assimilation of multiple variables, or experimentation with spatial localization and different snow models (FSM2/Snow17). It is less turnkey for data prep but more configurable for research into DA methods.

## Notable implementation details (from repos)
- openamundsen_da (`openamundsen_da/methods/pf/assimilate_scf.py`): Gaussian likelihood with optional binomial cloud/error scaling; ESS reported; resampling and rejuvenation controlled via `project.yml` (`resampling`, `rejuvenation` blocks); H(x) SCF operator configurable (`data_assimilation.h_of_x`).
- MuSA (`config.py`): DA algorithm selectable (`da_algorithm`), resampling algorithms, ensemble size, perturbation strategy, localization parameters (`c`, `distance_mat_calc`, `dimension_reduction`), restart/real-time options; observables list (`var_to_assim`), obs schedule (`dates_obs`), and error covariance (`r_cov`, `dyn_noise`).

## Gaps/considerations
- openamundsen_da currently omits Kalman-style updates and joint multi-variable assimilation; extending to additional observables would require new H(x) operators and obs preprocessing pipelines.
- MuSA assumes perfectly aligned forcing/obs grids and netCDF inputs; preprocessing to this schema is left to the user. Its GPL-3.0 license and required citation may influence operational adoption.

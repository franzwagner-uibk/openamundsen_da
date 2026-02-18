---
layout: default
title: 6. Running the project
parent: Tutorial
nav_order: 6
permalink: /tutorial/running-the-project/
---

# 6. Running the project

## Project pipeline

```bash
docker run --rm -v "$(pwd):/data" \
  --cpus 8 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  "$IMAGE" \
  python -m openamundsen_da.pipeline.project \
    --setup-dir "$SETUP" \
    --project-dir "$PROJECT" \
    --max-workers 8 \
    --overwrite \
    --log-level INFO
```

## Examine the outputs

Inspect `openamundsen-da/rofental/projects/project_2022_2023` after and during the run.

### Result Plots

- `plots/results/` (time series envelopes and setup result plots)

### DA Plots

- `plots/assim/` (assimilation diagnostics, weights, ESS behavior)

### Point outputs

- `point_*_envelope.csv` files in the project root
- station and ROI AOI summaries derived from the run

### Grid outputs

- `results/grids/da_output_grids.nc` (compact DA grid summary product)
- step/member raw grid products are reduced automatically in compact retention mode

{: .note }
> Grid outputs are calculated based on the DA logic. They include the following layers:
>
> - `open_loop_<var>`: open-loop baseline (no assimilation)
> - `ens_mean_<var>`: posterior ensemble mean
> - `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`: posterior spread/range
> - `increment_<var>`: DA increment, defined as `ens_mean_<var> - open_loop_<var>`

After the baseline run, test sensitivity directly in the same project:

- Change `resolution` in `rofental.yml` (`100`, `250`, `500` m) to compare runtime vs spatial detail.
- Change `ensemble_size` in `project_2022_2023.yml` (for example `10`, `20`, `50`) to compare uncertainty vs cost.
- Change `sigma_p` and `sigma_t` to tune forcing perturbation strength.
- Change `resampling.ess_threshold_ratio` (for example `0.2`-`0.8`) to tune resampling frequency.
- Edit assimilation dates and variables in `assimilation_events`, then rerun preprocessing (`project_skeleton`, `oa-da-scf`, `oa-da-wetsnow-project`) before `oa-da-project`.

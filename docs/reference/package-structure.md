---
layout: default
title: Package Structure
parent: Reference
nav_order: 1
---

# Package Structure
{: .no_toc }

High-level module overview of openamundsen_da.
{: .fs-6 .fw-300 }

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Module Hierarchy

```
openamundsen_da/
  core/        # config merging, prior forcing, member launching
  io/          # filesystem layout helpers (YAML discovery, member paths)
  observer/    # observation summarization + season helpers
  methods/     # DA operators and plotting
    h_of_x/    # SCF forward operator (model -> obs space)
    pf/        # particle filter steps (assimilate, resample, rejuvenate, diagnostics)
    wet_snow/  # wet-snow classification and ROI fraction summaries
    viz/       # plotting utilities
  pipeline/    # end-to-end season orchestration
  util/        # ROI/land-cover masking, DA event parsing, perf monitor
```

---

## Key Entry Points

- **Season pipeline**: `openamundsen_da.pipeline.season` (`oa-da-season`)
- **Step skeleton builder**: `openamundsen_da.pipeline.season_skeleton` (module has a CLI via `python -m ...`)
- **Snow-cover summary**: `openamundsen_da.observer.snowcover` (`oa-da-snowcover`, supports GeoTIFF/NetCDF)
- **Wet-snow summary**: `openamundsen_da.observer.wetsnow` (`oa-da-wetsnow`, supports GeoTIFF/NetCDF)
- **SCF per-step obs CSVs**: `openamundsen_da.observer.satellite_scf` (`oa-da-scf`)
- **Wet-snow per-step obs CSVs**: `openamundsen_da.observer.satellite_wet_snow_s1` (`oa-da-wetsnow-season`)

For the complete, up-to-date CLI list, see [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}).

---

## Notes On Stability

The project is **CLI-first**; internal module functions are primarily maintained to support the CLI and may change. If you need a stable integration point, prefer calling the CLI entry points.

---

## Next Steps

- [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) - On-disk directory layout
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Commands and options
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) - Algorithms

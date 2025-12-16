---
layout: default
title: API Reference
parent: Reference
nav_order: 2
---

# API Reference
{: .no_toc }

Python entry points used by the CLI.
{: .fs-6 .fw-300 }

{: .note }
> **CLI-first project**: openamundsen_da is primarily designed to be used via the CLI. The Python API is not yet stabilized; treat the functions below as internal entry points that may change.

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Configuration Merging

For each ensemble member, the framework merges `project.yml`, `season.yml`, and the step YAML and then parses the result via openAMUNDSEN.

```python
from pathlib import Path
from openamundsen_da.core.config import load_merged_config

cfg = load_merged_config(
    project_yaml=Path('/data/project.yml'),
    season_yaml=Path('/data/propagation/season_2019-2020/season.yml'),
    step_yaml=Path('/data/propagation/season_2019-2020/step_00_init/00.yml'),
    member_meteo_dir=Path('/data/propagation/season_2019-2020/step_00_init/ensembles/prior/member_001/meteo'),
)
```

---

## Forcing Perturbations (Prior Ensemble)

To build an open-loop forcing copy plus a perturbed prior ensemble under a step directory:

```python
from pathlib import Path
from openamundsen_da.core.prior_forcing import build_prior_ensemble

build_prior_ensemble(
    input_meteo_dir=Path('/data/meteo'),
    project_dir=Path('/data'),
    step_dir=Path('/data/propagation/season_2019-2020/step_00_init'),
    overwrite=False,
)
```

---

## Assimilation And Resampling

Single-date weight calculation:

```python
from datetime import datetime
from pathlib import Path
from openamundsen_da.methods.pf.assimilate_scf import assimilate_scf_for_date, assimilate_wet_snow_for_date

scf_weights = assimilate_scf_for_date(
    project_dir=Path('/data'),
    step_dir=Path('/data/propagation/season_2019-2020/step_01_*'),
    ensemble='prior',
    date=datetime(2019, 11, 22),
    aoi=Path('/data/env/roi.gpkg'),
)

wet_weights = assimilate_wet_snow_for_date(
    project_dir=Path('/data'),
    step_dir=Path('/data/propagation/season_2019-2020/step_05_*'),
    ensemble='prior',
    date=datetime(2020, 4, 12),
    aoi=Path('/data/env/roi.gpkg'),
)
```

Resampling from a weights CSV to materialize a posterior ensemble:

```python
from pathlib import Path
from openamundsen_da.methods.pf.resample import resample_from_weights

resample_from_weights(
    step_dir=Path('/data/propagation/season_2019-2020/step_01_*'),
    source_ensemble='prior',
    weights_csv=Path('/data/propagation/season_2019-2020/step_01_*/assim/weights_scf_20191122.csv'),
    target_ensemble='posterior',
    overwrite=False,
)
```

---

## Performance Monitoring

The season pipeline can run a background monitor; the standalone CLI uses the same implementation:

```python
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor

stop_event = start_perf_monitor(PerfMonitorConfig(season_dir='/data/propagation/season_2019-2020'))
# ... later: stop_event.set()
```

---

## Next Steps

- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Supported command-line tools
- [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) - Module overview
- [Workflow]({{ site.baseurl }}{% link workflow.md %}) - End-to-end concepts

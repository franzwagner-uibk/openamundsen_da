---
layout: default
title: API Reference
parent: Reference
nav_order: 2
---

# API Reference
{: .no_toc }

Python API documentation for openamundsen_da.
{: .fs-6 .fw-300 }

{: .note }
> **Work in Progress**: Full API documentation is being developed. For now, see the [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) for module overview.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Core API

### Configuration Loading

```python
from openamundsen_da.core.config import load_config

# Load project configuration
config = load_config('/data/project.yml')

# Access configuration values
ensemble_size = config['ensemble']['size']
sigma_t = config['ensemble']['prior_forcing']['sigma_t']
```

### Forcing Perturbation

```python
from openamundsen_da.core.prior_forcing import perturb_forcing

# Perturb meteorological forcing
perturb_forcing(
    input_dir='/data/meteo',
    output_dir='/data/step_01/ensembles/prior/member_001/input/meteo',
    sigma_t=1.5,
    sigma_p=0.20,
    seed=42
)
```

---

## Particle Filter API

### Assimilation

```python
from openamundsen_da.methods.pf.assimilate import assimilate_scf

# Assimilate SCF observations
weights, ess = assimilate_scf(
    obs_file='step_01/obs/obs_scf_MOD10A1_20191122.csv',
    model_scf_files=[
        'step_01/ensembles/prior/member_001/results/model_scf.csv',
        'step_01/ensembles/prior/member_002/results/model_scf.csv',
        # ...
    ],
    obs_error=0.10
)
```

### Resampling

```python
from openamundsen_da.methods.pf.resampling import systematic_resampling, compute_ess

# Compute ESS
ess = compute_ess(weights)

# Resample if needed
if ess < 0.5 * len(weights):
    indices = systematic_resampling(weights, N=len(weights), seed=42)
```

---

## Forward Operators

### SCF from Model

```python
from openamundsen_da.methods.h_of_x.scf_from_model import compute_scf_logistic

# Compute SCF using logistic function
scf = compute_scf_logistic(
    snow_depth=0.15,  # meters
    h0=0.05,
    k=50.0
)
# Returns: scf ≈ 0.99
```

---

## Observation Processing

### MODIS Preprocessing

```python
from openamundsen_da.observer.mod10a1_preprocess import preprocess_mod10a1

# Preprocess MODIS HDF files
preprocess_mod10a1(
    input_dir='/data/obs/MOD10A1_61_HDF',
    output_dir='/data/obs/season_2019-2020',
    roi_path='/data/env/roi.gpkg',
    target_epsg=32632,
    ndsi_threshold=0.4
)
```

---

## Utilities

### Logging

```python
from openamundsen_da.util.logging_config import setup_logging
import logging

# Setup logging
setup_logging(log_file='season.log', level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info('Starting assimilation')
```

### Performance Monitoring

```python
from openamundsen_da.util.performance import PerformanceMonitor

# Monitor performance
monitor = PerformanceMonitor(output_dir='plots/perf')
monitor.start()

# ... your code ...

monitor.update(step=1, total_steps=10)
monitor.plot()
monitor.stop()
```

---

## Type Hints and Signatures

{: .note }
> Full type-annotated API documentation will be generated using Sphinx autodoc in future releases.

For now, refer to the source code in the repository for detailed function signatures:
- [github.com/franzwagner-uibk/openamundsen_da](https://github.com/franzwagner-uibk/openamundsen_da)

---

## Next Steps

- [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) - Module organization
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) - Algorithm details
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Command-line tools

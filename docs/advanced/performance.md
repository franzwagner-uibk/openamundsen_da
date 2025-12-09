---
layout: default
title: Performance Tuning
parent: Advanced Topics
nav_order: 1
---

# Performance Tuning
{: .no_toc }

Optimization strategies for running large ensembles efficiently.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Running large ensemble data assimilation experiments can be computationally intensive. This guide covers strategies to optimize performance and reduce runtime.

---

## Parallel Execution

### Worker Configuration

The `--max-workers` parameter controls parallel ensemble execution:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 8
```

**Optimal worker count**:
```
workers = min(max_workers, CPU_count, N_ensemble_members)
```

**Guidelines**:
- **CPU-bound tasks**: Set workers = CPU cores (e.g., 8 cores → 8 workers)
- **Memory-limited**: Reduce workers to avoid swapping
- **I/O-bound tasks**: Can exceed CPU count slightly

### Docker Resource Allocation

Configure Docker resources in `.env`:

```bash
# .env
CPUS=8              # Number of CPU cores
MEMORY=16G          # RAM allocation
```

**Memory estimation**:
```
Memory per member ≈ 500MB - 2GB (depends on domain size)
Total RAM needed ≈ N_workers × Memory_per_member + 2GB overhead
```

**Example**:
- 8 workers, 1GB/member → 8GB + 2GB = 10GB minimum

---

## Domain Optimization

### Resolution Selection

Higher resolution = more computation. Consider your science goals:

| Resolution | Grid cells (100 km²) | Typical runtime (30 members) |
|:-----------|:---------------------|:-----------------------------|
| 1000m | 10,000 | Fast (minutes) |
| 500m | 40,000 | Moderate (hours) |
| 250m | 160,000 | Slow (days) |
| 100m | 1,000,000 | Very slow (weeks) |

**Strategy**: Start coarse, refine where needed.

### Spatial Subsetting

Reduce domain size to area of interest:

```python
# Clip ROI to smaller extent
import geopandas as gpd

roi = gpd.read_file('env/roi_full.gpkg')
roi_subset = roi.cx[xmin:xmax, ymin:ymax]  # Bounding box clip
roi_subset.to_file('env/roi_subset.gpkg')
```

---

## Ensemble Configuration

### Ensemble Size Trade-offs

| Ensemble size | Computational cost | Posterior quality |
|:--------------|:-------------------|:------------------|
| N=10 | Very low | Poor (high sampling error) |
| N=20-30 | Low | Adequate (small domains) |
| N=50 | Moderate | Good |
| N=100+ | High | Excellent (diminishing returns) |

**Recommendation**: Start with N=30, increase if ESS frequently near threshold.

### Observation Thinning

Fewer assimilation cycles = less computation:

```python
# Thin to every 7 days instead of daily
import pandas as pd

dates = pd.date_range('2019-11-01', '2020-07-31', freq='7D')
```

**Impact**:
- Runtime reduced by ~7×
- Posterior quality slightly degraded
- Still captures weekly snow dynamics

---

## Model Configuration

### Timestep Selection

Longer timesteps = faster execution:

```yaml
# project.yml
timestep: 3H  # vs. 1H (3× faster)
```

**Caveats**:
- Energy balance may be less accurate with long timesteps
- Recommended: 1H for detailed studies, 3H for seasonal runs

### Output Reduction

Minimize output variables to save I/O time:

```yaml
output_data:
  grids:
    variables:
      - snow_depth          # Essential
      - snow_water_equivalent  # Essential
      # Remove unnecessary variables:
      # - surface_temperature
      # - albedo
      # - lwc
```

**Impact**: Reduces write time by 30-50%.

### Disable Unnecessary Features

```yaml
# Disable point outputs if not needed
output_data:
  timeseries:
    enabled: false
```

---

## Storage Optimization

### Compression

Enable NetCDF compression:

```yaml
output_data:
  grids:
    format: netcdf
    compression: 4  # Level 1-9 (4 is good balance)
```

**Impact**:
- File size reduced by 50-80%
- Slight increase in write time (usually worth it)

### Selective Retention

Delete intermediate results:

```bash
# After season completes, keep only posterior results
find propagation/season_2019-2020 -path "*/prior/*" -name "*.nc" -delete
```

**Caution**: Only delete if you don't need to rerun from checkpoints.

---

## Profiling

### Performance Monitoring

Enable built-in monitoring:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --monitor-perf
```

**Output**: `plots/perf/season_perf.png` (updated live)

Shows:
- CPU usage per core
- Memory consumption (RSS, system)
- Disk usage
- ETA estimation

### Identify Bottlenecks

Use Python profiling for detailed analysis:

```bash
python -m cProfile -o profile.stats \
  -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020

# Analyze
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20
```

---

## Hardware Recommendations

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 100 GB SSD

**Typical runtime** (30 members, 100 km², 10 steps): 4-8 hours

### Recommended Configuration

- **CPU**: 16+ cores (AMD Ryzen 9 / Intel i9 / Xeon)
- **RAM**: 32-64 GB
- **Storage**: 500 GB NVMe SSD

**Typical runtime**: 1-2 hours

### HPC/Cloud

For large-scale experiments:

**Options**:
- University HPC clusters (SLURM, PBS)
- Cloud platforms (AWS, GCP, Azure)
- Distributed computing (Dask, Ray)

**Parallelization strategy**:
```bash
# Run multiple seasons in parallel on different nodes
sbatch run_season_2018.sh
sbatch run_season_2019.sh
sbatch run_season_2020.sh
```

---

## Best Practices Checklist

Performance optimization checklist:

- [ ] Start with coarse resolution and small ensemble for testing
- [ ] Enable `--monitor-perf` to identify bottlenecks
- [ ] Set `--max-workers` to CPU count (or slightly less if memory-limited)
- [ ] Use 3H timestep for seasonal runs (1H only if needed)
- [ ] Thin observations to every 5-10 days if runtime is critical
- [ ] Enable NetCDF compression (level 4)
- [ ] Remove unnecessary output variables
- [ ] Allocate sufficient Docker resources (CPU, RAM)
- [ ] Use SSD storage for faster I/O
- [ ] Delete intermediate results after completion

---

## Troubleshooting Performance Issues

### Issue: High CPU but slow progress

**Cause**: I/O bottleneck (slow disk)

**Solution**:
- Move data to SSD
- Reduce output frequency
- Enable compression

### Issue: High memory usage, swapping

**Cause**: Too many workers or large grids

**Solution**:
- Reduce `--max-workers`
- Increase Docker memory limit
- Reduce ensemble size or domain resolution

### Issue: Pipeline stalls without progress

**Cause**: Deadlock or single member stuck

**Solution**:
- Check logs: `propagation/season_*/step_*/ensembles/prior/member_*/run.log`
- Identify failing member
- Check openAMUNDSEN config or forcing data

---

## Next Steps

- [Troubleshooting Guide]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) - Common issues
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) - Optimize settings
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Complete workflow

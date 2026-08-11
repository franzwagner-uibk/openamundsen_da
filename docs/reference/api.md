---
layout: default
title: API Quick Reference
parent: Reference
nav_order: 2
---

# API Quick Reference

The supported Python API contains six workflow operations. Import them from the
top-level package:

```python
from openamundsen_da import (
    clean_project,
    prepare_project,
    preprocess_snow_cover,
    preprocess_wet_snow,
    render_project,
    run_project,
)
```

## Observation preprocessing

- `preprocess_snow_cover(project_dir, *, overwrite=False)`
- `preprocess_wet_snow(project_dir, *, overwrite=False)`

Both operations read the explicit observation format and paths from the
canonical project YAML. They return an immutable
`ObservationPreprocessingResult` and write a hash-tracked manifest under
`<project>/.openamundsen-da/manifests/`.

## Project workflow

- `prepare_project(project_dir, *, overwrite=False)` creates deterministic
  steps and maps configured observations. It returns `PreparationResult`.
- `run_project(project_dir, *, max_workers=None)` runs a prepared project,
  validates required scientific and rendered outputs, applies safe restart
  cleanup and returns `RunResult`.
- `render_project(project_dir, *, max_workers=None)` regenerates all configured
  plots, maps and the project report. It returns `RenderResult`.
- `clean_project(project_dir, *, apply=False)` previews package-owned artifacts
  eligible under compact retention. Pass `apply=True` to apply the ledger-backed
  cleanup. Full-retention projects return an empty preview. It returns
  `CleanupResult`.

Result objects and `WorkflowStatus` are immutable values from
`openamundsen_da.results`. Public workflow exceptions inherit from
`openamundsen_da.exceptions.OpenAmundsenDAError`.

## Stability boundary

The six operations, their result types and the public exception hierarchy are
the supported scripting boundary. Subdomain Python modules, particle-filter
functions, observers, plotting modules and path utilities are implementation
details; use the `openamundsen-da` command for those workflows.

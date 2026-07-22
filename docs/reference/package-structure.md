---
layout: default
title: Package Structure
parent: Reference
nav_order: 3
---

# Package Structure

The compatibility boundary is intentionally smaller than the source tree.

## Public modules

- `openamundsen_da`: version, six workflow operations, typed results and public exceptions
- `openamundsen_da.api`: single-domain workflow implementation used by the top-level exports
- `openamundsen_da.observations`: the two public observation preprocessors
- `openamundsen_da.exceptions`: exception hierarchy rooted at `OpenAmundsenDAError`

The installed console surface contains only `openamundsen-da`.

## Internal modules

- `core`: model launching, forcing and shared execution utilities
- `pipeline`: preparation, execution, rendering and cleanup orchestration
- `methods`: observation operators, particle-filter stages, benchmarks and visualization
- `io`: strict path/config/model-grid adapters
- `subdomain`: distinct data assimilation and plain-model staged workflows
- `util`: shared validation, manifests, grid summaries and spatial helpers

Subdomain Python modules and lower-level method functions are internal. Their
supported user interface is the umbrella command tree.

## Filesystem ownership

The setup root owns shared openAMUNDSEN inputs. A project owns its YAML, steps,
manifest and results. Generated steps and member directories are runtime data,
not source configuration. See [Configuration]({{ site.baseurl }}{% link guides/configuration.md %})
and [Output data]({{ site.baseurl }}{% link output-data.md %}).

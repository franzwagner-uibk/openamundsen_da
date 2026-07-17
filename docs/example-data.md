---
layout: default
title: Example Data
nav_order: 7
---

# Example Data

The release image contains two shipped, versioned examples under
`/workspace/examples`. They are user-facing baselines and CI contracts, not
runtime output archives.

## Rofental

`examples/rofental` is the single-domain tutorial and scientific regression
baseline. It contains prepared openAMUNDSEN grids and forcing, snow-cover and
wet-snow products, station observations and one Rofental 2022–2023 project.
Static inputs are available at 100, 250 and 500 m; the setup YAML selects the
active resolution.

Copy it from the image before use:

```bash
docker run --rm \
  -v "/absolute/path/to/workdir:/data" \
  {{ site.data.release.image }} \
  bash -lc 'cp -a /workspace/examples/rofental /data/rofental'
```

The [How to Use tutorial]({{ '/tutorial/' | relative_url }}) works through this
existing setup incrementally. Its 13 reference images are selected from the
canonical ES30 output by a hash-checked publication manifest.

## North Tyrol subdomains

`examples/subdomains` covers a larger North Tyrol domain split into eight
avalanche-report regions. It exercises both the staged data assimilation
subdomain workflow and the separate plain openAMUNDSEN model workflow. Static
grids are available at 50, 100, 250 and 500 m.

Copy it from the same release image before use:

```bash
docker run --rm \
  -v "/absolute/path/to/workdir:/data" \
  {{ site.data.release.image }} \
  bash -lc 'cp -a /workspace/examples/subdomains /data/subdomains'
```

See the [Subdomain Runbook]({{ site.baseurl }}{% link guides/subdomain-runbook.md %})
for the explicit prepare, run, merge and render sequence.

## Adapting an example

Copy an example outside the repository, keep the setup/project ownership boundary
and replace one input class at a time. Do not copy completed `steps/`, `results/`
or logs into a new experiment. Start with a short project, coarse grid and small
ensemble before increasing computational cost.

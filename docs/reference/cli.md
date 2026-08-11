---
layout: default
title: Command-Line Interface
parent: Reference
nav_order: 1
---

# Command-Line Interface

`openamundsen-da` provides one command tree for observation preprocessing,
single-domain data assimilation and explicit subdomain workflows. Run
`openamundsen-da COMMAND --help` at any level for the complete option list.

## Paths and notation

- `PROJECT_DIR` is one data-assimilation project below
  `<setup>/projects/<project>`. It contains the project YAML, prepared steps and
  project results.
- `SETUP_DIR` is a plain openAMUNDSEN setup. It is used only by the
  `subdomains model` commands, which do not perform data assimilation.
- Uppercase values such as `COUNT` and `PATH` are placeholders. Replace them
  with values for your machine and setup; do not type the angle brackets shown
  in explanatory prose.
- Add `--json` to a leaf command when a script needs one machine-readable result
  instead of the human summary.

## Single-domain data assimilation

Preprocess the configured observations, prepare deterministic steps and run the
project:

```bash
openamundsen-da observations snow-cover PROJECT_DIR
openamundsen-da observations wet-snow PROJECT_DIR
openamundsen-da prepare PROJECT_DIR
openamundsen-da run PROJECT_DIR --max-workers COUNT
```

After a successful run, regenerate configured plots, maps and reports without
rerunning the model:

```bash
openamundsen-da render PROJECT_DIR --max-workers COUNT
```

Successful compact project runs already remove validated member forcing,
points, grids and restart state according to the configured retention policy.
Full-retention projects preserve these artifacts. To inspect or apply cleanup
manually for an older or incomplete compact project:

```bash
openamundsen-da clean PROJECT_DIR
openamundsen-da clean PROJECT_DIR --apply
```

See [Running the model]({{ site.baseurl }}{% link running.md %}) for the execution order and
[Output data]({{ site.baseurl }}{% link output-data.md %}) for completion and
cleanup semantics.

## Subdomain data assimilation

These commands split one data-assimilation project into explicit regional
projects. Each region remains an independent particle-filter problem.

```bash
openamundsen-da subdomains prepare PROJECT_DIR --regions PATH
openamundsen-da subdomains run PROJECT_DIR \
  --max-workers COUNT \
  --inner-max-workers COUNT
openamundsen-da subdomains merge PROJECT_DIR
openamundsen-da subdomains render PROJECT_DIR --max-workers COUNT
```

When `--regions` is omitted, preparation uses `env/subdomains.gpkg`, falling
back to `env/roi.gpkg`. Use outer and inner worker limits together so their
product does not oversubscribe the machine.

## Plain-model subdomains

The separate `subdomains model` branch tiles one ordinary openAMUNDSEN setup.
It has no projects, ensembles or assimilation steps.

```bash
openamundsen-da subdomains model prepare SETUP_DIR --regions PATH
openamundsen-da subdomains model run SETUP_DIR --max-workers COUNT
openamundsen-da subdomains model merge SETUP_DIR
```

## Options that can replace or relocate outputs

`--overwrite` permits the named operation to replace existing preparation or
reusable run artifacts. Use it only after checking that the existing outputs are
not the accepted result you intend to preserve.

`--out-dir PATH` redirects merged grid outputs. Without it, DA merges write below
`PROJECT_DIR/results/grids`, while plain-model merges write below
`SETUP_DIR/subdomains/model/results/grids`.

`--coverage-sliver-tol-px PIXELS` controls the permitted uncovered edge sliver
during mosaic validation. Keep the default unless you have inspected the region
geometry and grid alignment.

---
layout: default
title: 5. Pre-processing
parent: Tutorial
nav_order: 5
permalink: /tutorial/pre-processing/
---

# 5. Pre-processing

In this tutorial, preprocessing means creating step-wise project inputs from existing Rofental summaries.

Build the project skeleton:

```bash
docker run --rm -v "$(pwd):/data" \
  "$IMAGE" \
  python -m openamundsen_da.pipeline.project_skeleton \
    --setup-dir "$SETUP" \
    --project-dir "$PROJECT" \
    --overwrite \
    --log-level INFO
```

Distribute SCF and wet-snow summaries into per-step observation CSVs:

```bash
docker run --rm -v "$(pwd):/data" \
  "$IMAGE" \
  oa-da-scf \
    --project-dir "$PROJECT" \
    --summary-csv "$SCF_SUM" \
    --overwrite \
    --log-level INFO

docker run --rm -v "$(pwd):/data" \
  "$IMAGE" \
  oa-da-wetsnow-project \
    --project-dir "$PROJECT" \
    --summary-csv "$WET_SUM" \
    --overwrite \
    --log-level INFO
```

{: .note }
> The tutorial setup already ships with observation summaries, so no external download is required.
> If you want to build your own summaries from raw products, use:
> - `oa-da-snowcover` for SCF
> - `oa-da-wetsnow` for wet snow

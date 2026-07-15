---
layout: default
title: 8. Adapting to Your Own Project
parent: How to Use
nav_order: 8
permalink: /tutorial/adapting-to-your-own-project/
---

# 8. Adapting to Your Own Project

Use the Rofental tutorial as a working reference, not as a template to copy blindly.
The safest path is to keep the workflow order intact: prepare the configuration,
validate the observations, generate summaries and per-step CSVs, run a small test,
and only then increase cost or complexity.

## Start with the configuration split

When you move to a new domain, keep the separation between setup and project files
strict. `<setup>.yml` should remain pure openAMUNDSEN configuration: domain,
resolution, terrain inputs, forcing stations, and model physics. The
`project_YYYY_YYYY.yml` file is where openAMUNDSEN-DA behavior belongs: dates,
`obs.*` paths and class mappings, `assimilation_events`, likelihood settings,
resampling and rejuvenation, and output options.

Observation products deserve extra care because most migration problems begin
there. Before you run any assimilation step, make sure the product class codes are
documented, the project YAML matches them exactly, the product tags in
`assimilation_events` are identical to the configured tags, and the selected
assimilation dates actually exist in the generated summary CSVs.

{: .warning }
> Do not rely on guessed defaults for observation products. openAMUNDSEN-DA is
> designed to fail fast when required mappings or product tags are missing, and
> that behavior helps prevent silent mistakes.

For setup-level details, use the external openAMUNDSEN documentation:
[https://doc.openamundsen.org/](https://doc.openamundsen.org/). For project-level
options, use the
[Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}).

## Recommended migration path

Start by copying the Rofental example into a new setup and replacing only the
setup-level openAMUNDSEN inputs. Keep the first project YAML intentionally small:
a short date range, only a few assimilation events, and the minimum outputs needed
to confirm that the workflow runs cleanly. This keeps the structure familiar while
you change data sources.

Next, validate the observations before you spend time on assimilation tuning.
Run `openamundsen-da observations snow-cover PROJECT_DIR` and/or
`openamundsen-da observations wet-snow PROJECT_DIR`, inspect `scf_summary.csv`
and `wet_snow_summary.csv`, and choose a small set of usable dates. Once those
dates are fixed, update `assimilation_events` and run
`openamundsen-da prepare PROJECT_DIR --overwrite`. Data assimilation debugging
is much easier when coverage, class mappings and product tags are already confirmed.

Only then run a cheap data assimilation test. If possible, use a coarser grid,
fewer assimilation events, a small ensemble such as `5-10`, and a shorter
seasonal window. Treat this run as a wiring check: read the logs, inspect ESS and
weight diagnostics, open the result plots, and confirm that the output NetCDF
looks sane before increasing resolution or ensemble size.

## Choosing assimilation dates

Choose dates from the summary CSVs rather than from visual impression alone. For
fSCA, prefer scenes with acceptable cloud fraction and enough valid pixels inside
the project period. For wet snow, look for dates that capture onset or progression
of melt rather than clustering all events in one phase.

A good first pass is to keep only a few events that cover early, middle, and late
parts of the season. After each change to `assimilation_events`, rebuild the
project skeleton and regenerate the per-step observation CSVs so the step windows
stay aligned with the selected dates. The
[Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %})
goes deeper into product-specific criteria.

## Common mistakes

Most failed migrations fall into a small set of patterns. The first is mixing
configuration scopes: data assimilation settings belong in the project YAML, not
the setup YAML. The second is changing too many things at once, which makes it
hard to tell whether a problem comes from observations, forcing, geometry, or
tuning. The third is using wrong class mappings or mismatched product tags, which
usually shows up as missing or implausible summary values. Another common mistake
is editing `assimilation_events` without rebuilding the step skeleton afterward.
Finally, apparent plot mismatches are sometimes just unit mismatches, especially
for SWE, so check units before diagnosing model behavior.

## Pre-run check

{: .checks }
> Before the first full experiment, confirm that:
> - the setup YAML contains only openAMUNDSEN configuration,
> - the project YAML defines dates, `obs`, and `data_assimilation`,
> - class mappings and product tags are explicit and match the summaries,
> - selected `assimilation_events` dates exist in the summary CSVs,
> - the step skeleton and per-step observation CSVs were regenerated after the
>   last event change,
> - and a short test run completes and produces logs, plots, and NetCDF outputs.

## After the first successful run

Once the workflow is stable, change one dimension at a time and compare against
the same diagnostics. Good first experiments are `ensemble_size`, grid
resolution, observation error settings, resampling thresholds, and the number of
assimilation dates. Document each change so runtime, ESS, and result-plot
differences remain interpretable.

For deeper reference material, continue with the
[CLI Reference]({{ site.baseurl }}{% link reference/cli.md %}),
[Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}),
[Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %}),
[Running]({{ site.baseurl }}{% link running.md %}),
[Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}),
[Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %}), and
the external openAMUNDSEN docs.

## What You Should Be Able To Do After This Chapter

You should now be able to run the bundled Rofental tutorial end to end, interpret
the main data assimilation outputs, and adapt the same workflow to a new project
with controlled changes.

# Tutorial Development Guideline (openAMUNDSEN-DA)

This document defines the scope, goals, structure, and quality criteria for the **full tutorial** in the `docs/` pages.

It is intended as a working guideline while writing and iterating the tutorial content.

## Purpose

The tutorial should enable a new user to:

- understand the **openAMUNDSEN-DA framework** conceptually and practically,
- understand how **openAMUNDSEN-DA works internally** (at a useful operational level),
- understand the **observation pre-processing workflow** (especially satellite-based observations),
- run the bundled example end-to-end,
- interpret outputs (plots, tables, DA statistics, grids),
- and confidently adapt the workflow to their **own project setup**.

## Scope (Target Tutorial Character)

The tutorial should be:

- **long-form** (several hours of work for the user),
- **hands-on** (command-by-command execution),
- **deep** (not only "how", but also "why"),
- **framework-oriented** (showing architecture and workflow logic),
- **practical** (using the bundled example and reproducible commands),
- **transferable** (so users can build their own projects afterwards).

## Central Case Study (Rofental Example)

The tutorial should use the bundled **Rofental example** ([`examples/rofental`](examples/rofental))
as the central case study throughout the full workflow.

This is the reference tutorial setup and should be treated as the default reproducible
example in the documentation.

Current tutorial baseline (important for docs consistency):

- domain/grid setup: **Rofental 100 m**
- ensemble size: **10**
- tutorial project period: **snow season only (October to June)**, not full hydrological year

With this current configuration, users with a **normal computer** should be able to run
the tutorial example in practice (runtime is still non-trivial, but feasible).

For the tutorial workflow, the example should start from **raw observation rasters**
(and station observations) and **not** ship precomputed project summary CSVs as the
default learning path. Users should generate summary tables themselves in the
pre-processing chapter.

Tutorial command style should use a **single interactive Docker container session**
(one bind mount to `/data`, then commands inside the container shell) to minimize
copy-paste errors and avoid repeated host-path boilerplate in later chapters.

## Core Requirements

The tutorial must cover:

- all necessary commands to set up and run a project:
  - step-by-step commands (individual tools / CLIs),
  - and the higher-level **project pipeline** workflow,
- pre-processing of satellite imagery / observations for the framework,
- examining results and DA statistics,
- examining plots, tables, and output grids,
- cross-references to deeper documentation pages (framework, config, CLI, theory, prerequisites),
- external links where useful (e.g. openAMUNDSEN docs, upstream data/product docs).

## Tutorial Style (Notebook-like)

The tutorial should feel like a **Jupyter notebook in Markdown form**:

- short explanation blocks followed by executable commands,
- code blocks for every important command,
- inline interpretation of outputs,
- screenshots / plot snippets / graphics where useful,
- tables for overview and comparison,
- notes, warnings, and troubleshooting hints,
- explicit transitions between conceptual and practical parts.

### Writing Pattern (recommended)

For each major step, use this sequence:

1. **What we do**
2. **Why we do it**
3. **Command(s)**
4. **What to expect (output/files/logs)**
5. **How to validate success**
6. **Where to learn more** (cross-links)

## Audience Assumptions

Assume users:

- are comfortable running terminal commands,
- may be new to `openAMUNDSEN-DA`,
- may be new to the framework-specific file structure and processing logic,
- may not yet understand DA terms (ESS, weights, resampling, rejuvenation),
- need a reproducible example before adapting to their own domain/project.

Do **not** assume deep prior knowledge of the codebase.

## Tutorial Learning Outcomes

After completing the tutorial, users should be able to:

- explain the roles of **setup / project / step / ensemble members**,
- prepare a project structure and config files,
- run observation pre-processing for supported observation types,
- execute a DA project using both granular commands and the project pipeline,
- inspect and interpret:
  - DA weights / ESS / diagnostics,
  - result plots,
  - observation summaries,
  - output NetCDF/grid products,
- modify an existing example into a new project.

## Content Coverage (Required Topics)

### 1. Orientation and Prerequisites

- What openAMUNDSEN-DA is and how it relates to openAMUNDSEN
- Docker runtime and repository layout
- Compute/runtime expectations (CPU, runtime, storage)
- Where the example data and configs live

Cross-reference examples:

- installation/prerequisites docs
- framework overview docs
- openAMUNDSEN documentation (external)

### 2. Framework Concepts (Deep Insight)

- setup-level vs project-level configuration
- step concept and temporal segmentation
- prior vs posterior ensemble
- open-loop role
- observation integration points
- DA cycle overview:
  - prior run,
  - diagnostics / H(x),
  - assimilation,
  - resampling,
  - rejuvenation,
  - next step

This section should explain **how the framework works**, not only CLI usage.

### 3. Project Preparation (Manual / Granular Workflow)

Cover all core commands one by one, with explanations and expected outputs:

- project skeleton generation
- observation pre-processing CLIs
- per-step observation mapping/alignment
- project execution command
- optional/advanced commands for inspecting intermediate artifacts

The user should understand what each command produces and where files are written.

### 4. Observation Pre-processing (Important)

This is a central tutorial goal and should be treated in detail:

- raw satellite data location and expected structure
- summary generation and per-step observation CSV creation
- variable-specific handling (e.g. snow cover / wet snow)
- class mapping and configuration relevance
- land-cover masking interaction
- fail-fast behavior and common errors (missing config / missing dates / mismatched dates)

The tutorial should clearly explain that pre-processors are **generic framework tools** and how configuration controls product-specific interpretation.

### 5. Project Pipeline Workflow (High-level)

After showing the granular commands, show the **project pipeline** as the practical shortcut:

- when to use it,
- what it automates,
- how it relates to the manual steps already shown,
- how to rerun safely (`--overwrite`, selective reruns, common iteration patterns).

### 6. Results, Diagnostics, and Interpretation

Show users where to find and how to interpret:

- DA statistics (weights, ESS, timelines)
- plots (forcing, results, fractions, performance)
- summary CSVs and envelopes
- output grids / NetCDF summaries
- log files (what to scan for success/failure)

Include examples of:

- "normal/expected" outputs,
- signals of problems (missing obs, bad dates, config mismatch, empty outputs).

### 7. Adapting to Own Project

Close the tutorial with a practical transfer section:

- minimal checklist to clone/modify the example
- what must be changed first (paths, dates, ROI, obs config, assimilation events)
- what can remain unchanged initially
- recommended incremental validation strategy

## Cross-Referencing Strategy (Mandatory)

The tutorial should not duplicate all details. It should guide and link.

Use links to:

- CLI reference pages
- configuration guide
- workflow/framework pages
- observations guide
- advanced troubleshooting/performance pages
- external docs:
  - openAMUNDSEN docs
  - relevant satellite product documentation (where helpful)

### Rule of Thumb

- Tutorial page = **task flow + interpretation**
- Reference pages = **complete option/config detail**

## Content Format Requirements

Use the following elements consistently:

- fenced code blocks with commands
- fenced code blocks for config snippets
- tables for key mappings / outputs / file locations
- note/warning callouts for common pitfalls
- file path references and expected output paths
- plot screenshots/snippets with short interpretation captions

Prefer concise, high-value explanations over long generic prose.

## Final Authoring Pattern (Current Tutorial Standard)

The tutorial now uses a **repeatable callout-based structure** in the docs. Apply this
pattern consistently across all `docs/Tutorial/*.md` pages where it adds clarity.

There are two valid page patterns:

- **Strict step template** (for command-heavy pages, e.g. dependencies, preprocessing, running)
- **Hybrid template** (for interpretation-heavy pages, e.g. workflow, framework, results)

### 1. Strict Step Template (command-heavy pages)

Use this sequence for each step:

1. Step title (numbered)
2. `Goal` callout (`{: .step }`) with one concise objective sentence
3. `Commands` section with command block(s)
4. Short command explanation in normal prose directly after the command block (paths/flags/results)
5. `Checks` callout (`{: .checks }`) with bullet points for verification
6. `References` callout (`{: .references }`) with short linked bullets

Notes:

- Do not split `Goal` and `Description` if they say the same thing.
- Avoid low-value callouts that only repeat "run this command here".
- Prefer normal prose for command explanations so the page keeps a healthy text-to-callout ratio.

### 2. Hybrid Template (concept / interpretation-heavy pages)

Use normal prose for conceptual explanation and interpretation, then add callouts only
where they improve scanning:

- `{: .checks }` for “what to inspect”, “what to look for”, sanity checks
- `{: .references }` for follow-up docs and cross-links
- `{: .warning }` for failure modes or misleading outputs

Do not force the full strict step template onto narrative sections.

## Tutorial Callout Usage (Just-the-Docs)

Use the tutorial-specific callouts configured in `docs/_config.yml`:

- `{: .step }` -> **Goal**
- `{: .checks }` -> **Checks**
- `{: .references }` -> **References**
- `{: .tip }`, `{: .note }`, `{: .warning }`, `{: .highlight }` as needed

### Callout Anti-Duplication Rules

Avoid duplicated labels such as a heading immediately followed by a callout with the same title.

Examples to avoid:

- `### References` followed by `{: .references }`
- `### Checks` followed by `{: .checks }`

Preferred:

- Use the callout header only, and place the content directly inside the callout block.

## Formatting Conventions for Snippets and Paths

### Snippet / Plot Placement Rule (Important)

Every table, snippet, and plot in the tutorial must have a **local teaching purpose** at
the point where it appears.

Do not include snippets/plots only because the file exists. Each one should help answer
the reader's question in that section.

Required pattern (around the snippet/plot):

- **Context before**: why this file/plot is shown here (what concept/check it supports)
- **Artifact**: the snippet/table/plot itself
- **Interpretation after**: what to look at, what is normal/suspicious, and what it means for the next step

If a snippet/plot does not support the local section goal, move it to a more relevant
section or remove it.

### File paths (single path)

Do **not** use a bullet list for a single path under `File path`.

Use:

```md
File path: `/data/.../file.csv`
```

Avoid:

```md
File path:

- `/data/.../file.csv`
```

### CSV snippets

Render reference CSV snippets as **Markdown tables**, not fenced `csv` code blocks.

Reasons:

- easier to scan in docs
- visually consistent with “reference snippet” style
- aligns with the tutorial’s interpretation-first approach

Use fenced code blocks for CSV only when the exact raw formatting itself is important (rare).

## Links and References Style

In `References` callout boxes:

- use bullet points
- use short link text + a short phrase (what the link is for)
- avoid prefixes like `External:` / `Internal:`

Preferred:

- [Docker run reference](...) (CLI flags and bind-mount syntax)

Avoid:

- External: docker run reference

## Flowcharts and Workflow Graphics

For tutorial workflow diagrams (e.g. preprocessing flow), use **SVG assets** committed to:

- `docs/assets/images/tutorial/diagrams/`

Embed them as normal images in Markdown.

Reasons:

- works reliably in the current Jekyll/Just-the-Docs setup (no Mermaid dependency)
- version-controlled and reviewable
- reusable across pages

### Standard for Tall Process Flows (Tutorial Default)

For longer stepwise processes that are easier to read vertically (for example:
raw inputs -> preprocessing -> step-aligned files -> DA execution -> outputs), the
default tutorial visualization should be a **tall vertical SVG flowchart**.

Use this as the standard pattern for tutorial process diagrams unless there is a clear
reason to use another layout.

Current reference example:

- `docs/assets/images/tutorial/diagrams/preprocessing-observation-flow.svg`

When a page currently uses an ASCII flow diagram and the process is central to the
learning path, prefer replacing it with an SVG diagram following this standard.

### SVG style guidelines (current preference)

- monochrome (black outlines/text on white background)
- rounded rectangles
- consistent box widths/alignment in a given diagram
- simple arrows (small arrowheads)
- concise labels (move long explanation to surrounding text/caption)

### Tall Flowchart Content Guidelines

- Use short labels inside boxes (noun phrase + short qualifier)
- Keep long explanations in normal prose before/after the figure
- Use top-to-bottom sequencing for dependency chains
- Keep one process per figure (avoid mixing multiple unrelated flows)
- Add a short caption that explains what moves from one stage to the next

## Tutorial Visual Overrides (Migration Reminder)

The tutorial currently uses a small shared CSS override for code-block backgrounds:

- `docs/assets/css/tutorial-code-blocks.css`

It is loaded from:

- `docs/_includes/head_custom.html`

This is intentionally a **low-impact docs-layer customization** (not a template rewrite).
If the tutorial is migrated to another documentation system, remember to migrate or
recreate this CSS so YAML and command blocks keep their visual distinction.

## Local Preview and Authoring Environment (Current Recommended Setup)

To avoid stale output and slow rebuilds during tutorial editing:

- use **local Ruby in WSL** (not Docker) for authoring preview
- keep the repo where it is (single-repo workflow is fine)
- use polling and disable incremental builds for reliability
- write Jekyll caches/output to WSL `/tmp`

Practical implications:

- no second clone is required
- no manual sync workflow is required
- `_config.yml` changes require a Jekyll restart
- content-only Markdown changes should hot-reload

Docker preview can remain as an occasional fallback / verification mode, but should not
be the default authoring loop.

## VS Code Safety for Markdown Editing

Protect tutorial Markdown from accidental formatting changes:

- disable Markdown format-on-save in workspace settings
- disable save-time auto-fixes for Markdown
- disable Prettier for Markdown in this workspace

This is especially important for mixed Markdown + HTML blocks (`<details>`, callouts,
embedded images), where auto-formatters can break rendering.

## Quality Criteria (Acceptance Checklist)

The tutorial is "ready" when:

- a user can follow it end-to-end on a normal computer (using the example),
- all listed commands are current and tested,
- outputs shown match the current example setup,
- observation pre-processing is explained clearly enough to adapt to new products/projects,
- both manual command workflow and project pipeline workflow are covered,
- DA diagnostics and outputs are interpreted (not just listed),
- cross-links to deeper docs exist for all major concepts,
- the tutorial ends with a concrete "how to build your own project" path.

## Recommended Authoring Workflow

While writing:

1. Write/update one tutorial page section.
2. Run local docs preview.
3. Execute/verify commands from the page.
4. Check links (internal + external).
5. Capture/update plots/snippets.
6. Only then continue to the next section.

This keeps the tutorial executable and avoids drift between docs and code.

## Suggested Tutorial Page-Level Structure (for implementation)

This is a recommended sequence for the final `docs/Tutorial/` pages:

1. Introduction / goals / prerequisites
2. Dependencies and runtime setup
3. Workflow overview (conceptual)
4. Framework internals (setup/project/step/member, DA cycle)
5. Pre-processing (manual commands)
6. Running the project (manual + pipeline)
7. Inspecting results, plots, diagnostics, output grids
8. Adapting the example to your own project
9. Troubleshooting / common mistakes (or cross-link to advanced troubleshooting)

## Non-Goals (to keep the tutorial focused)

The tutorial should not try to:

- document every CLI option inline (link to reference instead),
- replace the full configuration reference,
- provide a full DA theory textbook,
- cover every supported observation source/product in one page.

It should provide a strong, reproducible workflow and enough understanding for users to continue independently.

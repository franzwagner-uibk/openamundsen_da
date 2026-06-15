# openamundsen_da Documentation

This directory contains the documentation for the openamundsen_da project, built with [Jekyll](https://jekyllrb.com/) and the [Just the Docs](https://just-the-docs.github.io/just-the-docs/) theme.

## Viewing Locally

### Prerequisites

- Ruby 2.7+ ([installation guide](https://www.ruby-lang.org/en/documentation/installation/))
- Bundler (`gem install bundler`)

### Build and Serve (Recommended: WSL Live Sync)

If your repo is on Windows storage (for example `/mnt/c/...`) and you want the preview to
stay in live sync with edits to `docs/*.md`, use the live-sync wrapper from the repo root:

```bash
./scripts/docs/jekyll_serve_wsl_live_sync.sh
```

This mirrors `docs/` into `/tmp` (fast WSL filesystem) and runs Jekyll from that mirror,
while continuously syncing changes from your repo `docs/` folder.

Then open [http://127.0.0.1:4001/](http://127.0.0.1:4001/)

### Build and Serve (Direct Jekyll, WSL Reliable / Single Repo)

```bash
cd docs
bundle install
TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
bundle exec jekyll serve --host 127.0.0.1 --port 4001 --livereload --force_polling \
  --config _config.yml,_config_dev.yml,_config_wsl_reliable.yml
```

Then open [http://127.0.0.1:4001/](http://127.0.0.1:4001/)

### VS Code One-Click Preview

This repository includes VS Code tasks in `.vscode/tasks.json`:

- `Docs: Jekyll Serve (WSL Live Sync, Recommended)` - Mirrors `docs/` into `/tmp` and keeps preview live-synced with repo markdown edits
- `Docs: Jekyll Serve (Local Ruby, WSL Reliable)` - Single-repo reliable mode for WSL + `/mnt/c` (polling, no incremental, cache/output in `/tmp`)
- `Docs: Open Local Preview` - Opens `http://127.0.0.1:4001/`

Run them from `Terminal -> Run Task...`.

### Recommended Single-Repo WSL Workflow (No Sync)

If your repository lives on Windows storage (for example under `C:\...` / `/mnt/c/...`) and
you preview from WSL, prefer `Docs: Jekyll Serve (Local Ruby, WSL Reliable)`.

This mode keeps your source files in the current repo but moves Jekyll's generated output and
cache to `/tmp` inside WSL, which reduces lag and stale preview issues without using a second clone.

If you still see stale pages while editing under `/mnt/c`, switch to the **WSL Live Sync**
task/script above. It is designed specifically to keep the preview in sync with markdown edits.

## Cloudflare Pages Deployment

Live documentation is deployed to Cloudflare Pages at:

```
https://openamundsen-da.pages.dev/
```

Treat the workflow files as the authoritative deployment source.

## Structure

```
docs/
|-- _config.yml              # Jekyll configuration
|-- Gemfile                  # Ruby dependencies
|-- index.md                 # Home page
|-- installation.md          # Installation guide
|-- project-structure.md     # Project layout
|-- workflow.md              # data assimilation workflow
|-- guides/                  # User guides
|   |-- index.md             # Guides section index
|   |-- cli.md               # CLI reference [done]
|   |-- configuration.md     # YAML configuration guide [done]
|   |-- observations.md      # Satellite observation processing [done]
|   `-- experiments.md       # End-to-end walkthrough [done]
|-- reference/               # Technical reference
|   |-- index.md             # Reference section index [done]
|   |-- package-structure.md # Package map [done]
|   |-- api.md               # API quick reference [done]
|   `-- da-methods.md        # Particle-filter methods [done]
`-- advanced/                # Advanced topics
    |-- index.md             # Advanced section index [done]
    |-- troubleshooting.md   # Common issues and solutions [done]
    `-- performance.md       # Performance notes [done]
```

## Writing Documentation

### Front Matter

All pages must include YAML front matter:

```yaml
---
layout: default
title: Page Title
nav_order: 1
parent: Parent Page  # Optional
---
```

### Navigation

- `nav_order`: Determines menu order (lower numbers appear first)
- `parent`: Creates hierarchy (page appears under parent in nav)
- `has_children: true`: For parent pages with children

### Styling

Just the Docs provides built-in classes:

```markdown
{: .highlight }
> Highlighted callout

{: .note }
> Note callout

{: .warning }
> Warning callout

{: .fs-6 .fw-300 }
Large, light text
```

See [Just the Docs documentation](https://just-the-docs.github.io/just-the-docs/) for more.

## Updating Configuration

Edit `_config.yml` to customize:

- Site title and description
- GitHub repository URL
- Color scheme
- Navigation settings
- Footer content

## Completed

- Core documentation structure with Just the Docs theme
- Installation guide (Docker and native)
- Project structure documentation
- Workflow overview with mermaid diagrams
- Complete CLI reference
- Configuration reference (comprehensive YAML guide)
- Observation processing guide (snow-cover incl. MODIS via converted rasters, Sentinel-2, Sentinel-1)
- Running experiments guide (end-to-end walkthrough)
- Troubleshooting guide (common issues and solutions)
- Section index pages (Guides, Reference, Advanced)

## Notes

- All core user-facing documentation is complete and ready for use
- Reference pages are maintained as hand-written quick references
- The site is deployed through Cloudflare Pages
- API documentation generation can be reconsidered later if direct Python API usage becomes a supported public surface

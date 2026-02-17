# openamundsen_da Documentation

This directory contains the documentation for the openamundsen_da project, built with [Jekyll](https://jekyllrb.com/) and the [Just the Docs](https://just-the-docs.github.io/just-the-docs/) theme.

## Viewing Locally

### Prerequisites

- Ruby 2.7+ ([installation guide](https://www.ruby-lang.org/en/documentation/installation/))
- Bundler (`gem install bundler`)

### Build and Serve

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Then open [http://127.0.0.1:4001/](http://127.0.0.1:4001/)

### VS Code One-Click Preview

This repository includes VS Code tasks in `.vscode/tasks.json`:

- `Docs: Jekyll Serve (Docker)` - Recommended (no local Ruby/Bundler install required)
- `Docs: Jekyll Serve (Local Ruby)` - Uses your local Ruby toolchain
- `Docs: Open Local Preview` - Opens `http://127.0.0.1:4001/`

Run them from `Terminal -> Run Task...`.

## GitHub Pages Deployment

This documentation is configured for GitHub Pages deployment:

1. Go to your repository's Settings â†’ Pages
2. Set Source to "Deploy from a branch"
3. Set Branch to "main" and folder to "/docs"
4. Click Save

GitHub will automatically build and deploy the site to:
```
https://franzwagner-uibk.github.io/openamundsen_da/
```

## Structure

```
docs/
â”œâ”€â”€ _config.yml              # Jekyll configuration
â”œâ”€â”€ Gemfile                  # Ruby dependencies
â”œâ”€â”€ index.md                 # Home page
â”œâ”€â”€ installation.md          # Installation guide
â”œâ”€â”€ project-structure.md     # Project layout
â”œâ”€â”€ workflow.md              # DA workflow
â”œâ”€â”€ guides/                  # User guides
â”‚   â”œâ”€â”€ index.md            # Guides section index
â”‚   â”œâ”€â”€ cli.md              # CLI reference (16 commands) âœ…
â”‚   â”œâ”€â”€ configuration.md    # YAML configuration guide âœ…
â”‚   â”œâ”€â”€ observations.md     # Satellite observation processing âœ…
â”‚   â””â”€â”€ experiments.md      # End-to-end walkthrough âœ…
â”œâ”€â”€ reference/               # Technical reference
â”‚   â”œâ”€â”€ index.md            # Reference section index âœ…
â”‚   â”œâ”€â”€ package-structure.md # (TODO)
â”‚   â”œâ”€â”€ api.md              # (TODO)
â”‚   â””â”€â”€ da-methods.md       # (TODO)
â””â”€â”€ advanced/                # Advanced topics
    â”œâ”€â”€ index.md            # Advanced section index âœ…
    â”œâ”€â”€ troubleshooting.md  # Common issues and solutions âœ…
    â””â”€â”€ performance.md      # (TODO)
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

## Completed âœ…

- âœ… Core documentation structure with Just the Docs theme
- âœ… Installation guide (Docker and native)
- âœ… Project structure documentation
- âœ… Workflow overview with mermaid diagrams
- âœ… Complete CLI reference (all 16 commands)
- âœ… Configuration reference (comprehensive YAML guide)
- âœ… Observation processing guide (snow-cover incl. MODIS via converted rasters, Sentinel-2, Sentinel-1)
- âœ… Running experiments guide (end-to-end walkthrough)
- âœ… Troubleshooting guide (common issues and solutions)
- âœ… Section index pages (Guides, Reference, Advanced)

## TODOs

### High Priority

- [ ] Replace franzwagner-uibk with actual GitHub username in:
  - [ ] `_config.yml` (url and aux_links)
  - [ ] `index.md` (GitHub button link)
  - [ ] `installation.md` (clone command)

### Reference Pages

- [ ] `reference/api.md` - Python API documentation
- [ ] `reference/da-methods.md` - Particle filter implementation details
- [ ] `reference/package-structure.md` - Module architecture and design

### Advanced Pages

- [ ] `advanced/performance.md` - Optimization strategies
  - Ensemble size tuning
  - Parallelization best practices
  - Memory optimization
  - Disk I/O optimization

### General

- [ ] `contributing.md` - Contributing guidelines
  - Code style
  - Testing requirements
  - PR process
- [ ] Consider adding API documentation generation (Sphinx autodoc integration?)
- [ ] Add example notebooks/tutorials?
- [ ] Add search configuration tuning if needed

## Notes

- All core user-facing documentation is complete and ready for use
- Reference section is stubbed for future technical API docs
- The site is fully functional for GitHub Pages deployment
- Consider integrating Sphinx for automated API docs from docstrings



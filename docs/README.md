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

Then open [http://localhost:4000/openamundsen_da/](http://localhost:4000/openamundsen_da/)

## GitHub Pages Deployment

This documentation is configured for GitHub Pages deployment:

1. Go to your repository's Settings → Pages
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
├── _config.yml              # Jekyll configuration
├── Gemfile                  # Ruby dependencies
├── index.md                 # Home page
├── installation.md          # Installation guide
├── project-structure.md     # Project layout
├── workflow.md              # DA workflow
├── guides/                  # User guides
│   ├── index.md            # Guides section index
│   ├── cli.md              # CLI reference (16 commands) ✅
│   ├── configuration.md    # YAML configuration guide ✅
│   ├── observations.md     # Satellite observation processing ✅
│   └── experiments.md      # End-to-end walkthrough ✅
├── reference/               # Technical reference
│   ├── index.md            # Reference section index ✅
│   ├── package-structure.md # (TODO)
│   ├── api.md              # (TODO)
│   └── da-methods.md       # (TODO)
└── advanced/                # Advanced topics
    ├── index.md            # Advanced section index ✅
    ├── troubleshooting.md  # Common issues and solutions ✅
    └── performance.md      # (TODO)
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

## Completed ✅

- ✅ Core documentation structure with Just the Docs theme
- ✅ Installation guide (Docker and native)
- ✅ Project structure documentation
- ✅ Workflow overview with mermaid diagrams
- ✅ Complete CLI reference (all 16 commands)
- ✅ Configuration reference (comprehensive YAML guide)
- ✅ Observation processing guide (MODIS, Sentinel-2, Sentinel-1)
- ✅ Running experiments guide (end-to-end walkthrough)
- ✅ Troubleshooting guide (common issues and solutions)
- ✅ Section index pages (Guides, Reference, Advanced)

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

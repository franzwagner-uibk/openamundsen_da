# openAMUNDSEN-DA documentation source

The site uses Jekyll and Just the Docs. Published source pages are organized to
match the upstream openAMUNDSEN documentation where behavior overlaps:

```text
Documentation
  Installation
  Input data
  Configuration
  Running the model
  Output data
  Example data sets
How to Use
Advanced
Reference
```

`docs/_site`, Jekyll caches and temporary preview trees are generated artifacts.
Do not edit them as source.

## Validate and build

From the repository root:

```bash
python scripts/ci/validate_docs.py
cd docs
bundle exec jekyll build --trace
```

The documentation validator checks navigation parents, Jekyll source links,
local assets, removed review/history pages, legacy installed commands and the
hash-selected tutorial image set. When parser help changes, update the curated
CLI guide and its contract tests in the same commit.

## Local preview

The WSL live-sync wrapper mirrors source into `/tmp` and keeps the preview current:

```bash
./scripts/docs/jekyll_serve_wsl_live_sync.sh
```

Open <http://127.0.0.1:4001/>. The direct Jekyll workflow and VS Code tasks remain
available for environments where filesystem notifications are reliable.

## Publication boundary

This documentation focuses on the implemented software workflow. Detailed
scientific formulation and interpretation are outside its scope. Tutorial
reference images under
`docs/assets/images/tutorial/rofental_2022_2023_es30` are selected by the frozen
publication manifest and must not be replaced manually.

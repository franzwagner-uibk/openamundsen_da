# openAMUNDSEN-DA documentation source

The site uses Jekyll and Just the Docs. Published source pages are organized to
match the upstream openAMUNDSEN documentation where behavior overlaps:

```text
Installation
Input Data
Configuration
Running
Output Data
Example Data
How to Use
Advanced
Reference
```

`docs/_site`, Jekyll caches and temporary preview trees are generated artifacts.
Do not edit them as source.

## Validate and build

From the repository root:

```bash
python scripts/docs/render_cli_reference.py --check
python scripts/ci/validate_docs.py
cd docs
bundle exec jekyll build --trace
```

After changing `openamundsen_da.cli.build_parser`, regenerate the committed CLI
reference with:

```bash
python scripts/docs/render_cli_reference.py
```

The documentation validator checks navigation parents, Jekyll source links,
local assets, removed review/history pages, legacy installed commands, generated
CLI drift and the hash-selected tutorial image set.

## Local preview

The WSL live-sync wrapper mirrors source into `/tmp` and keeps the preview current:

```bash
./scripts/docs/jekyll_serve_wsl_live_sync.sh
```

Open <http://127.0.0.1:4001/>. The direct Jekyll workflow and VS Code tasks remain
available for environments where filesystem notifications are reliable.

## Publication boundary

Technical documentation describes software behavior. Link scientific method and
Rofental interpretation to Wagner et al. (2026) instead of duplicating the
manuscript. Tutorial reference images under
`docs/assets/images/tutorial/rofental_2022_2023_es30` are selected by the frozen
publication manifest and must not be replaced manually.

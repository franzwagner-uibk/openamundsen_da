# Templates Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds template-specific rules.

- `templates/project/` is scaffold source, not a runtime example; keep it aligned with the documented setup/project structure.
- Preserve config ownership rules in templates: setup YAML stays pure openAMUNDSEN, project YAML owns DA config, step YAML owns local overrides.
- Template placeholders and readmes should guide users toward the current workflow without embedding stale behavior.
- Keep scaffold outputs compatible with the current observation naming, project layout, and fail-fast configuration philosophy.
- If template structure changes, update docs, examples, and any tests that rely on the scaffold contract.
- Prefer minimal, explanatory placeholder content over verbose template prose.

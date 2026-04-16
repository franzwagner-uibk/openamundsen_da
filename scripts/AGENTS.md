# Scripts Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds script-specific rules.

- Keep scripts narrow and task-focused; prefer wrappers around existing repo behavior over new alternate workflows.
- `scripts/ci/` defines the supported validation entrypoints; keep other script areas clearly secondary.
- Avoid baking machine-specific assumptions into shared scripts unless the repo already depends on them.
- Preserve executable shebangs, bash compatibility, and relative-path assumptions when editing shell scripts.
- Prefer small composable wrappers over one-off orchestration that bypasses the documented package or CI entrypoints.
- If a script becomes part of user-facing workflow, document it in `README.md` or `tests/README.md`.

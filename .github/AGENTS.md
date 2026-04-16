# .github Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds workflow-only rules.

- Treat `.github/workflows/ci.yml` as the CI contract for lint, unit tests, integration tests, and GHCR publish behavior.
- Keep workflow files and referenced scripts in `scripts/ci/` synchronized; do not change one without the other.
- Do not loosen CI gates, runner labels, or fork-safety guards casually.
- Preserve docs-only path-ignore behavior unless the task intentionally changes when CI should run.
- CI is a hard quality gate for behavior, interface, output, and workflow changes even if branch protection is light.
- Prefer additive, reviewable workflow edits over broad refactors.
- If workflow behavior changes, update the relevant documentation in `README.md` or `tests/README.md`.

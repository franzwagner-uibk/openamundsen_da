# CI Scripts Agent Notes

Inherit `scripts/AGENTS.md`. This file adds CI-wrapper rules.

- Treat these scripts as the implementation of the CI contract referenced by `.github/workflows/ci.yml`.
- Keep local and CI execution behavior aligned; avoid hidden branch-only or machine-only differences.
- Prefer stable logs, explicit failure modes, and deterministic validation checks over convenience output.
- When adding or changing required outputs, update the validator scripts, tests, and documentation in the same work.
- Respect the current single-domain and sub-domain baselines; do not quietly retarget CI away from shipped examples.
- Keep the wrappers readable and reviewable; avoid collapsing many responsibilities into one script.

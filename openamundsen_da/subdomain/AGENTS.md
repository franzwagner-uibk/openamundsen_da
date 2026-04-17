# Subdomain Agent Notes

Inherit the parent `openamundsen_da/AGENTS.md`. This file adds sub-domain rules.

- Treat sub-domain orchestration, manifests, merged outputs, and project-level reports as CI-validated behavior.
- Keep assumptions about region coverage, naming, and merge order explicit; do not hide them in ad hoc heuristics.
- Preserve compatibility with `scripts/ci/run_integration_tests_subdomain.sh` and `scripts/ci/validate_trimmed_subdomain.py` unless the task intentionally changes that contract.
- `subdomain.*` wraps the main project pipeline, so changes here must stay consistent with the single-domain workflow unless the task explicitly introduces a deliberate divergence.
- When outputs or report schemas change, update validators, tests, docs, and `examples/subdomains` in the same work.
- Prefer deterministic merge/report behavior over convenience fallbacks.

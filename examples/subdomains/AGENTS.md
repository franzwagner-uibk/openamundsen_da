# Subdomains Example Notes

Inherit `examples/AGENTS.md`. This file adds local rules for the canonical sub-domain baseline.

- `examples/subdomains` is the shipped sub-domain regression setup and must stay compatible with the CI sub-domain pipeline.
- Keep `env/subdomains.gpkg`, project layout, and expected merged outputs aligned with `scripts/ci/run_integration_tests_subdomain.sh`.
- If sub-domain reports, manifests, or output paths change, update `scripts/ci/validate_trimmed_subdomain.py`, tests, and docs in the same work.
- Preserve deterministic region handling and merged project-level outputs because CI validates them as a contract.
- Prefer explicit project-level changes over hidden per-region special cases.
- Avoid large data churn unless it is necessary for the behavior under test.

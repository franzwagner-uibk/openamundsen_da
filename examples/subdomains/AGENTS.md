# Subdomains Example Notes

Inherit `examples/AGENTS.md`. This file adds local rules for the canonical sub-domain baseline.

- `examples/subdomains` is the shipped North Tyrol sub-domain regression setup and must stay compatible with the CI sub-domain pipelines.
- Keep `env/subdomains.gpkg`, project layout, and expected merged outputs aligned with `scripts/ci/run_integration_tests_subdomain.sh` and `scripts/ci/run_integration_tests_model_subdomain.sh`.
- If sub-domain reports, manifests, event filtering, or output paths change, update validators, tests, and docs in the same work.
- Preserve deterministic region handling and merged project-level outputs because CI validates them as a contract.
- Prefer explicit project-level event candidates plus generic sub-domain filtering over hidden per-region special cases.
- Avoid large data churn unless it is necessary for the behavior under test.

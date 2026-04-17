# Tests Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds test-specific rules.

- `tests/README.md` is the authoritative test runbook; keep it aligned with actual scripts and CI behavior.
- Prefer targeted unit tests for logic changes and the existing CI wrapper scripts for integration coverage.
- Keep tests deterministic, narrow, and cheap unless the task explicitly requires broader regression coverage.
- If behavior, CLI, outputs, or shipped examples change, update or add regression tests in the same work.
- Prefer contract tests around config ownership, path discovery, obs preprocessing, benchmark outputs, and CLI behavior over brittle implementation-detail assertions.
- Do not couple tests to generated artifacts that are not part of the repo contract.
- Prefer validating user-visible contracts over internal implementation details.

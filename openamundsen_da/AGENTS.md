# Package Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds package-level rules.

- Respect the package boundaries documented in `docs/reference/package-structure.md`; keep orchestration, observation preprocessing, methods, and subdomain logic separated.
- Keep modules small, explicit, and cohesive; extend existing helpers before adding new abstraction layers.
- Preserve public CLI, config, path, and output contracts unless the task explicitly changes them.
- Prefer fail-fast validation for missing dates, products, paths, grids, coverage, and observation assumptions.
- Avoid hidden fallback behavior, implicit defaults, and cross-module side effects.
- Reuse current helpers in `core/`, `io/`, and `util/` instead of reimplementing path discovery, config merge, or validation logic.
- When changing scientific or workflow behavior, add or update tests in `tests/` and align docs/examples if the contract is user-visible.

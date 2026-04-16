# Methods Agent Notes

Inherit the parent `openamundsen_da/AGENTS.md`. This file adds method-specific rules.

- Treat assimilation formulas, H(x) logic, weighting, uncertainty handling, and visualization as scientific behavior, not cosmetic refactors.
- Current DA support scale is ROI-aggregated particle filtering without spatial localization; do not imply local filters or propagated spatial updates unless the task explicitly adds them.
- Keep observable names, config keys, and method selection explicit; do not introduce silent aliases or hidden mode switches.
- Treat SCF, wet snow, and station signals as product-aware evidence with representativeness limits; uncertainty choices should stay explicit and defensible.
- Plot and rendering code under `methods/viz/` is part of the output contract; update tests and docs when figure content, names, or paths change.
- Prefer small, well-scoped method changes with direct regression coverage.
- When a method change affects examples, keep `examples/rofental` and `examples/subdomains` aligned with the new contract.

# Pipeline Agent Notes

Inherit the parent `openamundsen_da/AGENTS.md`. This file adds pipeline-specific rules.

- Enforce setup/project/step ownership exactly; do not move DA config into setup YAML or step-local state into project YAML.
- `pipeline.project` is the main end-to-end orchestrator; keep prior forcing, launch, observation prep, assimilation, rejuvenation, plotting, and DA-output export connected through explicit contracts.
- Keep pipeline errors explicit and early when required inputs, dates, obs products, grids, or paths are missing.
- Preserve step sequencing and directory conventions unless the task explicitly changes the workflow contract.
- Prefer reusing existing skeleton and observation-preparation helpers over adding alternate pipeline paths.
- Current design is sequential and event-driven; do not imply coupled multivariate updates or localized DA unless the task explicitly implements them.
- If pipeline behavior changes, update tests, docs, and shipped examples together.

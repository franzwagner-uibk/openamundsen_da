---
paths:
  - "openamundsen_da/pipeline/**/*.py"
  - "openamundsen_da/subdomain/**/*.py"
---

# Pipeline And Subdomain Rules

- Enforce setup/project/step ownership exactly and keep step sequencing deterministic.
- Preserve manifest, report, and merge contracts because CI validates both single-domain and sub-domain flows.
- Do not imply localized or coupled multivariate DA unless the task explicitly implements it.
- Update tests, docs, and shipped examples when workflow or output contracts change.

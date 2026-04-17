---
paths:
  - "openamundsen_da/methods/**/*.py"
---

# Methods Rules

- Current DA support scale is ROI-aggregated particle filtering, not spatial localization.
- Keep observable names, uncertainty behavior, and method selection explicit; no silent aliases or hidden mode switches.
- Treat SCF, wet snow, and station signals as product-aware evidence with representativeness limits.
- Update tests, docs, and shipped examples when method outputs or figure contracts change.

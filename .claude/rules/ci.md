---
paths:
  - ".github/workflows/**/*"
  - "scripts/ci/**/*"
---

# CI Rules

- Treat `.github/workflows/ci.yml` plus `scripts/ci/` as one contract.
- Keep local and CI behavior aligned and preserve current shipped-example baselines.
- Prefer explicit failure modes and deterministic validation over convenience shortcuts.

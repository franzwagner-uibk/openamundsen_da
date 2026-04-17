# Examples Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds example-specific rules.

- Shipped examples are maintained baselines for docs and CI, not disposable demo data.
- Keep example paths, naming, and structure aligned with `examples/README.md`, validators, and tutorial/docs references.
- Prefer minimal, intentional example edits; avoid unnecessary large data churn.
- Preserve the scientific positioning of the shipped examples: they demonstrate the current ROI-scale PF workflow, not future localization or generalized EO claims.
- For experiments or paper-specific runs, copy to `dev_examples/` instead of mutating shipped example trees.
- If example behavior or outputs change, update the matching tests, CI validators, and docs in the same work.

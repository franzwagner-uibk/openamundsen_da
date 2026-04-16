# Core Agent Notes

Inherit the parent `openamundsen_da/AGENTS.md`. This file adds execution-spine rules.

- `core` owns config merge, environment setup, runner/launcher behavior, and CLI execution contracts; preserve those seams.
- Keep config and path resolution deterministic and explicit; avoid alternate discovery paths unless the task changes the contract intentionally.
- Prefer existing helpers in `core.config`, `core.env`, and `io.paths` over duplicate parsing or path logic.
- Manifest, log, and run-output behavior should stay predictable because CI and downstream tools rely on it.
- If command behavior, runtime defaults, or discovery rules change, update tests, docs, and CI wrappers together.

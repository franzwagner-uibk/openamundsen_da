---
published: false
---

# Checkpoint Pointer Producer Provenance

## Problem

Compact retention deletes a predecessor restart checkpoint only after every
successor checkpoint is readable and the deleted artifacts have successful
producer evidence. A particle-filter posterior stores lightweight
`state_pointer.json` files that reference states produced by prior members.
Cleanup currently looks for `member_run.json` beside the posterior pointer,
although the successful producer manifest is beside the referenced prior
state. This makes the first rolling checkpoint cleanup fail after otherwise
successful propagation.

## Design

- Resolve a checkpoint pointer to its referenced state before locating producer
  evidence. Validate the successful manifest belonging to that state's actual
  prior or open-loop member.
- Accept contained relative targets and remap stale absolute setup-mount paths
  through the existing project-relative layout. Never accept a target outside
  the project.
- Keep direct member artifacts on the existing producer-manifest path. Reject
  missing, malformed, unreadable, unsuccessful or identity-mismatched producer
  evidence before the retention ledger is written or any file is deleted.
- Do not change cleanup eligibility, the retention ledger schema, public
  configuration or scientific outputs.

## Validation

Use the real PF directory shape in tests: prior states with successful member
manifests, posterior source and state pointers, duplicated resampling ancestry
and a complete successor checkpoint. Require successful cleanup and a valid
retention ledger. Add fail-closed cases for malformed, missing, mismatched and
external targets, then run the full unit suite and a pinned-image two-step
integration before production use.

# Retention Batch Consumer Guard

## Problem

Compact cleanup records byte-identical retained consumers before deleting raw
artifacts. `apply_retention_batch` currently recomputes the full SHA-256 digest
of every retained consumer before every source unlink. The North Tyrol pilot
therefore rereads a roughly 0.9--1.0 GB compact forcing NetCDF before each of
roughly 226,000--298,000 forcing CSV deletions per leaf. The scientific outputs
are valid, but cleanup becomes a multi-day CPU and storage-I/O workload.

## Scope

Optimize only dependency guarding inside one already planned retention batch.
Do not change cleanup eligibility, ledger schema, source-file validation,
retention generations, scientific outputs, public interfaces or full-retention
behavior.

## Considered approaches

1. **Pinned consumer guard (selected).** Hash every consumer once when a batch
   starts or resumes, keep its file descriptor open and verify stable
   device/inode/type/size/mtime/ctime identity before every unlink. Hash it
   again before completing the batch. This keeps restart validation and detects
   replacement or ordinary in-place mutation with constant-time hot-path work.
2. **Periodic full hashes.** Hash every fixed number of deletions. This retains
   large avoidable reads and creates an arbitrary detection window.
3. **Atomic directory cleanup.** Rename and remove complete artifact trees.
   This would require changing layouts, cleanup grouping and crash semantics;
   it is too broad for this defect.

## Design

An internal context-managed guard validates the recorded consumer inventory
with the existing full content hash, opens each consumer read-only without
following a final symlink and captures `fstat` identity. The guard requires the
opened inode and the current path to remain the same regular file with unchanged
size, nanosecond modification time and nanosecond change time. Any mismatch,
missing path, symlink, filesystem error or descriptor error raises
`CleanupSafetyError` before the next source deletion.

The batch deletion loop performs the constant-time guard check before every
unlink. Source candidates retain their existing per-file size and SHA-256
validation. After the last unlink, the guard performs one final full inventory
hash before the batch is marked complete. Descriptors always close through the
context manager, including exceptions.

The ledger remains authoritative across crashes. A resumed planned batch first
performs the full inventory hash again, then safely continues with only the
surviving recorded sources. No runtime-only guard identity is persisted.

## Failure behavior

- A consumer changed before entry: fail before deleting another source.
- A consumer is replaced, removed, resized or modified during the batch: fail
  before the next unlink.
- A final full hash mismatch: leave the batch planned and fail closed.
- A source changed after planning: preserve the existing fail-closed behavior.
- A crash: descriptors disappear and the next run revalidates full hashes from
  the durable ledger before resuming.

## Tests and acceptance

- Prove full consumer hashing occurs exactly twice per successful multi-source
  batch, independent of source count.
- Prove replacement, unlink, size-preserving write, truncation, timestamp or
  type changes stop cleanup before the next unlink.
- Prove final-hash failure leaves the batch planned.
- Prove descriptors close after success and exceptions.
- Prove interrupted planned batches resume idempotently and completed ledgers
  retain existing validation behavior.
- Run the focused retention/cleanup/API tests, the complete unit suite and CI.
- In an exact-image integration, resume a synthetic large planned forcing batch
  and demonstrate deletion work scales with source files rather than consumer
  bytes multiplied by source count.

For the active North Tyrol pilot, preserve evidence, stop only after the reviewed
image is available and resume the planned cleanup without `--overwrite` so the
accepted propagation and compact outputs are not recomputed.

"""Batch subregion toolkit for running openAMUNDSEN open-loop jobs in parallel.

This package provides:
- Preparation of per-subregion setups (clipped grids, meteo/obs subsets, ROI masks)
- Parallel execution of open-loop runs
- Merging of gridded and point outputs back to a global mosaic
- Lightweight plotting of station results against observations

Entrypoint CLI: ``oa-da-batch`` (see openamundsen_da.batch.cli).
"""

__all__ = [
    "cli",
    "prepare",
    "run",
    "merge",
    "plot",
    "manifest",
]

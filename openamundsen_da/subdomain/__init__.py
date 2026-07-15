"""Internal staged subdomain workflows.

This package provides:
- Preparation of per-sub-domain setups under ``<project>/subdomains``
- Parallel execution of independent DA project runs per sub-domain
- Hard-mosaic merge of compact DA grids back to the global grid
- Plain openAMUNDSEN model sub-domain prepare/run/merge helpers
- Project-level CSV reports summarizing sub-domain run/assimilation statistics
"""

__all__ = [
    "prepare",
    "run",
    "model",
    "merge",
    "render",
    "manifest",
    "status",
]

"""Sub-domain toolkit for setup/project workflows.

This package provides:
- Preparation of per-sub-domain setups under ``<project>/subdomains``
- Parallel execution of independent DA project runs per sub-domain
- Hard-mosaic merge of compact DA grids back to the global grid
- Project-level CSV reports summarizing sub-domain run/assimilation statistics

Entrypoint CLI: ``oa-da-subdomain`` (see ``openamundsen_da.subdomain.cli``).
"""

__all__ = [
    "cli",
    "prepare",
    "run",
    "merge",
    "plot",
    "manifest",
]


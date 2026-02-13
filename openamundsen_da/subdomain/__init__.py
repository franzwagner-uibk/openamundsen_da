"""Sub-domain toolkit for setup/project workflows.

This package provides:
- Preparation of per-sub-domain setups under ``<setup>/subdomains``
- Parallel execution of independent DA project runs per sub-domain
- Hard-mosaic merge of compact outputs back to the global grid
- Plotting of merged station results against observations

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


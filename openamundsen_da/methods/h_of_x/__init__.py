"""
H(x): Observation operators mapping model state to observation space.

Currently includes model-derived Snow Cover Fraction (SCF) from
openAMUNDSEN outputs using depth- or SWE-based operators.
"""

__all__ = [
    "compute_model_scf",
]


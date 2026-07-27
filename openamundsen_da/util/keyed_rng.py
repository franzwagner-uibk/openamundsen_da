"""Order-independent random generators for scientific workflow stages."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.random import Generator


RNG_SCHEME = "keyed-v1"


def keyed_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 128-bit seed from a configured seed and semantic key."""
    if isinstance(base_seed, bool) or int(base_seed) < 0:
        raise ValueError(f"base_seed must be a non-negative integer, got {base_seed!r}")
    tokens = [RNG_SCHEME, str(int(base_seed)), *(str(part) for part in parts)]
    digest = hashlib.sha256(" | ".join(tokens).encode("utf-8")).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def keyed_rng(base_seed: int, *parts: object) -> Generator:
    """Return a deterministic generator for one semantic workflow key."""
    return np.random.default_rng(keyed_seed(base_seed, *parts))


__all__ = ["RNG_SCHEME", "keyed_rng", "keyed_seed"]

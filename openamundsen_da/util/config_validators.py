"""Shared fail-fast config validators used across DA modules."""

from __future__ import annotations


def require_mapping(raw: object, *, path: str) -> dict[str, object]:
    """Return mapping if valid, else raise explicit ValueError."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


def require_nonempty_str(mapping: dict[str, object], key: str, *, path: str) -> str:
    """Return non-empty string value for key, else raise explicit ValueError."""
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {path}.{key}")
    raw = mapping.get(key)
    val = str(raw).strip() if raw is not None else ""
    if not val:
        raise ValueError(f"Configuration value must not be empty: {path}.{key}")
    return val

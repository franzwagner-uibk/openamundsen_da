from __future__ import annotations

import pytest

from openamundsen_da.util.keyed_rng import keyed_rng, keyed_seed


def test_keyed_rng_is_reproducible_and_order_independent() -> None:
    forward = {
        member: keyed_rng(113, "rejuvenation", "2023-04-26", member, "temperature").normal()
        for member in ["member_001", "member_002", "member_003"]
    }
    reverse = {
        member: keyed_rng(113, "rejuvenation", "2023-04-26", member, "temperature").normal()
        for member in reversed(["member_001", "member_002", "member_003"])
    }
    assert forward == reverse


def test_keyed_rng_separates_events_stages_and_variables() -> None:
    keys = {
        keyed_seed(113, "initial", "2023-04-26", "member_001", "temperature"),
        keyed_seed(113, "rejuvenation", "2023-04-26", "member_001", "temperature"),
        keyed_seed(113, "rejuvenation", "2023-05-26", "member_001", "temperature"),
        keyed_seed(113, "rejuvenation", "2023-04-26", "member_001", "precipitation"),
    }
    assert len(keys) == 4


@pytest.mark.parametrize("seed", [-1, True])
def test_keyed_seed_rejects_invalid_base_seed(seed) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        keyed_seed(seed, "stage")

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from openamundsen_da.util.figure_lifecycle import close_created_figures


@pytest.mark.parametrize("fail", [False, True])
def test_close_created_figures_preserves_existing_and_closes_new(fail: bool) -> None:
    existing = plt.figure()

    @close_created_figures
    def render() -> None:
        plt.figure()
        plt.subplots()
        if fail:
            raise RuntimeError("render failed")

    try:
        if fail:
            with pytest.raises(RuntimeError, match="render failed"):
                render()
        else:
            render()
        assert plt.get_fignums() == [existing.number]
    finally:
        plt.close(existing)

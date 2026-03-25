from __future__ import annotations

import pandas as pd

import openamundsen_da.methods.pf.plot_ess_timeline as ess_mod
from openamundsen_da.methods.viz.plot_result_overview import plot_result_overview


def _frame(col: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-02"]), col: values})


def test_plot_result_overview_accepts_backend_argument(tmp_path: Path) -> None:
    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
        scf_env=None,
        wet_env=None,
        output=out_path,
        backend="Agg",
    )
    assert out_path.is_file()


def test_plot_ess_timeline_renders_subtitle() -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "ess": [5.0, 6.0],
            "ess_norm": [0.5, 0.6],
            "n": [10, 10],
        }
    )
    fig = ess_mod._plot(
        df,
        normalized=False,
        threshold=None,
        title="ESS title",
        subtitle="Subtitle text",
        ensemble_size=10,
        backend="Agg",
    )
    texts = [text.get_text() for text in fig.texts]
    assert "Subtitle text" in texts
    plt.close(fig)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openamundsen_da.methods.viz.plots.assimilation.station_diagnostics import plot_station_diagnostics_for_csv
from openamundsen_da.methods.viz.plots.forcing_ensemble import _plot_station
from openamundsen_da.methods.viz.plots.observer.scf_summary import cli_main as plot_scf_cli_main


def test_plot_station_diagnostics_for_csv_writes_png(tmp_path: Path) -> None:
    csv_path = tmp_path / "station_diagnostics_station_hs_20230221.csv"
    pd.DataFrame(
        [
            {
                "station_id": "latschbloder",
                "member_id": "member_001",
                "obs_value": 1.1,
                "model_value": 0.9,
                "sigma": 0.3,
                "final_weight": 0.7,
                "variable": "station_hs",
                "date": "2023-02-21",
            },
            {
                "station_id": "latschbloder",
                "member_id": "member_002",
                "obs_value": 1.1,
                "model_value": 1.2,
                "sigma": 0.3,
                "final_weight": 0.3,
                "variable": "station_hs",
                "date": "2023-02-21",
            },
        ]
    ).to_csv(csv_path, index=False)

    out = plot_station_diagnostics_for_csv(csv_path, backend="Agg")

    assert out == csv_path.with_suffix(".png")
    assert out.is_file()


def test_plot_scf_summary_cli_main_writes_png(tmp_path: Path) -> None:
    csv_path = tmp_path / "scf_summary.csv"
    out_path = tmp_path / "scf_summary.png"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"]),
            "scf": [0.2, 0.5, 0.8],
        }
    ).to_csv(csv_path, index=False)

    rc = plot_scf_cli_main(
        [
            str(csv_path),
            "--output",
            str(out_path),
            "--backend",
            "Agg",
        ]
    )

    assert rc == 0
    assert out_path.is_file()


def test_plot_forcing_station_writes_png(tmp_path: Path) -> None:
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    ol_df = pd.DataFrame({"temp": [273.4, 274.0, 273.8], "precip": [0.0, 1.5, 0.2]}, index=idx)
    mem_a = pd.DataFrame({"temp": [272.8, 273.6, 273.0], "precip": [0.0, 1.0, 0.1]}, index=idx)
    mem_b = pd.DataFrame({"temp": [273.1, 273.9, 273.5], "precip": [0.2, 0.7, 0.0]}, index=idx)
    out_path = tmp_path / "forcing_station.png"

    _plot_station(
        ol_df=ol_df,
        mem_dfs=[mem_a, mem_b],
        temp_col="temp",
        precip_col="precip",
        hydro_m=10,
        hydro_d=1,
        title="Forcing Ensemble",
        subtitle="Latschbloder",
        backend="Agg",
        out_path=out_path,
    )

    assert out_path.is_file()

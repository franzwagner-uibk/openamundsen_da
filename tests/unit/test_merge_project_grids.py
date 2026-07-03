from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openamundsen_da.pipeline.merge_project_grids import cli_main, merge_project_da_output_grids
from openamundsen_da.util.storage_policy import da_summary_netcdf_encoding


def _write_project_da_grid(
    project_dir: Path,
    *,
    time1: list[str],
    time2: list[str],
    x: list[float] | None = None,
    include_swe: bool = True,
) -> None:
    x_vals = np.asarray(x if x is not None else [10.0, 20.0], dtype=np.float64)
    y_vals = np.asarray([100.0, 90.0], dtype=np.float64)
    time1_vals = np.asarray(time1, dtype="datetime64[ns]")
    time2_vals = np.asarray(time2, dtype="datetime64[ns]")
    coords = {
        "time1": time1_vals,
        "time2": time2_vals,
        "y": y_vals,
        "x": x_vals,
        "snow_layer": np.asarray([0, 1], dtype=np.int16),
        "nbnd": np.asarray([0, 1], dtype=np.int16),
        "crs": xr.DataArray(
            np.asarray(0, dtype=np.int16),
            attrs={
                "grid_mapping_name": "transverse_mercator",
                "crs_wkt": "EPSG:25832",
            },
        ),
        "time2_bounds": (
            ("time2", "nbnd"),
            np.stack([time2_vals, time2_vals + np.timedelta64(1, "D")], axis=1),
        ),
    }
    data_vars = {
        "ens_mean_snowdepth_daily": xr.DataArray(
            np.full((len(time1_vals), len(y_vals), len(x_vals)), 1.234, dtype=np.float32),
            dims=("time1", "y", "x"),
            attrs={
                "summary_metric": "ens_mean",
                "description": "Posterior ensemble mean",
            },
        ),
        "ens_mean_liquid_water_content": xr.DataArray(
            np.full((len(time2_vals), 2, len(y_vals), len(x_vals)), 2.0, dtype=np.float32),
            dims=("time2", "snow_layer", "y", "x"),
            attrs={
                "summary_metric": "ens_mean",
                "description": "Posterior ensemble mean",
            },
        ),
    }
    if include_swe:
        data_vars["ens_mean_swe_daily"] = xr.DataArray(
            np.full((len(time2_vals), len(y_vals), len(x_vals)), 3.0, dtype=np.float32),
            dims=("time2", "y", "x"),
            attrs={
                "summary_metric": "ens_mean",
                "description": "Posterior ensemble mean",
            },
        )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "da_output_version": "2",
            "summary_variables": "ens_mean",
            "source_step_count": "1",
            "source_steps": "step_00_init",
        },
    )
    out = project_dir / "results" / "grids" / "da_output_grids.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding=da_summary_netcdf_encoding(ds))


def test_merge_project_da_output_grids_concatenates_time_axes_and_preserves_encoding(tmp_path: Path) -> None:
    project_1 = tmp_path / "projects" / "project_2020_2021"
    project_2 = tmp_path / "projects" / "project_2021_2022"
    _write_project_da_grid(
        project_1,
        time1=["2020-10-01", "2020-10-02"],
        time2=["2020-10-01"],
    )
    _write_project_da_grid(
        project_2,
        time1=["2021-10-01"],
        time2=["2021-10-01"],
    )
    out = tmp_path / "merged" / "da_output_grids.nc"

    written = merge_project_da_output_grids([project_1, project_2], out)

    assert written == out
    with xr.open_dataset(out) as ds:
        assert ds.sizes["time1"] == 3
        assert ds.sizes["time2"] == 2
        assert ds.sizes["y"] == 2
        assert ds.sizes["x"] == 2
        assert ds.sizes["snow_layer"] == 2
        assert ds["time1"].dt.strftime("%Y-%m-%d").values.tolist() == [
            "2020-10-01",
            "2020-10-02",
            "2021-10-01",
        ]
        assert ds["time2"].dt.strftime("%Y-%m-%d").values.tolist() == ["2020-10-01", "2021-10-01"]
        assert "time2_bounds" in ds.coords
        assert ds.attrs["project_merge"] == "true"
        assert ds.attrs["project_merge_source_projects"] == "project_2020_2021,project_2021_2022"
        assert "source_step_count" not in ds.attrs
        assert "source_steps" not in ds.attrs
        np.testing.assert_allclose(ds["ens_mean_snowdepth_daily"].values[0, 0, 0], 1.234, atol=0.001)
    with xr.open_dataset(out, decode_cf=False) as raw:
        var = raw["ens_mean_snowdepth_daily"]
        assert var.dtype == np.dtype("int16")
        assert var.attrs["scale_factor"] == np.float32(0.001)
        assert var.attrs["_FillValue"] == np.int16(-32768)


def test_merge_project_da_output_grids_rejects_duplicate_times(tmp_path: Path) -> None:
    project_1 = tmp_path / "projects" / "project_2020_2021"
    project_2 = tmp_path / "projects" / "project_2021_2022"
    _write_project_da_grid(project_1, time1=["2020-10-01"], time2=["2020-10-01"])
    _write_project_da_grid(project_2, time1=["2020-10-01"], time2=["2021-10-01"])

    with pytest.raises(ValueError, match="Duplicate 'time1' timestamps"):
        merge_project_da_output_grids([project_1, project_2], tmp_path / "out.nc")


def test_merge_project_da_output_grids_rejects_variable_mismatch(tmp_path: Path) -> None:
    project_1 = tmp_path / "projects" / "project_2020_2021"
    project_2 = tmp_path / "projects" / "project_2021_2022"
    _write_project_da_grid(project_1, time1=["2020-10-01"], time2=["2020-10-01"])
    _write_project_da_grid(project_2, time1=["2021-10-01"], time2=["2021-10-01"], include_swe=False)

    with pytest.raises(ValueError, match="Data variables"):
        merge_project_da_output_grids([project_1, project_2], tmp_path / "out.nc")


def test_merge_project_da_output_grids_rejects_grid_mismatch(tmp_path: Path) -> None:
    project_1 = tmp_path / "projects" / "project_2020_2021"
    project_2 = tmp_path / "projects" / "project_2021_2022"
    _write_project_da_grid(project_1, time1=["2020-10-01"], time2=["2020-10-01"], x=[10.0, 20.0])
    _write_project_da_grid(project_2, time1=["2021-10-01"], time2=["2021-10-01"], x=[10.0, 30.0])

    with pytest.raises(ValueError, match="Static coordinate 'x'"):
        merge_project_da_output_grids([project_1, project_2], tmp_path / "out.nc")


def test_cli_resolves_setup_project_names(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    project_1 = setup / "projects" / "project_2020_2021"
    project_2 = setup / "projects" / "project_2021_2022"
    _write_project_da_grid(project_1, time1=["2020-10-01"], time2=["2020-10-01"])
    _write_project_da_grid(project_2, time1=["2021-10-01"], time2=["2021-10-01"])
    out = tmp_path / "merged.nc"

    rc = cli_main(
        [
            "--setup",
            str(setup),
            "--project",
            "project_2020_2021",
            "--project",
            "project_2021_2022",
            "--output-nc",
            str(out),
        ]
    )

    assert rc == 0
    assert out.is_file()


def test_cli_accepts_direct_project_dirs(tmp_path: Path) -> None:
    project_1 = tmp_path / "project_2020_2021"
    project_2 = tmp_path / "project_2021_2022"
    _write_project_da_grid(project_1, time1=["2020-10-01"], time2=["2020-10-01"])
    _write_project_da_grid(project_2, time1=["2021-10-01"], time2=["2021-10-01"])
    out = tmp_path / "merged.nc"

    rc = cli_main(
        [
            "--project-dir",
            str(project_1),
            "--project-dir",
            str(project_2),
            "--output-nc",
            str(out),
        ]
    )

    assert rc == 0
    assert out.is_file()

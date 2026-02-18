from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from openamundsen_da.util.da_output import (
    output_retention_mode,
    write_da_output_grids,
    write_project_da_output_grids,
)


def _write_nc(path: Path, values: np.ndarray) -> None:
    ds = xr.Dataset(
        data_vars={
            "snowdepth_daily": xr.DataArray(
                values.astype(np.float32),
                dims=("time1", "y", "x"),
                coords={"time1": [0], "y": [0, 1], "x": [0, 1]},
            )
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_step_member_ncs(step_dir: Path, open_loop_vals: np.ndarray, member_vals: list[np.ndarray], day: str) -> None:
    prior_root = step_dir / "ensembles" / "prior"
    time_val = np.array([np.datetime64(day)], dtype="datetime64[ns]")

    def _write(path: Path, values: np.ndarray) -> None:
        ds = xr.Dataset(
            data_vars={
                "snowdepth_daily": xr.DataArray(
                    values.astype(np.float32),
                    dims=("time1", "y", "x"),
                    coords={"time1": time_val, "y": [0, 1], "x": [0, 1]},
                )
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)

    _write(prior_root / "open_loop" / "results" / "output_grids.nc", open_loop_vals)
    for idx, vals in enumerate(member_vals, start=1):
        _write(prior_root / f"member_{idx:03d}" / "results" / "output_grids.nc", vals)


def test_write_da_output_grids_creates_expected_variables(tmp_path: Path) -> None:
    open_loop = tmp_path / "output_grids.nc"
    member_1 = tmp_path / "member_001_output_grids.nc"
    member_2 = tmp_path / "member_002_output_grids.nc"
    out_nc = tmp_path / "da_output_grids.nc"

    _write_nc(open_loop, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    _write_nc(member_1, np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32))
    _write_nc(member_2, np.array([[[4.0, 6.0], [8.0, 10.0]]], dtype=np.float32))

    written = write_da_output_grids(
        open_loop_nc=open_loop,
        member_ncs=[member_1, member_2],
        output_nc=out_nc,
    )

    assert written == out_nc
    assert out_nc.is_file()
    with xr.open_dataset(out_nc) as ds:
        assert "open_loop_snowdepth_daily" in ds.data_vars
        assert "ens_mean_snowdepth_daily" in ds.data_vars
        assert "ens_std_snowdepth_daily" in ds.data_vars
        assert "increment_snowdepth_daily" in ds.data_vars
        mean_vals = ds["ens_mean_snowdepth_daily"].values
        inc_vals = ds["increment_snowdepth_daily"].values
        assert np.allclose(mean_vals, np.array([[[3.0, 5.0], [7.0, 9.0]]], dtype=np.float32))
        assert np.allclose(inc_vals, np.array([[[2.0, 3.0], [4.0, 5.0]]], dtype=np.float32))
        assert ds["increment_snowdepth_daily"].attrs.get("summary_metric") == "increment"
        assert ds.attrs.get("increment_definition") == "increment_<var> = ens_mean_<var> - open_loop_<var>"


def test_write_project_da_output_grids_spans_all_steps(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init"
    step_01 = project_dir / "steps" / "step_01_followup"
    out_nc = project_dir / "results" / "grids" / "da_output_grids.nc"

    _write_step_member_ncs(
        step_00,
        np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32),
        [
            np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32),
            np.array([[[4.0, 4.0], [4.0, 4.0]]], dtype=np.float32),
        ],
        "2023-01-01",
    )
    _write_step_member_ncs(
        step_01,
        np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        [
            np.array([[[20.0, 20.0], [20.0, 20.0]]], dtype=np.float32),
            np.array([[[30.0, 30.0], [30.0, 30.0]]], dtype=np.float32),
        ],
        "2023-01-02",
    )

    written = write_project_da_output_grids(
        step_dirs=[step_00, step_01],
        output_nc=out_nc,
    )

    assert written == out_nc
    assert out_nc.is_file()
    with xr.open_dataset(out_nc) as ds:
        assert "ens_mean_snowdepth_daily" in ds.data_vars
        assert ds.sizes["time1"] == 2
        mean_vals = ds["ens_mean_snowdepth_daily"].values
        assert np.isclose(mean_vals[0, 0, 0], 3.0)
        assert np.isclose(mean_vals[1, 0, 0], 25.0)
        assert ds.attrs.get("source_step_count") == "2"


def test_output_retention_mode_defaults_to_compact(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
        encoding="utf-8",
    )

    assert output_retention_mode(project_dir) == "compact"


def test_output_retention_mode_reads_full(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "start_date: '2022-10-01'",
                "end_date: '2022-10-02'",
                "data_assimilation:",
                "  output:",
                "    retention: full",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert output_retention_mode(project_dir) == "full"

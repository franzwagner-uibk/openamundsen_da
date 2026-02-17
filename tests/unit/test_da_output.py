from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from openamundsen_da.util.da_output import output_retention_mode, write_da_output_grids


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
        assert "da_mean_snowdepth_daily" in ds.data_vars
        assert "da_std_snowdepth_daily" in ds.data_vars
        assert "da_increment_snowdepth_daily" in ds.data_vars
        mean_vals = ds["da_mean_snowdepth_daily"].values
        inc_vals = ds["da_increment_snowdepth_daily"].values
        assert np.allclose(mean_vals, np.array([[[3.0, 5.0], [7.0, 9.0]]], dtype=np.float32))
        assert np.allclose(inc_vals, np.array([[[2.0, 3.0], [4.0, 5.0]]], dtype=np.float32))


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

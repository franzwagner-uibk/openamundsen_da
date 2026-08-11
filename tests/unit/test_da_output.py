from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openamundsen_da.util import da_output as da_output_mod
from openamundsen_da.util.da_output import (
    output_retention_mode,
    validate_project_da_output_grids,
    validate_compact_output_file,
    write_da_output_grids,
    write_project_da_output_grids,
)
from openamundsen_da.methods.pf.weights import initialize_prior_weights, write_prior_weights


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
    initialize_prior_weights(
        step_dir,
        [f"member_{idx:03d}" for idx in range(1, len(member_vals) + 1)],
    )


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


def test_write_da_output_grids_stores_compact_int16_payloads(tmp_path: Path) -> None:
    open_loop = tmp_path / "output_grids.nc"
    member_1 = tmp_path / "member_001_output_grids.nc"
    member_2 = tmp_path / "member_002_output_grids.nc"
    out_nc = tmp_path / "da_output_grids.nc"

    _write_nc(open_loop, np.array([[[1.2344, 2.3456], [3.4567, 4.5678]]], dtype=np.float32))
    _write_nc(member_1, np.array([[[2.2344, 4.3456], [6.4567, 8.5678]]], dtype=np.float32))
    _write_nc(member_2, np.array([[[4.2344, 6.3456], [8.4567, 10.5678]]], dtype=np.float32))

    write_da_output_grids(
        open_loop_nc=open_loop,
        member_ncs=[member_1, member_2],
        output_nc=out_nc,
    )

    with xr.open_dataset(out_nc) as ds:
        np.testing.assert_allclose(ds["ens_mean_snowdepth_daily"].values[0, 0, 0], 3.234, atol=0.001)
    with xr.open_dataset(out_nc, decode_cf=False) as raw:
        var = raw["ens_mean_snowdepth_daily"]
        assert var.dtype == np.dtype("int16")
        assert var.attrs["_FillValue"] == np.int16(-32768)
        assert var.attrs["scale_factor"] == np.float32(0.001)
        assert var.attrs["add_offset"] == np.float32(0.0)
        assert var.encoding.get("zlib") is True
        assert var.encoding.get("shuffle") is True
        assert var.encoding.get("complevel") == 4


def test_write_da_output_grids_rejects_scaled_int16_overflow(tmp_path: Path) -> None:
    open_loop = tmp_path / "output_grids.nc"
    member = tmp_path / "member_001_output_grids.nc"
    out_nc = tmp_path / "da_output_grids.nc"

    _write_nc(open_loop, np.array([[[40.0, 1.0], [1.0, 1.0]]], dtype=np.float32))
    _write_nc(member, np.array([[[40.0, 1.0], [1.0, 1.0]]], dtype=np.float32))

    with pytest.raises(ValueError, match="exceeds compact int16 NetCDF storage range"):
        write_da_output_grids(
            open_loop_nc=open_loop,
            member_ncs=[member],
            output_nc=out_nc,
        )


def test_atomic_grid_write_preserves_accepted_output_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    open_loop = tmp_path / "output_grids.nc"
    member = tmp_path / "member_001_output_grids.nc"
    output = tmp_path / "da_output_grids.nc"
    _write_nc(open_loop, np.ones((1, 2, 2)))
    _write_nc(member, np.ones((1, 2, 2)))
    output.write_bytes(b"accepted")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        write_da_output_grids(
            open_loop_nc=open_loop,
            member_ncs=[member],
            output_nc=output,
        )

    assert output.read_bytes() == b"accepted"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp.nc"))


def test_atomic_grid_write_preserves_accepted_output_on_validation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    open_loop = tmp_path / "output_grids.nc"
    member = tmp_path / "member_001_output_grids.nc"
    output = tmp_path / "da_output_grids.nc"
    _write_nc(open_loop, np.ones((1, 2, 2)))
    _write_nc(member, np.ones((1, 2, 2)))
    output.write_bytes(b"accepted")
    monkeypatch.setattr(
        da_output_mod,
        "_validate_da_dataset_against_expected",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid temp")),
    )

    with pytest.raises(ValueError, match="invalid temp"):
        write_da_output_grids(open_loop_nc=open_loop, member_ncs=[member], output_nc=output)

    assert output.read_bytes() == b"accepted"


def test_grid_cleanup_completeness_requires_every_configured_metric_and_member(
    tmp_path: Path,
) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    step = project / "steps" / "step_00"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing: {ensemble_size: 1}\n"
        "  output:\n"
        "    grids:\n"
        "      variables:\n"
        "        - {var: snowdepth_daily, metrics: [open_loop, ens_mean]}\n",
        encoding="utf-8",
    )
    step.mkdir(parents=True)
    (step / "step.yml").write_text(
        "start_date: '2023-01-01'\nend_date: '2023-01-01'\n",
        encoding="utf-8",
    )
    _write_step_member_ncs(
        step,
        np.ones((1, 2, 2)),
        [np.full((1, 2, 2), 2.0)],
        "2023-01-01",
    )
    output = project / "results" / "grids" / "da_output_grids.nc"
    write_project_da_output_grids(step_dirs=[step], output_nc=output)
    assert validate_project_da_output_grids(project, output_nc=output) == output

    with xr.open_dataset(output) as dataset:
        incomplete = dataset.drop_vars("ens_mean_snowdepth_daily").load()
    incomplete.to_netcdf(output, mode="w")
    with pytest.raises(ValueError, match="missing configured metric-variable"):
        validate_project_da_output_grids(project, output_nc=output)
    write_project_da_output_grids(step_dirs=[step], output_nc=output)

    member_grid = step / "ensembles" / "prior" / "member_001" / "results" / "output_grids.nc"
    member_grid.unlink()
    with pytest.raises(FileNotFoundError, match="required for completeness"):
        validate_project_da_output_grids(project, output_nc=output)


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


@pytest.mark.parametrize("corruption", ["value", "coordinate", "time_order", "time_duplicate"])
def test_grid_cleanup_validation_rejects_scientific_or_coordinate_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing: {ensemble_size: 1}\n"
        "  output:\n"
        "    grids:\n"
        "      variables:\n"
        "        - {var: snowdepth_daily, metrics: [open_loop, ens_mean]}\n",
        encoding="utf-8",
    )
    steps = []
    for index, day in enumerate(("2023-01-01", "2023-01-02")):
        step = project / "steps" / f"step_{index:02d}"
        step.mkdir(parents=True)
        (step / "step.yml").write_text(
            f"start_date: '{day}'\nend_date: '{day}'\n", encoding="utf-8"
        )
        _write_step_member_ncs(
            step,
            np.full((1, 2, 2), float(index + 1)),
            [np.full((1, 2, 2), float(index + 2))],
            day,
        )
        steps.append(step)
    output = project / "results" / "grids" / "da_output_grids.nc"
    write_project_da_output_grids(step_dirs=steps, output_nc=output)
    with xr.open_dataset(output) as dataset:
        corrupted = dataset.load()
    if corruption == "value":
        corrupted["ens_mean_snowdepth_daily"].values[0, 0, 0] += 1.0
    elif corruption == "coordinate":
        corrupted = corrupted.assign_coords(x=np.asarray([0, 2]))
    elif corruption == "time_order":
        corrupted = corrupted.isel(time1=[1, 0])
    else:
        values = corrupted.time1.values.copy()
        values[1] = values[0]
        corrupted = corrupted.assign_coords(time1=values)
    corrupted.to_netcdf(output, mode="w")

    with pytest.raises(ValueError, match="validation"):
        validate_project_da_output_grids(project, output_nc=output)


def test_write_project_da_output_grids_adds_weighted_analysis_increment(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    step_00 = project_dir / "steps" / "step_00_init"
    step_01 = project_dir / "steps" / "step_01_followup"
    out_nc = project_dir / "results" / "grids" / "da_output_grids.nc"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2023.yml").write_text(
        "\n".join(
            [
                "start_date: '2023-01-01'",
                "end_date: '2023-01-02'",
                "data_assimilation:",
                "  assimilation_events:",
                "    - date: '2023-01-01'",
                "      variable: station_hs",
                "      product: STATION",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    step_00.mkdir(parents=True, exist_ok=True)
    step_01.mkdir(parents=True, exist_ok=True)
    (step_00 / "step_00.yml").write_text("start_date: '2023-01-01'\nend_date: '2023-01-01'\n", encoding="utf-8")
    (step_01 / "step_01.yml").write_text("start_date: '2023-01-02'\nend_date: '2023-01-02'\n", encoding="utf-8")
    _write_step_member_ncs(
        step_00,
        np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32),
        [
            np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32),
            np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        ],
        "2023-01-01",
    )
    _write_step_member_ncs(
        step_01,
        np.array([[[3.0, 3.0], [3.0, 3.0]]], dtype=np.float32),
        [
            np.array([[[4.0, 4.0], [4.0, 4.0]]], dtype=np.float32),
            np.array([[[6.0, 6.0], [6.0, 6.0]]], dtype=np.float32),
        ],
        "2023-01-02",
    )
    assim_dir = step_00 / "assim"
    assim_dir.mkdir(parents=True, exist_ok=True)
    (assim_dir / "weights_station_hs_20230101.csv").write_text(
        "member_id,weight\nmember_001,0.25\nmember_002,0.75\n",
        encoding="utf-8",
    )

    written = write_project_da_output_grids(
        step_dirs=[step_00, step_01],
        output_nc=out_nc,
    )

    assert written == out_nc
    with xr.open_dataset(out_nc) as ds:
        assert "analysis_mean_snowdepth_daily" in ds.data_vars
        assert "analysis_increment_snowdepth_daily" in ds.data_vars
        analysis_mean = ds["analysis_mean_snowdepth_daily"].values
        analysis_increment = ds["analysis_increment_snowdepth_daily"].values
        assert np.isclose(analysis_mean[0, 0, 0], 8.0)
        assert np.isclose(analysis_increment[0, 0, 0], 2.0)
        assert np.isnan(analysis_mean[1, 0, 0])
        assert np.isnan(analysis_increment[1, 0, 0])
        assert ds.attrs.get("analysis_increment_definition") == "analysis_increment_<var> = analysis_mean_<var> - ens_mean_<var>"
        assert ds.attrs.get("increment_definition") == "increment_<var> = ens_mean_<var> - open_loop_<var>"


def test_write_project_da_output_grids_uses_carried_propagation_weights(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step = project_dir / "steps" / "step_00_init"
    out_nc = project_dir / "results" / "grids" / "da_output_grids.nc"
    _write_step_member_ncs(
        step,
        np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        [
            np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32),
            np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        ],
        "2023-01-01",
    )
    write_prior_weights(
        step,
        member_ids=["member_001", "member_002"],
        weights=[0.25, 0.75],
        mode="carried_posterior",
        overwrite=True,
    )

    write_project_da_output_grids(step_dirs=[step], output_nc=out_nc)

    with xr.open_dataset(out_nc) as dataset:
        assert np.isclose(dataset["ens_mean_snowdepth_daily"].values[0, 0, 0], 8.0)
        assert np.isclose(dataset["ens_std_snowdepth_daily"].values[0, 0, 0], np.sqrt(12.0), atol=0.001)
        assert dataset.attrs["source_member_weighting"] == "pf_prior_ledger"


def test_write_project_da_output_grids_honors_configured_metrics(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    step_00 = project_dir / "steps" / "step_00_init"
    out_nc = project_dir / "results" / "grids" / "da_output_grids.nc"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2023.yml").write_text(
        "\n".join(
            [
                "start_date: '2023-01-01'",
                "end_date: '2023-01-01'",
                "data_assimilation:",
                "  output:",
                "    grids:",
                "      variables:",
                "        - var: snowdepth_daily",
                "          name: snowdepth_daily",
                "          metrics: [open_loop, ens_mean, analysis_mean]",
                "  assimilation_events:",
                "    - date: '2023-01-01'",
                "      variable: station_hs",
                "      product: STATION",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    step_00.mkdir(parents=True, exist_ok=True)
    (step_00 / "step_00.yml").write_text("start_date: '2023-01-01'\nend_date: '2023-01-01'\n", encoding="utf-8")
    _write_step_member_ncs(
        step_00,
        np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32),
        [
            np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32),
            np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=np.float32),
        ],
        "2023-01-01",
    )
    assim_dir = step_00 / "assim"
    assim_dir.mkdir(parents=True, exist_ok=True)
    (assim_dir / "weights_station_hs_20230101.csv").write_text(
        "member_id,weight\nmember_001,0.25\nmember_002,0.75\n",
        encoding="utf-8",
    )

    written = write_project_da_output_grids(
        step_dirs=[step_00],
        output_nc=out_nc,
    )

    assert written == out_nc
    with xr.open_dataset(out_nc) as ds:
        assert "open_loop_snowdepth_daily" in ds.data_vars
        assert "ens_mean_snowdepth_daily" in ds.data_vars
        assert "analysis_mean_snowdepth_daily" in ds.data_vars
        assert "ens_std_snowdepth_daily" not in ds.data_vars
        assert "ens_min_snowdepth_daily" not in ds.data_vars
        assert "ens_max_snowdepth_daily" not in ds.data_vars
        assert "increment_snowdepth_daily" not in ds.data_vars
        assert "analysis_increment_snowdepth_daily" not in ds.data_vars
        assert ds.attrs.get("summary_variables") == "open_loop,ens_mean,analysis_mean"


def test_project_compact_builder_reports_missing_variable_in_every_member_output(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    step = project_dir / "steps" / "step_00_init"
    out_nc = project_dir / "results" / "grids" / "da_output_grids.nc"
    project_dir.mkdir(parents=True)
    (project_dir / "project_2023.yml").write_text(
        "\n".join(
            [
                "start_date: '2023-01-01'",
                "end_date: '2023-01-01'",
                "data_assimilation:",
                "  output:",
                "    grids:",
                "      variables:",
                "        - var: snowdepth_daily",
                "          metrics: [open_loop, ens_mean]",
                "        - var: swe_daily",
                "          metrics: [open_loop, ens_mean]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_step_member_ncs(
        step,
        np.ones((1, 2, 2), dtype=np.float32),
        [
            np.full((1, 2, 2), 2.0, dtype=np.float32),
            np.full((1, 2, 2), 3.0, dtype=np.float32),
        ],
        "2023-01-01",
    )

    with pytest.raises(ValueError) as exc_info:
        write_project_da_output_grids(step_dirs=[step], output_nc=out_nc)

    message = str(exc_info.value)
    assert "swe_daily" in message
    assert "open_loop/results/output_grids.nc" in message
    assert "member_001/results/output_grids.nc" in message
    assert "member_002/results/output_grids.nc" in message
    assert not out_nc.exists()


def test_compact_output_file_reports_every_missing_configured_metric(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    output_nc = project_dir / "results" / "grids" / "da_output_grids.nc"
    project_dir.mkdir(parents=True)
    (project_dir / "project_2023.yml").write_text(
        "\n".join(
            [
                "data_assimilation:",
                "  output:",
                "    grids:",
                "      variables:",
                "        - var: snowdepth_daily",
                "          metrics: [open_loop, ens_mean, ens_std]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_nc.parent.mkdir(parents=True)
    xr.Dataset(
        {
            "ens_mean_snowdepth_daily": xr.DataArray(
                np.ones((1, 1, 1), dtype=np.float32),
                dims=("time1", "y", "x"),
            )
        }
    ).to_netcdf(output_nc)

    with pytest.raises(ValueError) as exc_info:
        validate_compact_output_file(project_dir=project_dir, output_nc=output_nc)

    message = str(exc_info.value)
    assert "open_loop_snowdepth_daily" in message
    assert "ens_std_snowdepth_daily" in message


def test_output_retention_mode_defaults_to_compact(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
        encoding="utf-8",
    )

    assert output_retention_mode(project_dir) == "compact"


def test_output_retention_mode_defaults_subdomain_to_full(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "run_mode: subdomain\nstart_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
        encoding="utf-8",
    )

    assert output_retention_mode(project_dir) == "full"


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


def test_output_retention_mode_explicit_compact_wins_for_subdomain(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "run_mode: subdomain",
                "start_date: '2022-10-01'",
                "end_date: '2022-10-02'",
                "data_assimilation:",
                "  output:",
                "    retention: compact",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert output_retention_mode(project_dir) == "compact"


def test_output_retention_mode_rejects_unknown_value(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "data_assimilation:\n  output:\n    retention: tiny\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be 'compact' or 'full'"):
        output_retention_mode(project_dir)

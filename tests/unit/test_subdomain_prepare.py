from __future__ import annotations

from pathlib import Path

import yaml
from shapely.geometry import box

from openamundsen_da.subdomain.prepare import _prepare_obs_station_subset, _write_subdomain_setup_yaml


def _write_obs(path: Path) -> None:
    path.write_text("time,snow_depth\n2022-10-01 00:00:00,0.1\n", encoding="utf-8")


def test_prepare_obs_subset_copies_requested_station_ids_without_metadata(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    _write_obs(obs_dir / "latschbloder.csv")
    _write_obs(obs_dir / "proviantdepot.csv")

    _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=0.0,
        crs=None,
        station_ids=["latschbloder"],
    )

    assert (out_dir / "latschbloder.csv").is_file()
    assert not (out_dir / "proviantdepot.csv").exists()


def test_prepare_obs_subset_copies_all_series_without_metadata(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    _write_obs(obs_dir / "latschbloder.csv")
    _write_obs(obs_dir / "proviantdepot.csv")

    _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=0.0,
        crs=None,
        station_ids=None,
    )

    assert (out_dir / "latschbloder.csv").is_file()
    assert (out_dir / "proviantdepot.csv").is_file()


def test_prepare_obs_subset_filters_station_metadata_files(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "stations_snow_depth.csv").write_text(
        "id,x,y\nstation_a,0.5,0.5\nstation_b,10,10\n",
        encoding="utf-8",
    )
    (obs_dir / "stations_da_metadata.csv").write_text(
        "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
        "station_a,10,0.1,true,true\n"
        "station_b,10,0.1,false,true\n",
        encoding="utf-8",
    )
    _write_obs(obs_dir / "station_a.csv")
    _write_obs(obs_dir / "station_b.csv")

    _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=0.0,
        crs=None,
        station_ids=None,
    )

    assert (out_dir / "station_a.csv").is_file()
    assert not (out_dir / "station_b.csv").exists()
    assert (out_dir / "stations_snow_depth.csv").read_text(encoding="utf-8").count("\n") == 2
    assert (out_dir / "stations_da_metadata.csv").read_text(encoding="utf-8").count("\n") == 2


def test_prepare_obs_subset_disables_roles_for_buffer_only_stations(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "stations_snow_depth.csv").write_text(
        "id,x,y\nstation_a,0.5,0.5\nstation_b,1.5,0.5\nstation_c,20,20\n",
        encoding="utf-8",
    )
    (obs_dir / "stations_da_metadata.csv").write_text(
        "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
        "station_a,10,0.1,true,true\n"
        "station_b,10,0.1,true,true\n"
        "station_c,10,0.1,true,true\n",
        encoding="utf-8",
    )
    for station_id in ("station_a", "station_b", "station_c"):
        _write_obs(obs_dir / f"{station_id}.csv")

    stats = _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=1.0,
        crs=None,
        station_ids=None,
    )

    assert (out_dir / "station_a.csv").is_file()
    assert (out_dir / "station_b.csv").is_file()
    assert not (out_dir / "station_c.csv").exists()
    metadata = (out_dir / "stations_da_metadata.csv").read_text(encoding="utf-8")
    assert "station_a,10,0.1,True,True" in metadata
    assert "station_b,10,0.1,False,False" in metadata
    assert stats["obs_stations_selected"] == 2
    assert stats["obs_stations_inside_grid"] == 1
    assert stats["obs_stations_da_active"] == 1
    assert stats["obs_stations_benchmark_active"] == 1
    assert stats["obs_station_series_copied"] == 2


def test_prepare_obs_subset_preserves_leading_zero_station_ids(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "stations_snow_depth.csv").write_text(
        "id,x,y\n04140864,1.5,0.5\n",
        encoding="utf-8",
    )
    (obs_dir / "stations_da_metadata.csv").write_text(
        "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
        "04140864,10,0.1,true,true\n",
        encoding="utf-8",
    )
    _write_obs(obs_dir / "04140864.csv")

    stats = _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=1.0,
        crs=None,
        station_ids=None,
    )

    assert (out_dir / "04140864.csv").is_file()
    snow_metadata = (out_dir / "stations_snow_depth.csv").read_text(encoding="utf-8")
    assert snow_metadata.splitlines()[1].split(",")[0] == "04140864"
    da_metadata = (out_dir / "stations_da_metadata.csv").read_text(encoding="utf-8")
    assert da_metadata.splitlines()[1].split(",")[0] == "04140864"
    assert stats["obs_stations_selected"] == 1
    assert stats["obs_stations_inside_grid"] == 0
    assert stats["obs_stations_da_active"] == 0
    assert stats["obs_stations_benchmark_active"] == 0
    assert stats["obs_station_series_copied"] == 1


def test_write_subdomain_setup_yaml_filters_configured_points(tmp_path: Path) -> None:
    source_cfg = {
        "domain": "full",
        "input_data": {"grids": {}, "meteo": {}},
        "output_data": {
            "timeseries": {
                "points": [
                    {"name": "inside", "x": 0.5, "y": 0.5},
                    {"name": "outside", "x": 10.0, "y": 10.0},
                    {"name": 6988000, "x": 0.6, "y": 0.6},
                    {"name": "kept_without_coords"},
                ]
            }
        },
    }
    out_yaml = _write_subdomain_setup_yaml(
        source_cfg=source_cfg,
        sub_setup_dir=tmp_path / "sd",
        domain="full_sd",
        grids_dir=tmp_path / "sd" / "grids",
        meteo_dir=tmp_path / "sd" / "meteo",
        roi_geom=box(0, 0, 1, 1),
    )

    text = out_yaml.read_text(encoding="utf-8")
    assert "inside" in text
    assert "kept_without_coords" in text
    assert "outside" not in text

    cfg = yaml.safe_load(text)
    kept_names = [point["name"] for point in cfg["output_data"]["timeseries"]["points"]]
    assert "6988000" in kept_names

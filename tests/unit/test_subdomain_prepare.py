from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
import yaml
from shapely.geometry import box

from openamundsen_da.subdomain.prepare import (
    _copy_project_support_inputs,
    _link_country_assets,
    _prepare_obs_station_subset,
    _write_subdomain_setup_yaml,
)


def _write_obs(path: Path) -> None:
    path.write_text("time,snow_depth\n2022-10-01 00:00:00,0.1\n", encoding="utf-8")


def test_copy_project_support_inputs_materializes_acquisition_manifest(
    tmp_path: Path,
) -> None:
    source_setup = tmp_path / "source"
    leaf_setup = tmp_path / "leaf"
    project = leaf_setup / "projects" / "winter"
    project.mkdir(parents=True)
    project_yaml = project / "winter.yml"
    project_yaml.write_text(
        "obs:\n  snowcover:\n"
        "    acquisition_manifest: obs/satellite_acquisition_times.csv\n",
        encoding="utf-8",
    )
    source = source_setup / "obs" / "satellite_acquisition_times.csv"
    source.parent.mkdir(parents=True)
    source.write_text("source,acquisition_time\nscene.tif,2023-01-01T10:00:00Z\n")

    _copy_project_support_inputs(
        source_setup_dir=source_setup,
        target_setup_dir=leaf_setup,
        project_yaml=project_yaml,
    )

    copied = leaf_setup / "obs" / "satellite_acquisition_times.csv"
    assert copied.read_bytes() == source.read_bytes()


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


def test_prepare_obs_subset_uses_da_metadata_coordinates_without_legacy_file(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "stations_da_metadata.csv").write_text(
        "station_id,x,y,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
        "04140864,0.5,0.5,10,0.1,true,false\n"
        "station_buffer,1.5,0.5,10,0.1,true,true\n"
        "station_outside,20,20,10,0.1,true,true\n",
        encoding="utf-8",
    )
    for station_id in ("04140864", "station_buffer", "station_outside"):
        _write_obs(obs_dir / f"{station_id}.csv")

    stats = _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=1.0,
        crs=None,
        station_ids=["unrelated_forcing_station"],
    )

    metadata = pd.read_csv(out_dir / "stations_da_metadata.csv", dtype={"station_id": "string"})
    metadata = metadata.set_index("station_id")
    assert set(metadata.index) == {"04140864", "station_buffer"}
    assert bool(metadata.loc["04140864", "use_for_da"])
    assert not bool(metadata.loc["04140864", "use_for_benchmark"])
    assert not bool(metadata.loc["station_buffer", "use_for_da"])
    assert not bool(metadata.loc["station_buffer", "use_for_benchmark"])
    assert (out_dir / "04140864.csv").is_file()
    assert (out_dir / "station_buffer.csv").is_file()
    assert not (out_dir / "station_outside.csv").exists()
    assert not (out_dir / "stations_snow_depth.csv").exists()
    assert stats == {
        "obs_stations_selected": 2,
        "obs_stations_inside_grid": 1,
        "obs_stations_da_active": 1,
        "obs_stations_benchmark_active": 0,
        "obs_station_series_copied": 2,
    }


def test_prepare_obs_subset_rejects_invalid_id_fallback(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    _write_obs(obs_dir / "snow_station.csv")

    with pytest.raises(ValueError, match="no same-ID observation series"):
        _prepare_obs_station_subset(
            obs_dir=obs_dir,
            out_dir=out_dir,
            geom=box(0, 0, 1, 1),
            buffer_m=0.0,
            crs=None,
            station_ids=["forcing_station"],
        )


def test_prepare_obs_subset_writes_empty_roles_when_no_station_is_in_buffer(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    out_dir = tmp_path / "out"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "stations_da_metadata.csv").write_text(
        "station_id,x,y,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
        "outside,20,20,10,0.1,true,true\n",
        encoding="utf-8",
    )
    _write_obs(obs_dir / "outside.csv")

    stats = _prepare_obs_station_subset(
        obs_dir=obs_dir,
        out_dir=out_dir,
        geom=box(0, 0, 1, 1),
        buffer_m=1.0,
        crs=None,
    )

    metadata = pd.read_csv(out_dir / "stations_da_metadata.csv")
    assert metadata.empty
    assert stats["obs_stations_selected"] == 0
    assert stats["obs_station_series_copied"] == 0


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
                    {"name": "01890168", "x": 0.7, "y": 0.7},
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
        roi_mask=np.asarray([[True, False], [False, False]]),
        roi_transform=from_origin(0, 1, 1, 1),
    )

    text = out_yaml.read_text(encoding="utf-8")
    assert "inside" in text
    assert "kept_without_coords" in text
    assert "outside" not in text
    assert "name: '01890168'" in text

    cfg = yaml.safe_load(text)
    assert cfg["input_data"]["grids"]["dir"] == "grids"
    assert cfg["input_data"]["meteo"]["dir"] == "meteo"
    kept_names = [point["name"] for point in cfg["output_data"]["timeseries"]["points"]]
    assert "6988000" in kept_names
    assert "01890168" in kept_names


def test_output_point_ownership_uses_final_raster_mask(tmp_path: Path) -> None:
    source_cfg = {
        "domain": "full",
        "input_data": {"grids": {}, "meteo": {}},
        "output_data": {
            "timeseries": {
                "points": [
                    {"name": "Breiter Grieskogel", "x": 1.5, "y": 1.5},
                    {"name": "other", "x": 0.5, "y": 1.5},
                ]
            }
        },
    }
    transform = from_origin(0, 2, 1, 1)
    owner_02 = np.asarray([[False, True], [False, False]])
    owner_03 = np.asarray([[True, False], [False, False]])

    yaml_02 = _write_subdomain_setup_yaml(
        source_cfg=source_cfg,
        sub_setup_dir=tmp_path / "AT-07-14-02",
        domain="full_02",
        grids_dir=tmp_path / "AT-07-14-02" / "grids",
        meteo_dir=tmp_path / "AT-07-14-02" / "meteo",
        roi_mask=owner_02,
        roi_transform=transform,
    )
    yaml_03 = _write_subdomain_setup_yaml(
        source_cfg=source_cfg,
        sub_setup_dir=tmp_path / "AT-07-14-03",
        domain="full_03",
        grids_dir=tmp_path / "AT-07-14-03" / "grids",
        meteo_dir=tmp_path / "AT-07-14-03" / "meteo",
        roi_mask=owner_03,
        roi_transform=transform,
    )

    names_02 = [
        point["name"]
        for point in yaml.safe_load(yaml_02.read_text())["output_data"]["timeseries"]["points"]
    ]
    names_03 = [
        point["name"]
        for point in yaml.safe_load(yaml_03.read_text())["output_data"]["timeseries"]["points"]
    ]
    assert "Breiter Grieskogel" in names_02
    assert "Breiter Grieskogel" not in names_03


def test_link_country_assets_exposes_staged_parent_files_offline(tmp_path: Path) -> None:
    source_setup = tmp_path / "setup"
    leaf_env = tmp_path / "leaf" / "env"
    source_env = source_setup / "env"
    source_env.mkdir(parents=True)
    for filename in (
        "CNTR_BN_01M_2020_3857.geojson",
        "CNTR_RG_01M_2020_3857.geojson",
        "CNTR_LB_2020_3857.geojson",
    ):
        (source_env / filename).write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    linked = _link_country_assets(
        source_setup_dir=source_setup,
        leaf_env_dir=leaf_env,
    )

    assert len(linked) == 3
    assert all(path.is_file() for path in linked)
    assert all(path.read_bytes() == (source_env / path.name).read_bytes() for path in linked)

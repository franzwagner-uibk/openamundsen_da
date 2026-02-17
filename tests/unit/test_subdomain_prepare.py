from __future__ import annotations

from pathlib import Path

from shapely.geometry import box

from openamundsen_da.subdomain.prepare import _prepare_obs_station_subset


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

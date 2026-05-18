from __future__ import annotations

from pathlib import Path

from openamundsen_da.util.station_da import read_station_metadata, station_ids_disabled_for_role


def test_read_station_metadata_preserves_leading_zero_station_ids(tmp_path: Path) -> None:
    metadata_path = tmp_path / "stations_da_metadata.csv"
    metadata_path.write_text(
        "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
        "04140864,10,0.1,false,true\n",
        encoding="utf-8",
    )

    metadata = read_station_metadata(metadata_path)

    assert "04140864" in metadata.index
    assert "4140864" not in metadata.index
    assert station_ids_disabled_for_role(metadata, "da") == {"04140864"}

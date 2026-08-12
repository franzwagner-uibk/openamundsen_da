from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from openamundsen_da.util.observation_time import (
    acquisition_from_manifest,
    match_observation_to_model_time,
    match_series_value_to_model_time,
    midnight_fallback,
    parse_model_timestep,
    parse_utc_timestamp,
    read_acquisition_manifest,
    resolve_acquisition_time,
)


def test_model_timestep_accepts_legacy_uppercase_hour_alias() -> None:
    assert parse_model_timestep("3H") == pd.Timedelta(hours=3)


def test_parse_utc_timestamp_requires_timezone() -> None:
    assert parse_utc_timestamp("2023-04-26T10:25:00+02:00", field="event") == datetime(
        2023, 4, 26, 8, 25, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError, match="include a timezone"):
        parse_utc_timestamp("2023-04-26T10:25:00", field="event")


def test_midnight_fallback_is_utc() -> None:
    assert midnight_fallback("2023-04-26") == datetime(2023, 4, 26, tzinfo=timezone.utc)


def test_match_observation_time_converts_to_model_clock() -> None:
    result = match_observation_to_model_time(
        observation_time=datetime(2023, 4, 26, 8, 25, tzinfo=timezone.utc),
        model_times=pd.date_range("2023-04-26 06:00", "2023-04-26 15:00", freq="3h"),
        timezone_config=1,
    )
    assert result.model_time == datetime(2023, 4, 26, 9)
    assert result.offset_seconds == 25 * 60


def test_match_rejects_tie_and_excessive_offset() -> None:
    timeline = pd.date_range("2023-04-26 09:00", "2023-04-26 12:00", freq="3h")
    with pytest.raises(ValueError, match="tied"):
        match_observation_to_model_time(
            observation_time=datetime(2023, 4, 26, 9, 30, tzinfo=timezone.utc),
            model_times=timeline,
            timezone_config=1,
        )
    with pytest.raises(ValueError, match="exceeding half"):
        match_observation_to_model_time(
            observation_time=datetime(2023, 4, 26, 20, tzinfo=timezone.utc),
            model_times=timeline,
            timezone_config=1,
        )


def test_series_match_accepts_unique_value_within_half_timestep() -> None:
    series = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2023-04-26 00:45", "2023-04-26 03:00"]),
    )

    match = match_series_value_to_model_time(
        series,
        model_time=datetime(2023, 4, 26, 0),
        timestep="3h",
        timezone_config=1,
    )

    assert match.matched_time == pd.Timestamp("2023-04-26 00:45")
    assert match.value == 1.0
    assert match.offset_seconds == 45 * 60
    assert match.source_offset_seconds == 45 * 60
    assert match.source_times == (pd.Timestamp("2023-04-26 00:45"),)
    assert match.source_values == (1.0,)
    assert not match.interpolated


def test_series_match_rejects_wrong_year() -> None:
    wrong_year = pd.Series([1.0], index=pd.to_datetime(["2024-04-26 00:00"]))
    with pytest.raises(ValueError, match="exceeding half"):
        match_series_value_to_model_time(
            wrong_year,
            model_time=datetime(2023, 4, 26),
            timestep="3h",
            timezone_config=1,
        )


def test_series_match_interpolates_symmetric_tie() -> None:
    tied = pd.Series(
        [1.0, 3.0],
        index=pd.to_datetime(["2023-04-25 23:00", "2023-04-26 01:00"]),
    )

    match = match_series_value_to_model_time(
        tied,
        model_time=datetime(2023, 4, 26),
        timestep="3h",
        timezone_config=1,
    )

    assert match.matched_time == pd.Timestamp("2023-04-26 00:00")
    assert match.value == 2.0
    assert match.offset_seconds == 0.0
    assert match.source_offset_seconds == 60 * 60
    assert match.source_times == (
        pd.Timestamp("2023-04-25 23:00"),
        pd.Timestamp("2023-04-26 01:00"),
    )
    assert match.source_values == (1.0, 3.0)
    assert match.interpolated


def test_series_match_accepts_inclusive_24_hour_symmetric_span() -> None:
    tied = pd.Series(
        [1.0, 3.0],
        index=pd.to_datetime(["2023-04-25 12:00", "2023-04-26 12:00"]),
    )

    match = match_series_value_to_model_time(
        tied,
        model_time=datetime(2023, 4, 26),
        timestep="24h",
        timezone_config=1,
    )

    assert match.value == 2.0
    assert match.source_offset_seconds == 12 * 60 * 60
    assert match.interpolated


@pytest.mark.parametrize(
    ("timestamps", "values", "timestep", "message"),
    [
        (
            ["2023-04-25 11:00", "2023-04-26 13:00"],
            [1.0, 3.0],
            "48h",
            "more than 24 hours apart",
        ),
        (
            ["2023-04-25 23:00", "2023-04-25 23:00", "2023-04-26 01:00"],
            [1.0, 2.0, 3.0],
            "3h",
            "exactly two",
        ),
    ],
)
def test_series_match_rejects_malformed_symmetric_ties(
    timestamps: list[str],
    values: list[float],
    timestep: str,
    message: str,
) -> None:
    tied = pd.Series(values, index=pd.to_datetime(timestamps))

    with pytest.raises(ValueError, match=message):
        match_series_value_to_model_time(
            tied,
            model_time=datetime(2023, 4, 26),
            timestep=timestep,
            timezone_config=1,
        )


def test_series_match_can_require_exact_model_output_time() -> None:
    series = pd.Series([1.0], index=pd.to_datetime(["2023-04-26 00:30"]))
    with pytest.raises(ValueError, match="exact model timestamp"):
        match_series_value_to_model_time(
            series,
            model_time=datetime(2023, 4, 26),
            timestep="3h",
            timezone_config=1,
            require_exact=True,
        )

    tied = pd.Series(
        [1.0, 3.0],
        index=pd.to_datetime(["2023-04-25 23:00", "2023-04-26 01:00"]),
    )
    with pytest.raises(ValueError, match="exact model timestamp"):
        match_series_value_to_model_time(
            tied,
            model_time=datetime(2023, 4, 26),
            timestep="3h",
            timezone_config=1,
            require_exact=True,
        )


def test_acquisition_manifest_is_strict_and_matches_basename(tmp_path) -> None:
    path = tmp_path / "acquisition.csv"
    path.write_text(
        "product,source,product_identity,acquisition_time,time_source,time_quality\n"
        "S2,/archive/scene.tif,S2_TEST,2023-04-26T10:25:00Z,provenance_manifest,verified\n",
        encoding="utf-8",
    )
    frame = read_acquisition_manifest(path)
    assert len(frame) == 1
    row = acquisition_from_manifest(path, product="s2", source="scene.tif")
    assert row["acquisition_time"] == "2023-04-26T10:25:00Z"


def test_acquisition_manifest_rejects_duplicate_source(tmp_path) -> None:
    path = tmp_path / "acquisition.csv"
    path.write_text(
        "product,source,product_identity,acquisition_time,time_source,time_quality\n"
        "S2,scene.tif,S2_A,2023-04-26T10:25:00Z,manifest,verified\n"
        "S2,scene.tif,S2_B,2023-04-26T10:25:00Z,manifest,verified\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source"):
        read_acquisition_manifest(path)


def test_acquisition_time_source_precedence_and_midnight_fallback(tmp_path, monkeypatch) -> None:
    source = tmp_path / "scene_20230426T102500.tif"
    monkeypatch.setattr(
        "openamundsen_da.util.observation_time._raster_metadata_acquisition_time",
        lambda _path: datetime(2023, 4, 26, 9, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "openamundsen_da.util.observation_time._sidecar_acquisition_time",
        lambda _path: datetime(2023, 4, 26, 10, 0, tzinfo=timezone.utc),
    )
    resolved = resolve_acquisition_time(
        source_path=source,
        product="S2",
        observation_date="2023-04-26",
        cf_time="2023-04-26T08:00:00Z",
        filename_parser="sentinel_1",
    )
    assert resolved.source == "cf_time_coordinate"
    assert resolved.value == datetime(2023, 4, 26, 8, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("openamundsen_da.util.observation_time._raster_metadata_acquisition_time", lambda _path: None)
    monkeypatch.setattr("openamundsen_da.util.observation_time._sidecar_acquisition_time", lambda _path: None)
    fallback = resolve_acquisition_time(
        source_path=tmp_path / "scene.tif",
        product="S2",
        observation_date="2023-04-26",
    )
    assert fallback.source == "midnight_fallback"
    assert fallback.quality == "fallback_midnight"

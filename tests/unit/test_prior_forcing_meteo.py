from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openamundsen_da.core.constants import (
    HUMIDITY_METHOD_DEW_POINT,
    HUMIDITY_METHOD_RELATIVE_HUMIDITY,
)
from openamundsen_da.core.prior_forcing import _read_prior_params
from openamundsen_da.methods.pf.rejuvenate import _read_rejuvenation_params
from openamundsen_da.util.humidity import (
    dew_point_to_relative_humidity,
    perturb_relative_humidity_via_dew_point,
    relative_humidity_to_dew_point,
)
from openamundsen_da.util.meteo import filter_and_write_meteo
from openamundsen_da.util.stats import sample_shortwave_factor


def test_filter_and_write_meteo_applies_dew_point_humidity_perturbation(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    (src_dir / "stations.csv").write_text("id,name,x,y,alt\nstation,Station,0,0,0\n", encoding="utf-8")
    (src_dir / "station.csv").write_text(
        "\n".join(
            [
                "date,temp,precip,rel_hum,sw_in",
                "2023-01-01T00:00:00,273.15,0.0,95.0,0.0",
                "2023-01-01T03:00:00,274.15,2.0,10.0,10.0",
            ]
        ),
        encoding="utf-8",
    )

    filter_and_write_meteo(
        src_dir=src_dir,
        dst_dir=dst_dir,
        start=pd.Timestamp("2023-01-01T00:00:00"),
        end=pd.Timestamp("2023-01-01T03:00:00"),
        delta_t=1.5,
        f_p=2.0,
        delta_rh=1.0,
        f_sw=3.0,
    )

    out = pd.read_csv(dst_dir / "station.csv")
    np.testing.assert_allclose(out["temp"].to_numpy(), [274.6, 275.6])
    assert out["precip"].tolist() == [0.0, 4.0]
    assert out["rel_hum"].between(0.0, 100.0).all()
    assert not np.isclose(out["rel_hum"].iloc[0], 100.0)
    assert out["sw_in"].tolist() == [0.0, 30.0]
    assert (dst_dir / "stations.csv").exists()


def test_filter_and_write_meteo_legacy_relative_humidity_method_clips(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    (src_dir / "station.csv").write_text(
        "\n".join(
            [
                "date,temp,precip,rel_hum,sw_in",
                "2023-01-01T00:00:00,273.15,0.0,95.0,0.0",
                "2023-01-01T03:00:00,274.15,2.0,10.0,10.0",
            ]
        ),
        encoding="utf-8",
    )

    filter_and_write_meteo(
        src_dir=src_dir,
        dst_dir=dst_dir,
        start=pd.Timestamp("2023-01-01T00:00:00"),
        end=pd.Timestamp("2023-01-01T03:00:00"),
        delta_rh=10.0,
        humidity_perturbation_method=HUMIDITY_METHOD_RELATIVE_HUMIDITY,
    )

    out = pd.read_csv(dst_dir / "station.csv")
    assert out["rel_hum"].tolist() == [100.0, 20.0]


def test_filter_and_write_meteo_perturbs_integer_precip_and_shortwave(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    (src_dir / "station.csv").write_text(
        "\n".join(
            [
                "date,temp,precip,rel_hum,sw_in",
                "2023-01-01T00:00:00,0,0,80,0",
                "2023-01-01T03:00:00,1,2,85,1",
                "2023-01-01T06:00:00,2,10,90,10",
            ]
        ),
        encoding="utf-8",
    )

    filter_and_write_meteo(
        src_dir=src_dir,
        dst_dir=dst_dir,
        start=pd.Timestamp("2023-01-01T00:00:00"),
        end=pd.Timestamp("2023-01-01T06:00:00"),
        f_p=1.25,
        f_sw=0.992303298,
    )

    out = pd.read_csv(dst_dir / "station.csv")
    np.testing.assert_allclose(out["precip"].to_numpy(), [0.0, 2.5, 12.5])
    np.testing.assert_allclose(out["sw_in"].to_numpy(), [0.0, 1.0, 10.0])


def test_filter_and_write_meteo_formats_known_columns_to_storage_precision(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    (src_dir / "station.csv").write_text(
        "\n".join(
            [
                "date,temp,precip,sw_in,rel_hum,wind_speed,wind_dir,unknown",
                "2023-01-01T03:00:00,273.154,0.0149,122.6,88.124,1.234,359.94,1.23456789",
            ]
        ),
        encoding="utf-8",
    )

    filter_and_write_meteo(
        src_dir=src_dir,
        dst_dir=dst_dir,
        start=pd.Timestamp("2023-01-01T03:00:00"),
        end=pd.Timestamp("2023-01-01T03:00:00"),
    )

    assert (dst_dir / "station.csv").read_text(encoding="utf-8").splitlines() == [
        "date,temp,precip,sw_in,rel_hum,wind_speed,wind_dir,unknown",
        "2023-01-01 03:00:00,273.2,0.01,123,88.12,1.23,359.9,1.23456789",
    ]


def test_new_prior_and_rejuvenation_sigmas_default_to_zero(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "demo.yml").write_text(
        "\n".join(
            [
                "start_date: 2022-10-01",
                "end_date: 2023-06-30",
                "data_assimilation:",
                "  prior_forcing:",
                "    ensemble_size: 5",
                "    random_seed: 42",
                "    sigma_t: 0.5",
                "    mu_p: 0.0",
                "    sigma_p: 0.5",
                "  rejuvenation:",
                "    sigma_t: 0.2",
                "    sigma_p: 0.2",
                "    seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    prior = _read_prior_params(project_dir)
    rejuvenation = _read_rejuvenation_params(project_dir)

    assert prior.sigma_rh == 0.0
    assert prior.sigma_sw == 0.0
    assert rejuvenation.sigma_rh == 0.0
    assert rejuvenation.sigma_sw == 0.0
    assert prior.humidity_perturbation_method == HUMIDITY_METHOD_DEW_POINT
    assert rejuvenation.humidity_perturbation_method == HUMIDITY_METHOD_DEW_POINT


def test_prior_and_rejuvenation_read_humidity_method(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "demo.yml").write_text(
        "\n".join(
            [
                "start_date: 2022-10-01",
                "end_date: 2023-06-30",
                "data_assimilation:",
                "  prior_forcing:",
                "    ensemble_size: 5",
                "    random_seed: 42",
                "    sigma_t: 0.5",
                "    mu_p: 0.0",
                "    sigma_p: 0.5",
                "    humidity_perturbation_method: relative_humidity",
                "  rejuvenation:",
                "    humidity_perturbation_method: relative_humidity",
                "    seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    prior = _read_prior_params(project_dir)
    rejuvenation = _read_rejuvenation_params(project_dir)

    assert prior.humidity_perturbation_method == HUMIDITY_METHOD_RELATIVE_HUMIDITY
    assert rejuvenation.humidity_perturbation_method == HUMIDITY_METHOD_RELATIVE_HUMIDITY


def test_rejuvenation_rejects_humidity_method_mismatch(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "demo.yml").write_text(
        "\n".join(
            [
                "start_date: 2022-10-01",
                "end_date: 2023-06-30",
                "data_assimilation:",
                "  prior_forcing:",
                "    ensemble_size: 5",
                "    random_seed: 42",
                "    sigma_t: 0.5",
                "    mu_p: 0.0",
                "    sigma_p: 0.5",
                "    humidity_perturbation_method: relative_humidity",
                "  rejuvenation:",
                "    humidity_perturbation_method: dew_point",
                "    seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must match"):
        _read_rejuvenation_params(project_dir)


def test_prior_rejects_invalid_humidity_method(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "demo.yml").write_text(
        "\n".join(
            [
                "start_date: 2022-10-01",
                "end_date: 2023-06-30",
                "data_assimilation:",
                "  prior_forcing:",
                "    ensemble_size: 5",
                "    random_seed: 42",
                "    sigma_t: 0.5",
                "    mu_p: 0.0",
                "    sigma_p: 0.5",
                "    humidity_perturbation_method: bogus",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="humidity_perturbation_method"):
        _read_prior_params(project_dir)


def test_dew_point_roundtrip_and_kelvin_handling() -> None:
    temp_k = np.array([263.15, 273.15, 283.15])
    rel_hum = np.array([60.0, 80.0, 100.0])

    dew_point = relative_humidity_to_dew_point(temp_k, rel_hum)
    out = dew_point_to_relative_humidity(temp_k, dew_point)

    np.testing.assert_allclose(out, rel_hum)
    assert dew_point[-1] == pytest.approx(10.0)


def test_dew_point_perturbation_rejects_invalid_relative_humidity() -> None:
    with pytest.raises(ValueError, match="Relative humidity values"):
        relative_humidity_to_dew_point([273.15, 273.15], [0.0, 50.0])
    with pytest.raises(ValueError, match="Relative humidity values"):
        relative_humidity_to_dew_point([273.15], [101.0])


def test_dew_point_perturbation_caps_at_perturbed_air_temperature() -> None:
    out = perturb_relative_humidity_via_dew_point(
        [273.15],
        [95.0],
        delta_tdew=10.0,
        delta_t=1.0,
    )

    assert out[0] == pytest.approx(100.0)


def test_dew_point_method_requires_temperature_when_humidity_is_perturbed(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    (src_dir / "station.csv").write_text(
        "\n".join(
            [
                "date,precip,rel_hum,sw_in",
                "2023-01-01T00:00:00,0.0,95.0,0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires air temperature"):
        filter_and_write_meteo(
            src_dir=src_dir,
            dst_dir=dst_dir,
            start=pd.Timestamp("2023-01-01T00:00:00"),
            end=pd.Timestamp("2023-01-01T00:00:00"),
            delta_rh=1.0,
        )


def test_shortwave_factor_sampler_remains_positive() -> None:
    generator = np.random.default_rng(42)
    assert sample_shortwave_factor(generator, 0.0) == 1.0
    assert sample_shortwave_factor(generator, 0.5) > 0.0

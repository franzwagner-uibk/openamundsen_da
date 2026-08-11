from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openamundsen_da.core.prior_forcing import (
    _read_step_window,
    _read_prior_params,
    build_prior_ensemble,
    validate_prior_forcing_manifest,
)
from openamundsen_da.methods.pf.rejuvenate import _read_next_step_dates, _read_rejuvenation_params
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
                "2023-01-01T03:00:00,278.15,2.0,10.0,10.0",
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
    expected_rh = perturb_relative_humidity_via_dew_point(
        [273.15, 278.15],
        [95.0, 10.0],
        1.0,
        delta_t=1.5,
    )
    np.testing.assert_allclose(out["temp"].to_numpy(), [274.6, 279.6])
    np.testing.assert_allclose(out["precip"].to_numpy(), [0.0, 4.0])
    np.testing.assert_allclose(out["rel_hum"].to_numpy(), np.round(expected_rh, 2))
    np.testing.assert_allclose(out["sw_in"].to_numpy(), [0.0, 30.0])
    assert (dst_dir / "stations.csv").exists()


def test_dew_point_humidity_transform_changes_with_temperature_and_caps() -> None:
    warmed = perturb_relative_humidity_via_dew_point(
        [273.15],
        [80.0],
        delta_tdew=0.0,
        delta_t=2.0,
    )
    saturated = perturb_relative_humidity_via_dew_point(
        [273.15],
        [95.0],
        delta_tdew=20.0,
        delta_t=-5.0,
    )

    np.testing.assert_allclose(warmed, [69.28720908942724])
    np.testing.assert_allclose(saturated, [100.0])


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


def test_prior_and_rejuvenation_require_configured_scientific_sigmas(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="sigma_rh"):
        _read_prior_params(project_dir)
    with pytest.raises(ValueError, match="sigma_rh"):
        _read_rejuvenation_params(project_dir)


def test_prior_forcing_manifest_controls_reuse_and_detects_output_tampering(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    input_meteo_dir = setup_dir / "meteo"
    project_dir = setup_dir / "projects" / "demo"
    step_dir = project_dir / "steps" / "step_00"
    input_meteo_dir.mkdir(parents=True)
    step_dir.mkdir(parents=True)
    (setup_dir / "setup.yml").write_text("input_data: {}\n", encoding="utf-8")
    (input_meteo_dir / "stations.csv").write_text(
        "id,name,x,y,alt\nstation,Station,0,0,0\n",
        encoding="utf-8",
    )
    (input_meteo_dir / "station.csv").write_text(
        "date,temp,precip,rel_hum,sw_in\n"
        "2023-01-01T00:00:00,273.15,1.0,80.0,100.0\n"
        "2023-01-02T00:00:00,274.15,2.0,81.0,110.0\n"
        "2023-01-03T00:00:00,275.15,3.0,82.0,120.0\n",
        encoding="utf-8",
    )
    (project_dir / "demo.yml").write_text(
        "end_date: 2023-01-03T00:00:00\n"
        "data_assimilation:\n"
        "  prior_forcing:\n"
        "    ensemble_size: 2\n"
        "    random_seed: 42\n"
        "    sigma_t: 0.5\n"
        "    mu_p: 0.1\n"
        "    sigma_p: 0.2\n"
        "    sigma_rh: 0.3\n"
        "    sigma_sw: 0.05\n",
        encoding="utf-8",
    )
    (step_dir / "step_00.yml").write_text(
        "start_date: 2023-01-01T00:00:00\n"
        "end_date: 2023-01-01T00:00:00\n",
        encoding="utf-8",
    )

    build_prior_ensemble(input_meteo_dir, project_dir, step_dir, max_workers=1)

    manifest = validate_prior_forcing_manifest(
        input_meteo_dir=input_meteo_dir,
        project_dir=project_dir,
        step_dir=step_dir,
    )
    assert manifest["rng_scheme"] == "keyed-v1"
    assert manifest["window_end"] == "2023-01-01T00:00:00"
    assert [row["member"] for row in manifest["members"]] == ["member_001", "member_002"]
    assert (step_dir / "ensembles" / "prior" / "member_001" / "INFO.txt").is_file()
    generated = pd.read_csv(
        step_dir / "ensembles" / "prior" / "member_001" / "meteo" / "station.csv"
    )
    assert pd.to_datetime(generated["date"]).tolist() == [pd.Timestamp("2023-01-01T00:00:00")]

    build_prior_ensemble(input_meteo_dir, project_dir, step_dir, max_workers=1)

    generated_path = step_dir / "ensembles" / "prior" / "member_001" / "meteo" / "station.csv"
    generated_path.write_text(
        generated_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="output_inventory_sha256"):
        build_prior_ensemble(input_meteo_dir, project_dir, step_dir, max_workers=1)


def test_prior_and_rejuvenated_forcing_use_exact_step_window(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "demo"
    step_dir = project_dir / "steps" / "step_01"
    step_dir.mkdir(parents=True)
    (setup_dir / "setup.yml").write_text("input_data: {}\n", encoding="utf-8")
    (project_dir / "demo.yml").write_text(
        "start_date: 2023-01-01T00:00:00\n"
        "end_date: 2023-09-30T21:00:00\n",
        encoding="utf-8",
    )
    (step_dir / "step_01.yml").write_text(
        "start_date: 2023-01-07T00:00:00\n"
        "end_date: 2023-01-12T21:00:00\n",
        encoding="utf-8",
    )

    expected = (pd.Timestamp("2023-01-07T00:00:00"), pd.Timestamp("2023-01-12T21:00:00"))
    assert _read_step_window(step_dir) == expected
    assert _read_next_step_dates(step_dir) == expected


def test_step_window_requires_step_end_date_even_when_project_has_end(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "demo"
    step_dir = project_dir / "steps" / "step_01"
    step_dir.mkdir(parents=True)
    (project_dir / "demo.yml").write_text("end_date: 2023-09-30T21:00:00\n", encoding="utf-8")
    (step_dir / "step_01.yml").write_text("start_date: 2023-01-07T00:00:00\n", encoding="utf-8")

    with pytest.raises(ValueError, match="end_date"):
        _read_step_window(step_dir)
    with pytest.raises(ValueError, match="end_date"):
        _read_next_step_dates(step_dir)


def test_removed_humidity_method_config_is_rejected(tmp_path: Path) -> None:
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
                "    humidity_perturbation_method: dew_point",
                "  rejuvenation:",
                "    seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="humidity_perturbation_method"):
        _read_prior_params(project_dir)
    with pytest.raises(ValueError, match="humidity_perturbation_method"):
        _read_rejuvenation_params(project_dir)


def test_dew_point_roundtrip_and_kelvin_handling() -> None:
    temp_k = np.array([263.15, 273.15, 283.15])
    rel_hum = np.array([60.0, 80.0, 100.0])

    dew_point = relative_humidity_to_dew_point(temp_k, rel_hum)
    out = dew_point_to_relative_humidity(temp_k, dew_point)

    np.testing.assert_allclose(out, rel_hum)
    assert dew_point[-1] == pytest.approx(10.0)


def test_dew_point_perturbation_clips_relative_humidity_bounds() -> None:
    dew_point = relative_humidity_to_dew_point([273.15, 273.15], [0.0, 150.0])
    out = dew_point_to_relative_humidity([273.15, 273.15], dew_point)

    np.testing.assert_allclose(out, [1e-6, 100.0], rtol=1e-6, atol=1e-10)


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

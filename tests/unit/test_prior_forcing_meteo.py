from pathlib import Path

import numpy as np
import pandas as pd

from openamundsen_da.core.prior_forcing import _read_prior_params
from openamundsen_da.methods.pf.rejuvenate import _read_rejuvenation_params
from openamundsen_da.util.meteo import (
    filter_and_write_meteo,
    perturb_relative_humidity_via_dew_point,
)
from openamundsen_da.util.stats import sample_shortwave_factor


def test_filter_and_write_meteo_applies_four_variable_perturbations_and_guards(tmp_path: Path) -> None:
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
        delta_rh=10.0,
        f_sw=3.0,
    )

    out = pd.read_csv(dst_dir / "station.csv")
    np.testing.assert_allclose(out["temp"].to_numpy(), [274.65, 279.65])
    np.testing.assert_allclose(out["precip"].to_numpy(), [0.0, 4.0])
    np.testing.assert_allclose(out["rel_hum"].to_numpy(), [100.0, 21.191555068819333])
    np.testing.assert_allclose(out["sw_in"].to_numpy(), [0.0, 30.0])
    assert (dst_dir / "stations.csv").exists()


def test_dew_point_humidity_transform_changes_with_temperature_and_caps() -> None:
    warmed = perturb_relative_humidity_via_dew_point(
        pd.Series([273.15]),
        pd.Series([80.0]),
        delta_t=2.0,
        delta_dew_point=0.0,
    )
    saturated = perturb_relative_humidity_via_dew_point(
        pd.Series([273.15]),
        pd.Series([95.0]),
        delta_t=-5.0,
        delta_dew_point=20.0,
    )

    np.testing.assert_allclose(warmed.to_numpy(), [69.28720908942724])
    np.testing.assert_allclose(saturated.to_numpy(), [100.0])


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
    np.testing.assert_allclose(out["sw_in"].to_numpy(), [0.0, 0.992303298, 9.92303298])


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


def test_shortwave_factor_sampler_remains_positive() -> None:
    generator = np.random.default_rng(42)
    assert sample_shortwave_factor(generator, 0.0) == 1.0
    assert sample_shortwave_factor(generator, 0.5) > 0.0

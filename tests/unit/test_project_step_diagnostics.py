from __future__ import annotations

from pyproj import CRS

from openamundsen_da.pipeline import project as project_mod
from openamundsen_da.methods.wet_snow.classify import WetSnowClassificationConfig
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig


def test_compute_prior_step_diagnostics_runs_existing_and_new_diagnostics(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str | None, str | None, float | None]] = []

    def _record(name: str):
        def _inner(*args, **kwargs):
            calls.append(
                (
                    name,
                    kwargs.get("variable"),
                    kwargs.get("classification_method"),
                    kwargs.get("liquid_water_amount_threshold_mm"),
                )
            )

        return _inner

    monkeypatch.setattr(project_mod, "compute_step_scf_daily_for_all_members", _record("scf"))
    monkeypatch.setattr(project_mod, "compute_step_roi_mean_daily_for_all_members", _record("roi"))
    monkeypatch.setattr(project_mod, "classify_step_wet_snow", _record("wet_classify"))
    monkeypatch.setattr(project_mod, "compute_step_wet_snow_daily_for_all_members", _record("wet_daily"))

    cfg = project_mod.OrchestratorConfig(
        project_dir=tmp_path / "projects" / "project_2022_2023",
        setup_dir=tmp_path,
        overwrite=True,
    )
    lc_cfg = LandcoverMaskConfig(
        enabled=False,
        path=None,
        classes=tuple(),
        project_crs=CRS.from_epsg(4326),
    )

    project_mod._compute_prior_step_diagnostics(
        cfg=cfg,
        step_dir=tmp_path / "steps" / "step_00_init",
        roi=tmp_path / "env" / "roi.gpkg",
        lc_cfg=lc_cfg,
        workers=2,
        scf_enabled=True,
        wet_snow_enabled=True,
        wet_snow_classification=WetSnowClassificationConfig(
            method="liquid_water_amount",
            threshold_percent=float("nan"),
            liquid_water_amount_threshold_mm=5.0,
        ),
    )

    assert calls == [
        ("scf", None, None, None),
        ("roi", "swe", None, None),
        ("roi", "hs", None, None),
        ("wet_classify", None, "liquid_water_amount", 5.0),
        ("wet_daily", None, None, None),
    ]

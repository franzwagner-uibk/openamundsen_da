"""Gaussian-likelihood weighting for fraction-based ROI observations.

Supports ROI snow-cover fraction and ROI wet-snow fraction observations
against model-derived H(x) values per ensemble member.

Configuration
- H(x) configuration is read from project YAML (data_assimilation.h_of_x) and is required.
- Likelihood configuration is read from project YAML (likelihood block).
  Falls back to sensible defaults if missing.

Logging
- Uses loguru with a green timestamp format defined in constants.LOGURU_FORMAT.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import (
    LIKELIHOOD_BLOCK,
    LIK_OBS_SIGMA,
    LIK_USE_BINOMIAL,
    LIK_SIGMA_FLOOR,
    LIK_SIGMA_CLOUD_SCALE,
    LIK_MIN_SIGMA,
    LIK_MIN_SUPPORT_COVERAGE_RATIO,
    LIK_MIN_MODEL_FINITE_FRACTION,
    LIK_MIN_WET_BANDS,
    LIK_MIN_WET_PIXELS_TOTAL,
    OBS_DIR_NAME,
    RESAMPLING_BLOCK,
    RESAMPLING_ESS_THRESHOLD,
    RESAMPLING_ESS_THRESHOLD_RATIO,
)
from openamundsen_da.io.paths import (
    list_member_dirs,
    default_results_dir,
    find_project_yaml,
    infer_project_dir,
    infer_setup_dir_from_project,
)
from openamundsen_da.methods.h_of_x.model_scf import compute_model_scf, load_hofx_from_project
from openamundsen_da.methods.pf.fraction_support import ObservationSupportMask, load_observation_support_mask
from openamundsen_da.methods.wet_snow.area import compute_model_wet_snow_fraction, compute_model_wet_snow_line
from openamundsen_da.util.stats import gaussian_logpdf, normalize_log_weights, effective_sample_size, compute_obs_sigma
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.util.config_validators import require_mapping
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig, resolve_landcover_mask
from openamundsen_da.observer.fraction_obs import (
    build_obs_candidate_paths,
    build_obs_csv_path,
    resolve_obs_product_tag,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.da_observables import weights_csv_name
from openamundsen_da.util.uncertainty_common import parse_assimilation_block


@dataclass
class LikelihoodParams:
    obs_sigma: float = 0.10
    use_binomial: bool = True
    sigma_floor: float = 0.05
    sigma_cloud_scale: float = 0.10
    min_sigma: float = 0.03
    min_support_coverage_ratio: float = 0.0
    min_model_finite_fraction: float = 1.0
    min_wet_pixels_total: int = 0
    min_wet_bands: int = 0


@dataclass(frozen=True)
class FractionModelEvaluation:
    value_model: float
    value_model_full_roi: float
    value_model_obs_support: float
    full_roi_n_valid: int | None = None
    obs_support_n_valid: int | None = None


@dataclass(frozen=True)
class FractionUncertaintyAssimilationConfig:
    enabled: bool
    sigma_mode: str
    aggregate_metric: str | None


ScfUncertaintyAssimilationConfig = FractionUncertaintyAssimilationConfig
WetSnowUncertaintyAssimilationConfig = FractionUncertaintyAssimilationConfig


def _disabled_uncertainty_config() -> FractionUncertaintyAssimilationConfig:
    return FractionUncertaintyAssimilationConfig(enabled=False, sigma_mode="formula", aggregate_metric=None)


def _read_fraction_uncertainty_assimilation_config(
    project_dir: Path,
    observable: str,
) -> FractionUncertaintyAssimilationConfig:
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    da_cfg = require_mapping(cfg.get("data_assimilation"), path="project.data_assimilation")
    unc_root = da_cfg.get("uncertainty")
    if unc_root is None:
        return _disabled_uncertainty_config()
    unc_cfg = require_mapping(unc_root, path="project.data_assimilation.uncertainty")
    unc_raw = unc_cfg.get(observable)
    if unc_raw is None:
        return _disabled_uncertainty_config()
    unc = require_mapping(unc_raw, path=f"project.data_assimilation.uncertainty.{observable}")
    if not bool(unc.get("enabled", False)):
        return _disabled_uncertainty_config()

    assim_path = f"project.data_assimilation.uncertainty.{observable}.assimilation"
    assim = require_mapping(unc.get("assimilation"), path=assim_path)
    sigma_mode, aggregate_metric = parse_assimilation_block(
        assim,
        path=assim_path,
    )
    return FractionUncertaintyAssimilationConfig(
        enabled=True,
        sigma_mode=sigma_mode,
        aggregate_metric=aggregate_metric,
    )


def _read_scf_uncertainty_assimilation_config(project_dir: Path) -> ScfUncertaintyAssimilationConfig:
    return _read_fraction_uncertainty_assimilation_config(project_dir, "scf")


def _read_wet_snow_uncertainty_assimilation_config(project_dir: Path) -> WetSnowUncertaintyAssimilationConfig:
    return _read_fraction_uncertainty_assimilation_config(project_dir, "wet_snow")


def _coerce_float(raw: object, *, path: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value at {path}: {raw!r}") from exc


def _coerce_bool(raw: object, *, path: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    if isinstance(raw, (int, float)) and raw in {0, 1}:
        return bool(raw)
    raise ValueError(f"Invalid boolean value at {path}: {raw!r}")


def _coerce_int(raw: object, *, path: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value at {path}: {raw!r}") from exc


def _read_likelihood_from_project(project_dir: Path, observable: str) -> LikelihoodParams:
    """Read likelihood settings from project YAML for a given observable if available."""
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    da_cfg_raw = cfg.get("data_assimilation")
    da_cfg = {} if da_cfg_raw is None else require_mapping(da_cfg_raw, path="project.data_assimilation")

    lk_root_raw = da_cfg.get(LIKELIHOOD_BLOCK)
    lk_root_path = f"project.data_assimilation.{LIKELIHOOD_BLOCK}"
    if lk_root_raw is None:
        return LikelihoodParams()

    lk_root = require_mapping(lk_root_raw, path=lk_root_path)
    lk_raw = lk_root.get(observable)
    if lk_raw is None:
        lk = lk_root
        lk_path = lk_root_path
    else:
        lk_path = f"{lk_root_path}.{observable}"
        lk = require_mapping(lk_raw, path=lk_path)

    params = LikelihoodParams()
    if LIK_OBS_SIGMA in lk:
        params.obs_sigma = _coerce_float(lk[LIK_OBS_SIGMA], path=f"{lk_path}.{LIK_OBS_SIGMA}")
    if LIK_USE_BINOMIAL in lk:
        params.use_binomial = _coerce_bool(lk[LIK_USE_BINOMIAL], path=f"{lk_path}.{LIK_USE_BINOMIAL}")
    if LIK_SIGMA_FLOOR in lk:
        params.sigma_floor = _coerce_float(lk[LIK_SIGMA_FLOOR], path=f"{lk_path}.{LIK_SIGMA_FLOOR}")
    if LIK_SIGMA_CLOUD_SCALE in lk:
        params.sigma_cloud_scale = _coerce_float(
            lk[LIK_SIGMA_CLOUD_SCALE],
            path=f"{lk_path}.{LIK_SIGMA_CLOUD_SCALE}",
        )
    if LIK_MIN_SIGMA in lk:
        params.min_sigma = _coerce_float(lk[LIK_MIN_SIGMA], path=f"{lk_path}.{LIK_MIN_SIGMA}")
    if LIK_MIN_SUPPORT_COVERAGE_RATIO in lk:
        params.min_support_coverage_ratio = _coerce_float(
            lk[LIK_MIN_SUPPORT_COVERAGE_RATIO],
            path=f"{lk_path}.{LIK_MIN_SUPPORT_COVERAGE_RATIO}",
        )
        if not (0.0 <= params.min_support_coverage_ratio <= 1.0):
            raise ValueError(
                f"{lk_path}.{LIK_MIN_SUPPORT_COVERAGE_RATIO} must be within [0, 1]"
            )
    if LIK_MIN_MODEL_FINITE_FRACTION in lk:
        params.min_model_finite_fraction = _coerce_float(
            lk[LIK_MIN_MODEL_FINITE_FRACTION],
            path=f"{lk_path}.{LIK_MIN_MODEL_FINITE_FRACTION}",
        )
        if not (0.0 <= params.min_model_finite_fraction <= 1.0):
            raise ValueError(
                f"{lk_path}.{LIK_MIN_MODEL_FINITE_FRACTION} must be within [0, 1]"
            )
    if LIK_MIN_WET_PIXELS_TOTAL in lk:
        params.min_wet_pixels_total = _coerce_int(
            lk[LIK_MIN_WET_PIXELS_TOTAL],
            path=f"{lk_path}.{LIK_MIN_WET_PIXELS_TOTAL}",
        )
        if params.min_wet_pixels_total < 0:
            raise ValueError(f"{lk_path}.{LIK_MIN_WET_PIXELS_TOTAL} must be >= 0")
    if LIK_MIN_WET_BANDS in lk:
        params.min_wet_bands = _coerce_int(
            lk[LIK_MIN_WET_BANDS],
            path=f"{lk_path}.{LIK_MIN_WET_BANDS}",
        )
        if params.min_wet_bands < 0:
            raise ValueError(f"{lk_path}.{LIK_MIN_WET_BANDS} must be >= 0")
    return params


def _read_resampling_ess_threshold_ratio(project_dir: Path) -> float | None:
    """Return configured resampling ESS threshold ratio for diagnostics."""
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    da_cfg_raw = cfg.get("data_assimilation")
    da_cfg = {} if da_cfg_raw is None else require_mapping(da_cfg_raw, path="project.data_assimilation")
    resampling_raw = da_cfg.get(RESAMPLING_BLOCK)
    resampling_path = f"project.data_assimilation.{RESAMPLING_BLOCK}"
    if resampling_raw is None:
        resampling_raw = cfg.get(RESAMPLING_BLOCK)
        resampling_path = f"project.{RESAMPLING_BLOCK}"
    if resampling_raw is None:
        return None
    resampling = require_mapping(resampling_raw, path=resampling_path)
    ratio_raw = resampling.get(RESAMPLING_ESS_THRESHOLD_RATIO)
    if ratio_raw is not None:
        ratio = _coerce_float(ratio_raw, path=f"{resampling_path}.{RESAMPLING_ESS_THRESHOLD_RATIO}")
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"{resampling_path}.{RESAMPLING_ESS_THRESHOLD_RATIO} must be within [0, 1]")
        return ratio
    threshold_raw = resampling.get(RESAMPLING_ESS_THRESHOLD)
    if threshold_raw is None:
        return None
    threshold = _coerce_float(threshold_raw, path=f"{resampling_path}.{RESAMPLING_ESS_THRESHOLD}")
    if 0.0 < threshold <= 1.0:
        return threshold
    return None


def _read_obs(csv_path: Path, value_col: str, *, uncertainty_metric: str | None = None) -> dict:
    """Read observation CSV; expect at least the given value column.

    Optional columns: 'n_valid', 'cloud_fraction'.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Observation CSV has no rows: {csv_path}")
    row = df.iloc[0]
    out = {value_col: float(row[value_col]) if value_col in row else None}
    if out[value_col] is None:
        raise ValueError(f"Observation CSV missing '{value_col}' column: {csv_path}")
    out["n_valid"] = int(row["n_valid"]) if "n_valid" in row and not pd.isna(row["n_valid"]) else None
    out["cloud_fraction"] = float(row["cloud_fraction"]) if "cloud_fraction" in row and not pd.isna(row["cloud_fraction"]) else 0.0
    if uncertainty_metric is not None:
        if uncertainty_metric not in row or pd.isna(row[uncertainty_metric]):
            raise ValueError(
                f"Observation CSV missing required uncertainty metric '{uncertainty_metric}': {csv_path}"
            )
        out[uncertainty_metric] = float(row[uncertainty_metric])
    return out


def _compute_sigma(
    *,
    obs: dict,
    y: float,
    prm: LikelihoodParams,
    sigma_mode: str,
    uncertainty_metric: str | None,
    obs_path: Path,
) -> float:
    if sigma_mode == "formula":
        return compute_obs_sigma(
            y,
            obs.get("n_valid"),
            float(obs.get("cloud_fraction", 0.0)),
            use_binomial=prm.use_binomial,
            sigma_floor=prm.sigma_floor,
            sigma_cloud_scale=prm.sigma_cloud_scale,
            min_sigma=prm.min_sigma,
            obs_sigma=prm.obs_sigma,
        )
    if sigma_mode != "uncertainty_layer":
        raise ValueError(f"Unsupported sigma_mode: {sigma_mode!r}")
    if uncertainty_metric is None:
        raise ValueError("uncertainty_metric must be configured for sigma_mode='uncertainty_layer'")
    unc_raw = obs.get(uncertainty_metric)
    if unc_raw is None:
        raise ValueError(
            f"Missing uncertainty metric '{uncertainty_metric}' required by sigma_mode='uncertainty_layer' in {obs_path}"
        )
    unc = float(unc_raw)
    if not np.isfinite(unc):
        raise ValueError(f"Uncertainty metric '{uncertainty_metric}' is not finite in {obs_path}")
    if unc < 0.0 or unc > 100.0:
        raise ValueError(
            f"Uncertainty metric '{uncertainty_metric}' out of [0,100] in {obs_path}: {unc}"
        )
    return max(float(prm.min_sigma), unc / 100.0)


def assimilate_fraction_for_date(
    *,
    project_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    obs_csv: Optional[Path] = None,
    value_col: str,
    observable: str,
    obs_candidates: Sequence[Path],
    model_eval: Callable[[Path, Path, datetime, ObservationSupportMask], FractionModelEvaluation],
    sigma_mode: str = "formula",
    uncertainty_metric: str | None = None,
) -> pd.DataFrame:
    """Generic fraction assimilation for one observable/date.

    Returns a DataFrame with columns:
    member_id, value_model, value_obs, residual, sigma, log_weight, weight
    """
    lk = _read_likelihood_from_project(project_dir, observable)

    candidates = list(obs_candidates)
    if obs_csv is not None:
        obs_path = obs_csv
    else:
        obs_path = next((p for p in candidates if p.exists()), None)
        if obs_path is None:
            missing = ", ".join(p.name for p in candidates) or "<none>"
            raise FileNotFoundError(
                f"Observation CSV not found for {observable} at {date.date()}: "
                f"expected one of [{missing}] under {step_dir / OBS_DIR_NAME}"
            )
    obs = _read_obs(
        obs_path,
        value_col,
        uncertainty_metric=(uncertainty_metric if sigma_mode == "uncertainty_layer" else None),
    )
    y = float(obs[value_col])
    setup_dir = infer_setup_dir_from_project(project_dir)
    lc_cfg = resolve_landcover_mask(setup_dir, project_dir)
    support_info = load_observation_support_mask(
        setup_dir=setup_dir,
        project_dir=project_dir,
        obs_csv=obs_path,
        observable=observable,
        landcover_cfg=lc_cfg,
    )
    support_gate_triggered = support_info.coverage_ratio < float(lk.min_support_coverage_ratio)
    support_gate_reason = (
        f"obs_support_coverage_ratio<{lk.min_support_coverage_ratio:.4f}"
        if support_gate_triggered
        else ""
    )
    sigma = (
        float("nan")
        if support_gate_triggered
        else _compute_sigma(
            obs=obs,
            y=y,
            prm=lk,
            sigma_mode=sigma_mode,
            uncertainty_metric=uncertainty_metric,
            obs_path=obs_path,
        )
    )

    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{ensemble}")

    rows: list[dict] = []
    for m in members:
        results = default_results_dir(m)
        model = model_eval(results, aoi, date, support_info)
        r = y - float(model.value_model)
        rows.append({
            "member_id": m.name,
            "value_model": float(model.value_model),
            "value_model_full_roi": float(model.value_model_full_roi),
            "value_model_obs_support": float(model.value_model_obs_support),
            "value_obs": y,
            "residual": r,
            "full_roi_n_valid": model.full_roi_n_valid,
            "obs_support_n_valid": model.obs_support_n_valid,
            "obs_support_coverage_ratio": support_info.coverage_ratio,
            "min_support_coverage_ratio": float(lk.min_support_coverage_ratio),
            "support_gate_triggered": bool(support_gate_triggered),
            "support_gate_reason": support_gate_reason,
        })

    df = pd.DataFrame(rows)
    df["sigma"] = sigma
    if support_gate_triggered:
        logL = np.zeros(len(df), dtype=float)
        w = np.full(len(df), 1.0 / float(len(df)), dtype=float)
    else:
        logL = gaussian_logpdf(df["residual"].to_numpy(), sigma)
        w = normalize_log_weights(logL)
    df["log_weight"] = logL
    df["weight"] = w
    ess = effective_sample_size(w)
    logger.info(
        "{} Assimilation | date={} members={} sigma={} support={:.3f} ESS={:.1f} gate={}",
        observable,
        date.strftime("%Y-%m-%d"),
        len(rows),
        "nan" if support_gate_triggered else f"{sigma:.3f}",
        support_info.coverage_ratio,
        ess,
        support_gate_triggered,
    )
    return df


def assimilate_scf_for_date(
    *,
    setup_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    obs_csv: Optional[Path] = None,
    product: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper: SCF-specific assimilation for one date."""
    project_dir = infer_project_dir(step_dir)
    method, variable, hofx_params = load_hofx_from_project(project_dir)
    lc_cfg = landcover_cfg or resolve_landcover_mask(setup_dir, project_dir)
    unc_cfg = _read_scf_uncertainty_assimilation_config(project_dir)
    prod_tag = str(product).strip().upper() if product else resolve_obs_product_tag("scf", setup_dir=setup_dir, project_dir=project_dir)
    if unc_cfg.enabled:
        # Deterministic tagged-path selection avoids accidental fallback to stale untagged files.
        obs_candidates = [
            build_obs_csv_path(
                step_dir=step_dir,
                variable="scf",
                date=date,
                product=prod_tag,
                include_product_tag=True,
            )
        ]
    else:
        obs_candidates = build_obs_candidate_paths(
            step_dir=step_dir,
            variable="scf",
            date=date,
            product=prod_tag,
        )

    def _model_eval(
        results_dir: Path,
        aoi_path: Path,
        dt: datetime,
        support_info: ObservationSupportMask,
    ) -> FractionModelEvaluation:
        out = compute_model_scf(
            setup_dir=setup_dir,
            project_dir=project_dir,
            results_dir=results_dir,
            aoi_path=aoi_path,
            landcover_cfg=lc_cfg,
            date=dt,
            variable=variable,  # type: ignore[arg-type]
            method=("logistic" if method == "logistic" else "depth_threshold"),  # type: ignore[arg-type]
            params=hofx_params,
            support_mask=support_info.mask,
        )
        return FractionModelEvaluation(
            value_model=float(out["scf"]),
            value_model_full_roi=float(out["scf_full_roi"]),
            value_model_obs_support=float(out["scf"]),
            full_roi_n_valid=int(out["n_valid_full_roi"]),
            obs_support_n_valid=int(out["n_valid"]),
        )

    df = assimilate_fraction_for_date(
        project_dir=project_dir,
        step_dir=step_dir,
        ensemble=ensemble,
        date=date,
        aoi=aoi,
        obs_csv=obs_csv,
        value_col="scf",
        observable="scf",
        obs_candidates=obs_candidates,
        model_eval=_model_eval,
        sigma_mode=(unc_cfg.sigma_mode if unc_cfg.enabled else "formula"),
        uncertainty_metric=(unc_cfg.aggregate_metric if unc_cfg.sigma_mode == "uncertainty_layer" else None),
    )
    df["scf_model"] = df["value_model"]
    df["scf_obs"] = df["value_obs"]
    return df


def assimilate_wet_snow_for_date(
    *,
    setup_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    obs_csv: Optional[Path] = None,
    product: str | None = None,
) -> pd.DataFrame:
    """Wet-snow assimilation for one date (Sentinel-1 AOI fraction)."""
    project_dir = infer_project_dir(step_dir)
    lc_cfg = landcover_cfg or resolve_landcover_mask(setup_dir, project_dir)
    unc_cfg = _read_wet_snow_uncertainty_assimilation_config(project_dir)
    prod_tag = str(product).strip().upper() if product else resolve_obs_product_tag("wet_snow", setup_dir=setup_dir, project_dir=project_dir)
    if unc_cfg.enabled:
        obs_candidates = [
            build_obs_csv_path(
                step_dir=step_dir,
                variable="wet_snow",
                date=date,
                product=prod_tag,
                include_product_tag=True,
            )
        ]
    else:
        obs_candidates = build_obs_candidate_paths(
            step_dir=step_dir,
            variable="wet_snow",
            date=date,
            product=prod_tag,
        )

    def _model_eval(
        results_dir: Path,
        aoi_path: Path,
        dt: datetime,
        support_info: ObservationSupportMask,
    ) -> FractionModelEvaluation:
        out = compute_model_wet_snow_fraction(
            setup_dir=setup_dir,
            project_dir=project_dir,
            results_dir=results_dir,
            aoi_path=aoi_path,
            landcover_cfg=lc_cfg,
            date=dt,
            support_mask=support_info.mask,
        )
        return FractionModelEvaluation(
            value_model=float(out["wet_fraction"]),
            value_model_full_roi=float(out["wet_fraction_full_roi"]),
            value_model_obs_support=float(out["wet_fraction"]),
            full_roi_n_valid=int(out["n_valid_full_roi"]),
            obs_support_n_valid=int(out["n_valid"]),
        )

    df = assimilate_fraction_for_date(
        project_dir=project_dir,
        step_dir=step_dir,
        ensemble=ensemble,
        date=date,
        aoi=aoi,
        obs_csv=obs_csv,
        value_col="wet_snow_fraction",
        observable="wet_snow",
        obs_candidates=obs_candidates,
        model_eval=_model_eval,
        sigma_mode=(unc_cfg.sigma_mode if unc_cfg.enabled else "formula"),
        uncertainty_metric=(unc_cfg.aggregate_metric if unc_cfg.sigma_mode == "uncertainty_layer" else None),
    )
    df["wet_snow_model"] = df["value_model"]
    df["wet_snow_obs"] = df["value_obs"]
    return df


def assimilate_wet_snow_line_for_date(
    *,
    setup_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    obs_csv: Optional[Path] = None,
    product: str | None = None,
) -> pd.DataFrame:
    """Wet-snow-line assimilation for one date using an elevation-space Gaussian likelihood."""

    project_dir = infer_project_dir(step_dir)
    lc_cfg = landcover_cfg or resolve_landcover_mask(setup_dir, project_dir)
    lk = _read_likelihood_from_project(project_dir, "wet_snow_line")
    if lk.use_binomial:
        raise ValueError("project.data_assimilation.likelihood.wet_snow_line.use_binomial must be false")
    prod_tag = str(product).strip().upper() if product else resolve_obs_product_tag(
        "wet_snow_line",
        setup_dir=setup_dir,
        project_dir=project_dir,
    )
    obs_candidates = build_obs_candidate_paths(
        step_dir=step_dir,
        variable="wet_snow_line",
        date=date,
        product=prod_tag,
    )
    obs_path = Path(obs_csv) if obs_csv is not None else next((p for p in obs_candidates if p.exists()), None)
    if obs_path is None:
        missing = ", ".join(p.name for p in obs_candidates) or "<none>"
        raise FileNotFoundError(
            f"Observation CSV not found for wet_snow_line at {date.date()}: "
            f"expected one of [{missing}] under {step_dir / OBS_DIR_NAME}"
        )
    obs_df = pd.read_csv(obs_path)
    if obs_df.empty:
        raise ValueError(f"Observation CSV has no rows: {obs_path}")
    obs_row = obs_df.iloc[0]
    if "wet_snow_line" not in obs_row or pd.isna(obs_row["wet_snow_line"]):
        y = float("nan")
    else:
        y = float(obs_row["wet_snow_line"])
    support_info = load_observation_support_mask(
        setup_dir=setup_dir,
        project_dir=project_dir,
        obs_csv=obs_path,
        observable="wet_snow_line",
        landcover_cfg=lc_cfg,
    )
    support_gate_triggered = support_info.coverage_ratio < float(lk.min_support_coverage_ratio)
    support_gate_reason = (
        f"obs_support_coverage_ratio<{lk.min_support_coverage_ratio:.4f}"
        if support_gate_triggered
        else ""
    )
    obs_n_wet = (
        int(obs_row["wet_snow_line_n_wet"])
        if "wet_snow_line_n_wet" in obs_row and not pd.isna(obs_row["wet_snow_line_n_wet"])
        else 0
    )
    obs_wet_bands = (
        int(obs_row["wet_snow_line_wet_bands"])
        if "wet_snow_line_wet_bands" in obs_row and not pd.isna(obs_row["wet_snow_line_wet_bands"])
        else 0
    )
    raw_obs_gate_reason = obs_row.get("wet_snow_line_gate_reason", "")
    if pd.isna(raw_obs_gate_reason):
        obs_gate_reason = ""
    else:
        obs_gate_reason = str(raw_obs_gate_reason or "").strip()
    if not np.isfinite(y) and not obs_gate_reason:
        obs_gate_reason = "no_crossing_fraction"
    wet_information_gate_triggered = (
        bool(obs_gate_reason)
        or obs_n_wet < int(lk.min_wet_pixels_total)
        or obs_wet_bands < int(lk.min_wet_bands)
    )
    if obs_gate_reason:
        wet_information_gate_reason = obs_gate_reason
    elif obs_n_wet < int(lk.min_wet_pixels_total):
        wet_information_gate_reason = f"wet_pixels<{lk.min_wet_pixels_total}"
    elif obs_wet_bands < int(lk.min_wet_bands):
        wet_information_gate_reason = f"wet_bands<{lk.min_wet_bands}"
    else:
        wet_information_gate_reason = ""
    gate_triggered = support_gate_triggered or wet_information_gate_triggered
    sigma = (
        float("nan")
        if gate_triggered
        else max(float(lk.obs_sigma), float(lk.sigma_floor), float(lk.min_sigma))
    )

    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{ensemble}")

    rows: list[dict[str, object]] = []
    for m in members:
        results = default_results_dir(m)
        model = compute_model_wet_snow_line(
            setup_dir=setup_dir,
            project_dir=project_dir,
            results_dir=results,
            aoi_path=aoi,
            landcover_cfg=lc_cfg,
            date=date,
            support_mask=support_info.mask,
        )
        value_model = model["wet_snow_line"]
        value_model_full_roi = model["wet_snow_line_full_roi"]
        residual = y - float(value_model) if (value_model is not None and np.isfinite(y)) else np.nan
        rows.append(
            {
                "member_id": m.name,
                "value_model": value_model,
                "value_model_full_roi": value_model_full_roi,
                "value_model_obs_support": value_model,
                "value_obs": y,
                "residual": residual,
                "full_roi_n_valid": model["n_valid_full_roi"],
                "obs_support_n_valid": model["n_valid"],
                "obs_support_coverage_ratio": support_info.coverage_ratio,
                "min_support_coverage_ratio": float(lk.min_support_coverage_ratio),
                "min_model_finite_fraction": float(lk.min_model_finite_fraction),
                "support_gate_triggered": bool(support_gate_triggered),
                "support_gate_reason": support_gate_reason,
                "wet_information_gate_triggered": bool(wet_information_gate_triggered),
                "wet_information_gate_reason": wet_information_gate_reason,
                "value_model_gate_reason": str(model.get("wet_snow_line_gate_reason", "") or ""),
                "value_model_wet_bands": model.get("wet_bands"),
            }
        )

    df = pd.DataFrame(rows)
    model_finite_mask = pd.to_numeric(df["value_model"], errors="coerce").notna()
    model_missing_mask = ~model_finite_mask
    model_member_count = int(len(df))
    model_finite_member_count = int(model_finite_mask.sum())
    model_finite_fraction = (
        float(model_finite_member_count) / float(model_member_count) if model_member_count else 0.0
    )
    model_gate_triggered = bool(
        model_finite_member_count == 0
        or model_finite_fraction < float(lk.min_model_finite_fraction)
    )
    model_gate_reasons = sorted(
        {
            str(reason).strip()
            for reason in df.loc[model_missing_mask, "value_model_gate_reason"].tolist()
            if str(reason).strip()
        }
    )
    if model_gate_triggered:
        if model_finite_member_count == 0:
            finite_support_reason = "model_no_finite_wet_snow_line"
        else:
            finite_support_reason = f"model_finite_fraction<{lk.min_model_finite_fraction:.4f}"
        model_gate_reason = ";".join([finite_support_reason, *model_gate_reasons])
    else:
        model_gate_reason = ""
    df["model_gate_triggered"] = model_gate_triggered
    df["model_gate_reason"] = model_gate_reason
    df["model_finite_member_count"] = model_finite_member_count
    df["model_member_count"] = model_member_count
    df["model_finite_fraction"] = model_finite_fraction
    df["model_finite_fraction_threshold"] = float(lk.min_model_finite_fraction)
    df["sigma"] = sigma
    gate_triggered = support_gate_triggered or wet_information_gate_triggered or model_gate_triggered
    if gate_triggered:
        df["sigma"] = float("nan")
        logL = np.zeros(len(df), dtype=float)
        w = np.full(len(df), 1.0 / float(len(df)), dtype=float)
    else:
        residuals = pd.to_numeric(df["residual"], errors="coerce").to_numpy(dtype=float)
        logL = gaussian_logpdf(residuals, sigma)
        logL[~np.isfinite(logL)] = -1.0e12
        w = normalize_log_weights(logL)
    df["log_weight"] = logL
    df["weight"] = w
    df["wet_snow_line_model"] = df["value_model"]
    df["wet_snow_line_obs"] = df["value_obs"]
    ess = effective_sample_size(w)
    df["ess"] = ess
    ess_threshold_ratio = _read_resampling_ess_threshold_ratio(project_dir)
    df["ess_threshold_ratio"] = np.nan if ess_threshold_ratio is None else float(ess_threshold_ratio)
    df["ess_below_threshold"] = False
    if ess_threshold_ratio is not None:
        ess_threshold = float(ess_threshold_ratio) * float(len(df))
        ess_below_threshold = bool(ess < ess_threshold)
        df["ess_below_threshold"] = ess_below_threshold
        if ess_below_threshold:
            logger.warning(
                "wet_snow_line Assimilation | date={} ESS={:.1f} below configured resampling threshold {:.1f} "
                "(ratio={:.3f}); resampling stage handles resampling",
                date.strftime("%Y-%m-%d"),
                ess,
                ess_threshold,
                ess_threshold_ratio,
            )
    logger.info(
        "wet_snow_line Assimilation | date={} members={} finite={}/{} sigma={} support={:.3f} ESS={:.1f} gate={}",
        date.strftime("%Y-%m-%d"),
        len(rows),
        model_finite_member_count,
        model_member_count,
        "nan" if gate_triggered else f"{sigma:.1f}",
        support_info.coverage_ratio,
        ess,
        gate_triggered,
    )
    return df


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: compute Gaussian weights for SCF on one date.

    Example
    -------
    oa-da-assimilate-scf \
      --setup-dir C:/.../examples/test-project \
      --step-dir C:/.../projects/project_2017-2018/steps/step_00_init \
      --ensemble prior \
      --date 2018-02-15 \
      --aoi C:/.../env/GMBA_Inventory_L8_15422.gpkg
    """
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-assimilate-scf", description="Compute Gaussian weights for SCF vs model H(x)")
    p.add_argument("--setup-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--date", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--aoi", "--roi", dest="aoi", required=True, type=Path, help="ROI vector (single feature)")
    p.add_argument("--product", type=str, help="Product code used in obs filename (default: project obs.snowcover.product_tag)")
    p.add_argument("--obs-csv", type=Path, help="Optional path to obs_scf_*.csv; default: <step>/obs")
    p.add_argument("--output", type=Path, help="Optional output CSV path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    # Logger
    configure_cli_logger(args.log_level)

    # Run
    try:
        step_dir = Path(args.step_dir)
        project_dir = infer_project_dir(step_dir)
        setup_dir = infer_setup_dir_from_project(project_dir)
        if setup_dir.resolve() != Path(args.setup_dir).resolve():
            logger.warning(
                "Step {} belongs to setup {}; overriding provided setup {}",
                step_dir,
                setup_dir,
                args.setup_dir,
            )
        dt = datetime.strptime(args.date, "%Y-%m-%d")
        lc_cfg = resolve_landcover_mask(setup_dir, project_dir)
        df = assimilate_scf_for_date(
            setup_dir=setup_dir,
            step_dir=step_dir,
            ensemble=str(args.ensemble),
            date=dt,
            aoi=Path(args.aoi),
            landcover_cfg=lc_cfg,
            obs_csv=Path(args.obs_csv) if args.obs_csv else None,
            product=str(args.product) if args.product else None,
        )
    except Exception as e:
        logger.error(f"Assimilation failed: {e}")
        return 1

    out = args.output
    if out is None:
        out_dir = Path(args.step_dir) / "assim"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / weights_csv_name("scf", dt)
    df.to_csv(out, index=False)
    logger.info("Wrote weights: {}", out)
    return 0


def cli_main_wet_snow(argv: list[str] | None = None) -> int:
    """CLI: compute Gaussian weights for wet-snow fractions on one date."""
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-assimilate-wet-snow", description="Compute Gaussian weights for wet snow vs model H(x)")
    p.add_argument("--setup-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--date", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--aoi", required=True, type=Path, help="AOI vector (single feature)")
    p.add_argument("--obs-csv", type=Path, help="Optional path to obs_wet_snow_*.csv; default: <step>/obs")
    p.add_argument("--output", type=Path, help="Optional output CSV path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_cli_logger(args.log_level)

    try:
        step_dir = Path(args.step_dir)
        project_dir = infer_project_dir(step_dir)
        setup_dir = infer_setup_dir_from_project(project_dir)
        if setup_dir.resolve() != Path(args.setup_dir).resolve():
            logger.warning(
                "Step {} belongs to setup {}; overriding provided setup {}",
                step_dir,
                setup_dir,
                args.setup_dir,
            )
        dt = datetime.strptime(args.date, "%Y-%m-%d")
        lc_cfg = resolve_landcover_mask(setup_dir, project_dir)
        df = assimilate_wet_snow_for_date(
            setup_dir=setup_dir,
            step_dir=step_dir,
            ensemble=str(args.ensemble),
            date=dt,
            aoi=Path(args.aoi),
            landcover_cfg=lc_cfg,
            obs_csv=Path(args.obs_csv) if args.obs_csv else None,
        )
    except Exception as e:
        logger.error(f"Wet-snow assimilation failed: {e}")
        return 1

    out = args.output
    if out is None:
        out_dir = Path(args.step_dir) / "assim"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / weights_csv_name("wet_snow", dt)
    df.to_csv(out, index=False)
    logger.info("Wrote wet-snow weights: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

"""openamundsen_da.methods.pf.assimilate_scf

Gaussian-likelihood weighting for SCF observations against model-derived SCF
per ensemble member (particle filter step without resampling).

Reads one observation for a date/region and computes model SCF H(x) for all
members, then evaluates log-likelihoods and normalized weights. Outputs a CSV
with per-member weights and summary stats (ESS).

Configuration
- H(x) configuration is read from project.yml (data_assimilation.h_of_x) and is required.
- Likelihood configuration is read from project.yml (likelihood block).
  Falls back to sensible defaults if missing.

Logging
- Uses loguru with a green timestamp format defined in constants.LOGURU_FORMAT.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import numpy as np
from loguru import logger

from openamundsen_da.core.constants import (
    LOGURU_FORMAT,
    LIKELIHOOD_BLOCK,
    LIK_OBS_SIGMA,
    LIK_USE_BINOMIAL,
    LIK_SIGMA_FLOOR,
    LIK_SIGMA_CLOUD_SCALE,
    LIK_MIN_SIGMA,
    OBS_DIR_NAME,
)
from openamundsen_da.io.paths import list_member_dirs, default_results_dir, find_project_yaml
from openamundsen_da.methods.h_of_x.model_scf import compute_model_scf, SCFParams, load_hofx_from_project
from openamundsen_da.methods.wet_snow.area import compute_model_wet_snow_fraction
from openamundsen_da.util.stats import gaussian_logpdf, normalize_log_weights, effective_sample_size, compute_obs_sigma
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.util.glacier_mask import resolve_glacier_mask


@dataclass
class LikelihoodParams:
    obs_sigma: float = 0.10
    use_binomial: bool = True
    sigma_floor: float = 0.05
    sigma_cloud_scale: float = 0.10
    min_sigma: float = 0.03


def _read_likelihood_from_project(project_dir: Path, observable: str) -> LikelihoodParams:
    """Read likelihood settings from project.yml for a given observable if available.

    The config may either be flat under ``likelihood`` (legacy) or nested
    as ``likelihood.<observable>`` (preferred for multiple observables).
    """
    try:
        proj = find_project_yaml(project_dir)
        cfg = _read_yaml_file(proj) or {}
        lk_root = cfg.get(LIKELIHOOD_BLOCK) or {}
        lk = lk_root.get(observable, lk_root) if isinstance(lk_root, dict) else {}
        p = LikelihoodParams()
        if LIK_OBS_SIGMA in lk:
            p.obs_sigma = float(lk[LIK_OBS_SIGMA])
        if LIK_USE_BINOMIAL in lk:
            p.use_binomial = bool(lk[LIK_USE_BINOMIAL])
        if LIK_SIGMA_FLOOR in lk:
            p.sigma_floor = float(lk[LIK_SIGMA_FLOOR])
        if LIK_SIGMA_CLOUD_SCALE in lk:
            p.sigma_cloud_scale = float(lk[LIK_SIGMA_CLOUD_SCALE])
        if LIK_MIN_SIGMA in lk:
            p.min_sigma = float(lk[LIK_MIN_SIGMA])
        return p
    except Exception:
        return LikelihoodParams()


def _read_obs(csv_path: Path, value_col: str) -> dict:
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
    return out


def _compute_sigma(y: float, n_valid: Optional[int], cloud_fraction: float, prm: LikelihoodParams) -> float:
    return compute_obs_sigma(
        y,
        n_valid,
        cloud_fraction,
        use_binomial=prm.use_binomial,
        sigma_floor=prm.sigma_floor,
        sigma_cloud_scale=prm.sigma_cloud_scale,
        min_sigma=prm.min_sigma,
        obs_sigma=prm.obs_sigma,
    )


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
    obs_pattern: str,
    model_eval: Callable[[Path, Path, datetime], float],
) -> pd.DataFrame:
    """Generic fraction assimilation for one observable/date.

    Returns a DataFrame with columns:
    member_id, value_model, value_obs, residual, sigma, log_weight, weight
    """
    lk = _read_likelihood_from_project(project_dir, observable)

    # Read observation
    if obs_csv is not None:
        obs_path = obs_csv
    else:
        obs_dir = step_dir / OBS_DIR_NAME
        patt = obs_pattern.format(yyyymmdd=date.strftime("%Y%m%d"))
        obs_path = obs_dir / patt
        if not obs_path.exists():
            raise FileNotFoundError(
                f"Observation CSV not found for {observable} at {date.date()}: expected {obs_path.name} under {obs_dir}"
            )
    obs = _read_obs(obs_path, value_col)
    y = float(obs[value_col])
    sigma = _compute_sigma(y, obs.get("n_valid"), float(obs.get("cloud_fraction", 0.0)), lk)

    # Gather member result dirs
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{ensemble}")

    # Compute model value per member
    rows: list[dict] = []
    for m in members:
        results = default_results_dir(m)
        model_val = float(model_eval(results, aoi, date))
        r = y - model_val
        rows.append({
            "member_id": m.name,
            "value_model": model_val,
            "value_obs": y,
            "residual": r,
        })

    df = pd.DataFrame(rows)
    # Likelihood and weights
    logL = gaussian_logpdf(df["residual"].to_numpy(), sigma)
    w = normalize_log_weights(logL)
    df["sigma"] = sigma
    df["log_weight"] = logL
    df["weight"] = w
    # Summary
    ess = effective_sample_size(w)
    logger.info(
        "{} Assimilation | date={} members={} sigma={:.3f} ESS={:.1f}",
        observable,
        date.strftime("%Y-%m-%d"),
        len(rows),
        sigma,
        ess,
    )
    return df


def assimilate_scf_for_date(
    *,
    project_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    glacier_path: Path | None = None,
    obs_csv: Optional[Path] = None,
    product: str = "MOD10A1",
) -> pd.DataFrame:
    """Backward-compatible wrapper: SCF-specific assimilation for one date."""
    method, variable, hofx_params = load_hofx_from_project(project_dir)

    def _model_eval(results_dir: Path, aoi_path: Path, dt: datetime) -> float:
        out = compute_model_scf(
            results_dir=results_dir,
            aoi_path=aoi_path,
            glacier_path=glacier_path,
            date=dt,
            variable=variable,  # type: ignore[arg-type]
            method=("logistic" if method == "logistic" else "depth_threshold"),  # type: ignore[arg-type]
            params=hofx_params,
        )
        return float(out["scf"])

    df = assimilate_fraction_for_date(
        project_dir=project_dir,
        step_dir=step_dir,
        ensemble=ensemble,
        date=date,
        aoi=aoi,
        obs_csv=obs_csv,
        value_col="scf",
        observable="scf",
        obs_pattern=f"obs_scf_{str(product).upper()}_" + "{yyyymmdd}.csv",
        model_eval=_model_eval,
    )
    # Preserve SCF-specific column names for downstream tools.
    df = df.rename(columns={"value_model": "scf_model", "value_obs": "scf_obs"})
    return df


def assimilate_wet_snow_for_date(
    *,
    project_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    glacier_path: Path | None = None,
    obs_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Wet-snow assimilation for one date (Sentinel-1 AOI fraction)."""

    def _model_eval(results_dir: Path, aoi_path: Path, dt: datetime) -> float:
        out = compute_model_wet_snow_fraction(
            results_dir=results_dir,
            aoi_path=aoi_path,
            glacier_path=glacier_path,
            date=dt,
        )
        return float(out["wet_fraction"])

    df = assimilate_fraction_for_date(
        project_dir=project_dir,
        step_dir=step_dir,
        ensemble=ensemble,
        date=date,
        aoi=aoi,
        obs_csv=obs_csv,
        value_col="wet_snow_fraction",
        observable="wet_snow",
        obs_pattern="obs_wet_snow_S1_{yyyymmdd}.csv",
        model_eval=_model_eval,
    )
    df = df.rename(columns={"value_model": "wet_snow_model", "value_obs": "wet_snow_obs"})
    return df


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: compute Gaussian weights for SCF on one date.

    Example
    -------
    oa-da-assimilate-scf \
      --project-dir C:/.../examples/test-project \
      --step-dir C:/.../propagation/season_2017-2018/step_00_init \
      --ensemble prior \
      --date 2018-02-15 \
      --aoi C:/.../env/GMBA_Inventory_L8_15422.gpkg
    """
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-assimilate-scf", description="Compute Gaussian weights for SCF vs model H(x)")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--date", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--aoi", "--roi", dest="aoi", required=True, type=Path, help="ROI vector (single feature)")
    p.add_argument("--product", type=str, default="MOD10A1", help="Product code used in obs filename (default: MOD10A1)")
    p.add_argument("--obs-csv", type=Path, help="Optional path to obs_scf_*.csv; default: <step>/obs")
    p.add_argument("--output", type=Path, help="Optional output CSV path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    # Logger
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    # Run
    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
        glacier_cfg = resolve_glacier_mask(Path(args.project_dir))
        glacier_path = glacier_cfg.path if glacier_cfg.enabled else None
        df = assimilate_scf_for_date(
            project_dir=Path(args.project_dir),
            step_dir=Path(args.step_dir),
            ensemble=str(args.ensemble),
            date=dt,
            aoi=Path(args.aoi),
            glacier_path=glacier_path,
            obs_csv=Path(args.obs_csv) if args.obs_csv else None,
            product=str(args.product or "MOD10A1"),
        )
    except Exception as e:
        logger.error(f"Assimilation failed: {e}")
        return 1

    out = args.output
    if out is None:
        out_dir = Path(args.step_dir) / "assim"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"weights_scf_{dt.strftime('%Y%m%d')}.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote weights: {}", out)
    return 0


def cli_main_wet_snow(argv: list[str] | None = None) -> int:
    """CLI: compute Gaussian weights for wet-snow fractions on one date."""
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-assimilate-wet-snow", description="Compute Gaussian weights for wet snow vs model H(x)")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--date", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--aoi", required=True, type=Path, help="AOI vector (single feature)")
    p.add_argument("--obs-csv", type=Path, help="Optional path to obs_wet_snow_*.csv; default: <step>/obs")
    p.add_argument("--output", type=Path, help="Optional output CSV path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
        glacier_cfg = resolve_glacier_mask(Path(args.project_dir))
        glacier_path = glacier_cfg.path if glacier_cfg.enabled else None
        df = assimilate_wet_snow_for_date(
            project_dir=Path(args.project_dir),
            step_dir=Path(args.step_dir),
            ensemble=str(args.ensemble),
            date=dt,
            aoi=Path(args.aoi),
            glacier_path=glacier_path,
            obs_csv=Path(args.obs_csv) if args.obs_csv else None,
        )
    except Exception as e:
        logger.error(f"Wet-snow assimilation failed: {e}")
        return 1

    out = args.output
    if out is None:
        out_dir = Path(args.step_dir) / "assim"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"weights_wet_snow_{dt.strftime('%Y%m%d')}.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote wet-snow weights: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

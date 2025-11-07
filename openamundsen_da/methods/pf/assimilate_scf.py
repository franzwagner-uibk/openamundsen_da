"""openamundsen_da.methods.pf.assimilate_scf

Gaussian-likelihood weighting for SCF observations against model-derived SCF
per ensemble member (particle filter step without resampling).

Reads one observation for a date/region and computes model SCF H(x) for all
members, then evaluates log-likelihoods and normalized weights. Outputs a CSV
with per-member weights and summary stats (ESS).

Configuration
- Reads optional defaults from the step YAML (h_of_x block) and project.yml
  (likelihood block). Falls back to sensible defaults if missing.

Logging
- Uses loguru with a green timestamp format defined in constants.LOGURU_FORMAT.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

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
    HOFX_BLOCK,
    HOFX_METHOD,
    HOFX_VARIABLE,
    HOFX_PARAMS,
    HOFX_PARAM_H0,
    HOFX_PARAM_K,
    OBS_DIR_NAME,
)
from openamundsen_da.io.paths import list_member_dirs, default_results_dir, find_step_yaml, find_project_yaml
from openamundsen_da.methods.h_of_x.model_scf import compute_model_scf, SCFParams
from openamundsen_da.util.stats import gaussian_logpdf, normalize_log_weights, effective_sample_size, compute_obs_sigma
from openamundsen_da.core.env import _read_yaml_file


@dataclass
class LikelihoodParams:
    obs_sigma: float = 0.10
    use_binomial: bool = True
    sigma_floor: float = 0.05
    sigma_cloud_scale: float = 0.10
    min_sigma: float = 0.03


def _read_step_hofx(step_dir: Path) -> tuple[str, str, SCFParams]:
    """Read H(x) defaults from step YAML if available."""
    try:
        import ruamel.yaml as _yaml
        yml = find_step_yaml(step_dir)
        y = _yaml.YAML(typ="safe")
        with Path(yml).open("r", encoding="utf-8") as f:
            cfg = y.load(f) or {}
        hofx = (cfg or {}).get(HOFX_BLOCK) or {}
        method = str(hofx.get(HOFX_METHOD, "depth_threshold"))
        variable = str(hofx.get(HOFX_VARIABLE, "hs"))
        params = SCFParams()
        p = (hofx.get(HOFX_PARAMS) or {})
        if HOFX_PARAM_H0 in p:
            params.h0 = float(p[HOFX_PARAM_H0])
        if HOFX_PARAM_K in p:
            params.k = float(p[HOFX_PARAM_K])
        return method, variable, params
    except Exception:
        return "depth_threshold", "hs", SCFParams()


def _read_likelihood_from_project(project_dir: Path) -> LikelihoodParams:
    """Read likelihood settings from project.yml if available."""
    try:
        proj = find_project_yaml(project_dir)
        cfg = _read_yaml_file(proj) or {}
        lk = (cfg.get(LIKELIHOOD_BLOCK) or {})
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


def _find_obs_csv(step_dir: Path, date: datetime) -> Optional[Path]:
    obs_dir = step_dir / OBS_DIR_NAME
    patt = f"obs_scf_MOD10A1_{date.strftime('%Y%m%d')}.csv"
    p = obs_dir / patt
    return p if p.exists() else None


def _read_obs(csv_path: Path) -> dict:
    """Read observation CSV; expect at least 'scf'.

    Optional columns: 'n_valid', 'cloud_fraction'.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Observation CSV has no rows: {csv_path}")
    row = df.iloc[0]
    out = {"scf": float(row["scf"]) if "scf" in row else None}
    if out["scf"] is None:
        raise ValueError(f"Observation CSV missing 'scf' column: {csv_path}")
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


def assimilate_scf_for_date(
    *,
    project_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    aoi: Path,
    obs_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute SCF weights for all members for a single date.

    Returns a DataFrame with columns:
    member_id, scf_model, scf_obs, residual, sigma, log_weight, weight
    """
    # Read config blocks
    method, variable, hofx_params = _read_step_hofx(step_dir)
    lk = _read_likelihood_from_project(project_dir)

    # Read observation
    obs_path = obs_csv or _find_obs_csv(step_dir, date)
    if obs_path is None:
        raise FileNotFoundError("Observation CSV not found; provide --obs-csv or generate via oa-da-scf")
    obs = _read_obs(obs_path)
    y = float(obs["scf"])
    sigma = _compute_sigma(y, obs.get("n_valid"), float(obs.get("cloud_fraction", 0.0)), lk)

    # Gather member result dirs
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{ensemble}")

    # Compute model SCF per member
    rows: list[dict] = []
    for m in members:
        results = default_results_dir(m)
        out = compute_model_scf(
            results_dir=results,
            aoi_path=aoi,
            date=date,
            variable=variable,  # type: ignore[arg-type]
            method=("logistic" if method == "logistic" else "depth_threshold"),  # type: ignore[arg-type]
            params=hofx_params,
        )
        r = y - float(out["scf"])
        rows.append({
            "member_id": out["member_id"],
            "scf_model": float(out["scf"]),
            "scf_obs": y,
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
    logger.info("SCF Assimilation | date={} members={} sigma={:.3f} ESS={:.1f}", date.strftime("%Y-%m-%d"), len(rows), sigma, ess)
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
    p.add_argument("--aoi", required=True, type=Path, help="AOI vector (single feature)")
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
        df = assimilate_scf_for_date(
            project_dir=Path(args.project_dir),
            step_dir=Path(args.step_dir),
            ensemble=str(args.ensemble),
            date=dt,
            aoi=Path(args.aoi),
            obs_csv=Path(args.obs_csv) if args.obs_csv else None,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

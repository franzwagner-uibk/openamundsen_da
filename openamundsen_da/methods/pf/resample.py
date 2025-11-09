"""openamundsen_da.methods.pf.resample

Systematic resampling of a single-date weights CSV to form a posterior ensemble.

Features
- Reads a weights CSV (from oa-da-assimilate-scf) with normalized weights.
- Computes ESS and, if below a threshold, draws N indices via systematic resampling.
- Materializes `<step>/ensembles/posterior/member_XXX` by copying (symlink if possible)
  from the selected source members under `<step>/ensembles/<source>/member_*`.
- Writes a compact manifest and indices CSV for traceability.
- If ESS >= threshold, optionally skip resampling and mirror source -> target.

Logging uses constants.LOGURU_FORMAT (green timestamp | level | message).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import (
    LOGURU_FORMAT,
    RESAMPLING_BLOCK,
    RESAMPLING_ALGORITHM,
    RESAMPLING_ESS_THRESHOLD,
    RESAMPLING_ESS_THRESHOLD_RATIO,
    DA_BLOCK,
    DA_RANDOM_SEED,
    MEMBER_PREFIX,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import list_member_dirs, find_project_yaml
from openamundsen_da.util.stats import effective_sample_size, systematic_resample


@dataclass(frozen=True)
class ResamplingConfig:
    algorithm: str = "systematic"
    ess_threshold: float = 0.0  # absolute; if 0, never skip (always resample)
    ess_threshold_ratio: float | None = None  # 0..1
    seed: Optional[int] = None


def _read_resampling_from_project(project_dir: Path) -> ResamplingConfig:
    """Read resampling defaults from project.yml if present.

    Falls back to DA_RANDOM_SEED for seed when available.
    """
    try:
        proj = find_project_yaml(project_dir)
        cfg = _read_yaml_file(proj) or {}
        # Prefer nested under data_assimilation.resampling; fallback to top-level resampling
        r = ((cfg.get(DA_BLOCK) or {}).get(RESAMPLING_BLOCK) or (cfg.get(RESAMPLING_BLOCK) or {}))
        seed = r.get("seed")
        if seed is None:
            seed = cfg.get("data_assimilation", {}).get("prior_forcing", {}).get(DA_RANDOM_SEED)
        algo = str(r.get(RESAMPLING_ALGORITHM, "systematic"))
        thr = r.get(RESAMPLING_ESS_THRESHOLD)
        thr_ratio = r.get(RESAMPLING_ESS_THRESHOLD_RATIO)
        # Interpret thresholds: if ratio key present, use it; else if 0<thr<=1 -> ratio
        ratio_val = None
        abs_val = 0.0
        if thr_ratio is not None:
            ratio_val = float(thr_ratio)
        if thr is not None:
            tv = float(thr)
            if 0.0 < tv <= 1.0 and ratio_val is None:
                ratio_val = tv
            else:
                abs_val = tv
        return ResamplingConfig(
            algorithm=algo,
            ess_threshold=abs_val,
            ess_threshold_ratio=ratio_val,
            seed=(int(seed) if seed is not None else None),
        )
    except Exception:
        return ResamplingConfig()


def _load_weights(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"weight"}
    if not needed.issubset(df.columns):
        raise ValueError("Weights CSV missing 'weight' column")
    # Optional member_id column is used to map onto member directories
    return df


def _mirror_or_resample(
    *,
    step_dir: Path,
    source_ensemble: str,
    target_ensemble: str,
    members_order: list[Path],
    draw_indices: Optional[np.ndarray],
    overwrite: bool,
    force_copy: bool,
    weights: Optional[np.ndarray] = None,
) -> list[tuple[str, str, float | None]]:
    """Create target ensemble by mirroring or resampling from source.

    Returns list of (target_member_id, source_member_id) pairs.
    """
    tgt_root = Path(step_dir) / "ensembles" / target_ensemble
    tgt_root.mkdir(parents=True, exist_ok=True)

    # Determine mapping from target index -> source member path
    if draw_indices is None:
        mapping = [members_order[i] for i in range(len(members_order))]
        w_map = list(weights) if weights is not None else [None] * len(members_order)
    else:
        mapping = [members_order[int(i)] for i in draw_indices]
        w_map = [float(weights[int(i)]) if weights is not None else None for i in draw_indices]

    pairs: list[tuple[str, str, float | None]] = []
    for k, (src_member, wv) in enumerate(zip(mapping, w_map), start=1):
        tgt_member = tgt_root / f"{MEMBER_PREFIX}{k:03d}"
        pairs.append((tgt_member.name, src_member.name, wv))

        if tgt_member.exists() and overwrite:
            shutil.rmtree(tgt_member, ignore_errors=True)
        if tgt_member.exists():
            # Keep existing unless overwrite requested
            continue

        # Create target member dir and populate by linking or copying
        tgt_member.mkdir(parents=True, exist_ok=True)
        for child in src_member.iterdir():
            dst = tgt_member / child.name
            if force_copy:
                if child.is_dir():
                    shutil.copytree(child, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, dst)
                continue
            try:
                if child.is_dir():
                    _symlink_dir(child, dst)
                else:
                    _symlink_file(child, dst)
            except Exception:
                # Symlink not available/allowed -> copy fallback
                if child.is_dir():
                    shutil.copytree(child, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, dst)
    return pairs


def _symlink_dir(src: Path, dst: Path) -> None:
    # On Windows, enable junctions for directories if possible
    if os.name == "nt":
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except OSError:
            pass
    os.symlink(src, dst, target_is_directory=True)


def _symlink_file(src: Path, dst: Path) -> None:
    os.symlink(src, dst)


def _write_manifest(
    *,
    out_dir: Path,
    weights_csv: Path,
    alg: str,
    ess: float,
    n: int,
    seed: Optional[int],
    skipped: bool,
    pairs: list[tuple[str, str, float | None]],
    ess_threshold: float,
) -> tuple[Path, Path]:
    """Write resampling indices CSV and a small JSON manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Derive label from weights filename if it matches weights_scf_YYYYMMDD.csv
    label = ""
    stem = weights_csv.stem
    if stem.startswith("weights_scf_") and len(stem) >= len("weights_scf_YYYYMMDD"):
        label = stem.split("weights_scf_")[-1]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    idx_csv = out_dir / (f"resample_indices_{label}.csv" if label else f"resample_indices_{ts}.csv")
    man_json = out_dir / (f"resample_manifest_{label}.json" if label else f"resample_manifest_{ts}.json")

    # Indices CSV: posterior_id, source_id, weight
    pd.DataFrame(pairs, columns=["posterior_member_id", "source_member_id", "weight"]).to_csv(idx_csv, index=False)

    manifest = {
        "algorithm": alg,
        "ess": float(ess),
        "n": int(n),
        "seed": (int(seed) if seed is not None else None),
        "skipped": bool(skipped),
        "ess_threshold": float(ess_threshold),
        "weights_csv": str(weights_csv),
        "created_utc": ts,
        "mapping_csv": str(idx_csv),
    }
    try:
        import json

        man_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort; indices CSV is the critical artifact
        pass
    return idx_csv, man_json


def resample_from_weights(
    *,
    step_dir: Path,
    source_ensemble: str,
    weights_csv: Path,
    target_ensemble: str,
    seed: Optional[int],
    algorithm: str,
    ess_threshold: float,
    ess_threshold_ratio: float | None,
    overwrite: bool,
    always_copy: bool,
) -> dict:
    """Core API: read weights, compute ESS, and materialize target ensemble.

    Returns a small dict with summary stats and output paths.
    """
    # Load weights
    df = _load_weights(weights_csv)
    w = np.asarray(df["weight"], dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("Weights contain non-finite values")
    w = w / np.sum(w)
    ess = effective_sample_size(w)
    n = w.size

    # Source members and order
    src_members = list_member_dirs(step_dir / "ensembles", source_ensemble)
    if not src_members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{source_ensemble}")

    # If member_id column present, align to it; otherwise assume order matches
    if "member_id" in df.columns:
        id_to_path = {p.name: p for p in src_members}
        try:
            src_members = [id_to_path[str(mid)] for mid in df["member_id"].tolist()]
        except KeyError as e:
            raise RuntimeError(f"member_id in weights not found in source ensemble: {e}")
    if len(src_members) != n:
        raise RuntimeError(f"Mismatch: weights N={n} vs source members={len(src_members)}")

    # Decide: resample vs mirror
    if algorithm and algorithm != "systematic":
        raise NotImplementedError(f"Resampling algorithm '{algorithm}' not implemented (use 'systematic')")
    do_resample = True
    # Compute effective absolute threshold
    thr_abs = 0.0
    if ess_threshold_ratio is not None and ess_threshold_ratio > 0:
        thr_abs = float(ess_threshold_ratio) * float(n)
    elif ess_threshold and ess_threshold > 0:
        thr_abs = float(ess_threshold) if ess_threshold > 1.0 else float(ess_threshold) * float(n)
    if thr_abs and ess >= thr_abs:
        do_resample = False
    if seed is None:
        # fall back to a time-based seed for traceability (logged in manifest)
        seed = int(datetime.utcnow().timestamp())

    indices: Optional[np.ndarray]
    if do_resample:
        rng = np.random.default_rng(int(seed))
        indices = systematic_resample(rng, w, n=n)
        logger.info(
            "Resampling ({}) | N={} ESS={:.1f} thr_abs={:.1f} thr_ratio={}",
            (algorithm or "systematic"),
            n,
            ess,
            thr_abs,
            (f"{ess_threshold_ratio:.2f}" if ess_threshold_ratio else "NA"),
        )
    else:
        indices = None
        logger.info(
            "Skipping resampling | ESS={:.1f} >= thr_abs={:.1f} (mirror source->target)",
            ess,
            thr_abs,
        )

    # Materialize posterior
    pairs = _mirror_or_resample(
        step_dir=step_dir,
        source_ensemble=source_ensemble,
        target_ensemble=target_ensemble,
        members_order=src_members,
        draw_indices=indices,
        overwrite=overwrite,
        force_copy=always_copy,
        weights=w,
    )

    # Manifests
    assim_dir = Path(step_dir) / "assim"
    idx_csv, man_json = _write_manifest(
        out_dir=assim_dir,
        weights_csv=weights_csv,
        alg=(algorithm or "systematic"),
        ess=ess,
        n=n,
        seed=seed,
        skipped=(indices is None),
        pairs=pairs,
        ess_threshold=thr_abs,
    )

    # Uniqueness stats for transparency
    unique_sources = len({src for _post, src, _w in pairs}) if pairs else 0

    return {
        "N": n,
        "ESS": ess,
        "resampled": bool(indices is not None),
        "indices_csv": str(idx_csv),
        "manifest_json": str(man_json),
        "target_root": str(Path(step_dir) / "ensembles" / target_ensemble),
        "unique_sources": int(unique_sources),
        "unique_fraction": (float(unique_sources) / float(n) if n > 0 else 0.0),
    }


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-resample", description="Systematic resampling to form a posterior ensemble")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"), help="Source ensemble")
    p.add_argument("--weights", required=True, type=Path, help="Path to weights CSV (single date)")
    p.add_argument("--target", default="posterior", choices=("posterior",), help="Target ensemble name")
    p.add_argument("--seed", type=int, help="Random seed for resampling")
    p.add_argument("--ess-threshold", type=float, help="Absolute threshold; if 0<val<=1 treated as ratio")
    p.add_argument("--ess-threshold-ratio", type=float, help="Ratio threshold in (0,1]; overrides --ess-threshold if set")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing target members")
    p.add_argument("--always-copy", action="store_true", help="Copy files instead of creating symlinks (default)")
    p.add_argument("--prefer-symlink", action="store_true", help="Prefer symlinks over copying (opt-in)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    # Defaults from project.yml
    rs_cfg = _read_resampling_from_project(Path(args.project_dir))
    seed = int(args.seed) if args.seed is not None else (rs_cfg.seed if rs_cfg.seed is not None else None)
    # Parse thresholds with precedence: CLI ratio > CLI abs > config ratio > config abs
    cli_ratio = float(args.ess_threshold_ratio) if getattr(args, "ess_threshold_ratio", None) is not None else None
    cli_abs = float(args.ess_threshold) if args.ess_threshold is not None else None
    ess_thr_ratio = cli_ratio if cli_ratio is not None else rs_cfg.ess_threshold_ratio
    ess_thr_abs = cli_abs if cli_abs is not None else rs_cfg.ess_threshold

    try:
        # Default policy: copy; allow opt-in to symlink preference
        always_copy = True
        if args.prefer_symlink:
            always_copy = False
        if args.always_copy:
            always_copy = True

        algo = rs_cfg.algorithm or "systematic"
        summary = resample_from_weights(
            step_dir=Path(args.step_dir),
            source_ensemble=str(args.ensemble),
            weights_csv=Path(args.weights),
            target_ensemble=str(args.target),
            seed=seed,
            algorithm=algo,
            ess_threshold=ess_thr_abs,
            ess_threshold_ratio=ess_thr_ratio,
            overwrite=bool(args.overwrite),
            always_copy=bool(always_copy),
        )
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return 1

    # If uniqueness stats present, extend the summary line
    if "unique_sources" in summary:
        logger.info(
            "Done | N={N} ESS={ESS:.1f} resampled={resampled} unique={unique_sources}/{N} ({unique_fraction:.2f}) indices={indices_csv}",
            **summary,
        )
    else:
        logger.info(
            "Done | N={N} ESS={ESS:.1f} resampled={resampled} indices={indices_csv}",
            **summary,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

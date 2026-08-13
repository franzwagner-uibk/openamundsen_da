"""openamundsen_da.methods.pf.resample

Systematic resampling of a single-date weights CSV to form a posterior ensemble.

Features
- Reads a weights CSV produced by an assimilation step with normalized weights.
- Computes ESS and, if below a threshold, draws N indices via systematic resampling.
- Materializes `<step>/ensembles/posterior/member_XXX` by copying (symlink if possible)
  from the selected source members under `<step>/ensembles/<source>/member_*`.
- Writes a compact manifest and indices CSV for traceability.
- If ESS >= threshold, optionally skip resampling and mirror source -> target.

Logging uses constants.LOGURU_FORMAT (green timestamp | level | message).
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import json
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import (
    RESAMPLING_BLOCK,
    RESAMPLING_ALGORITHM,
    RESAMPLING_ESS_THRESHOLD,
    RESAMPLING_ESS_THRESHOLD_RATIO,
    DA_BLOCK,
    MEMBER_PREFIX,
    MEMBER_SOURCE_POINTER,
    STATE_POINTER_JSON,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import list_member_dirs, find_project_yaml, infer_project_dir
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.manifests import hash_json, load_manifest, sha256_file, write_manifest_atomic
from openamundsen_da.util.keyed_rng import RNG_SCHEME, keyed_rng, keyed_seed
from openamundsen_da.util.stats import effective_sample_size, normalize_weights, systematic_resample


@dataclass(frozen=True)
class ResamplingConfig:
    algorithm: str = "systematic"
    ess_threshold: float = 0.0  # absolute; if 0, never skip (always resample)
    ess_threshold_ratio: float | None = None  # 0..1
    seed: int | None = None


def _read_resampling_from_project(project_dir: Path) -> ResamplingConfig:
    """Read and validate required resampling settings from project YAML."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    da = cfg.get(DA_BLOCK)
    if not isinstance(da, dict):
        raise ValueError(f"Missing required configuration mapping: {DA_BLOCK} in {project_yaml}")
    r = da.get(RESAMPLING_BLOCK)
    if not isinstance(r, dict):
        raise ValueError(f"Missing required configuration mapping: {DA_BLOCK}.{RESAMPLING_BLOCK}")
    if "seed" not in r:
        raise ValueError(f"Missing required configuration key: {DA_BLOCK}.{RESAMPLING_BLOCK}.seed")
    seed = int(r["seed"])
    if seed < 0:
        raise ValueError(f"{DA_BLOCK}.{RESAMPLING_BLOCK}.seed must be non-negative")
    algo = str(r.get(RESAMPLING_ALGORITHM, "systematic")).strip().lower()
    if algo != "systematic":
        raise ValueError(f"{DA_BLOCK}.{RESAMPLING_BLOCK}.algorithm must be 'systematic'")
    thr = r.get(RESAMPLING_ESS_THRESHOLD)
    thr_ratio = r.get(RESAMPLING_ESS_THRESHOLD_RATIO)
    if thr is not None and thr_ratio is not None:
        raise ValueError(
            f"Configure only one of {DA_BLOCK}.{RESAMPLING_BLOCK}.{RESAMPLING_ESS_THRESHOLD} "
            f"or {RESAMPLING_ESS_THRESHOLD_RATIO}"
        )
    ratio_val: float | None = None
    abs_val = 0.0
    if thr_ratio is not None:
        ratio_val = float(thr_ratio)
        if not 0.0 < ratio_val <= 1.0:
            raise ValueError(f"{RESAMPLING_ESS_THRESHOLD_RATIO} must lie in (0, 1]")
    elif thr is not None:
        threshold = float(thr)
        if threshold <= 0.0:
            raise ValueError(f"{RESAMPLING_ESS_THRESHOLD} must be positive")
        if threshold <= 1.0:
            ratio_val = threshold
        else:
            abs_val = threshold
    else:
        raise ValueError(
            f"Missing required resampling threshold: configure {RESAMPLING_ESS_THRESHOLD_RATIO}"
        )
    return ResamplingConfig(
        algorithm=algo,
        ess_threshold=abs_val,
        ess_threshold_ratio=ratio_val,
        seed=seed,
    )


def _load_weights(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"weight"}
    if not needed.issubset(df.columns):
        raise ValueError("Weights CSV missing 'weight' column")
    if "member_id" in df.columns:
        df["member_id"] = df["member_id"].astype(str)
        if df["member_id"].duplicated().any():
            raise ValueError("Weights CSV contains duplicate member_id values")
        df = df.sort_values("member_id").reset_index(drop=True)
    return df


def _weights_digest(csv_path: Path) -> str:
    """Hash the canonical member-aligned weights rather than CSV row order."""
    frame = _load_weights(csv_path)
    columns = [column for column in ("member_id", "weight") if column in frame.columns]
    records = frame[columns].to_dict(orient="records")
    return hash_json(records)


def _mirror_or_resample(
    *,
    step_dir: Path,
    source_ensemble: str,
    target_ensemble: str,
    members_order: list[Path],
    draw_indices: Optional[np.ndarray],
    overwrite: bool,
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
    from openamundsen_da.core.constants import STATE_DEFAULT_NAME
    patt = STATE_DEFAULT_NAME

    for k, (src_member, wv) in enumerate(zip(mapping, w_map), start=1):
        tgt_member = tgt_root / f"{MEMBER_PREFIX}{k:03d}"
        pairs.append((tgt_member.name, src_member.name, wv))

        if tgt_member.exists() and overwrite:
            shutil.rmtree(tgt_member, ignore_errors=True)
        if tgt_member.exists():
            pointer_path = tgt_member / MEMBER_SOURCE_POINTER
            if not pointer_path.is_file():
                raise RuntimeError(f"Existing posterior member is missing its source pointer: {pointer_path}")
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                actual_source = Path(pointer["member_dir"]).resolve()
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid posterior member source pointer: {pointer_path}") from exc
            if actual_source != src_member.resolve():
                raise RuntimeError(
                    f"Existing posterior ancestry mismatch for {tgt_member.name}: "
                    f"expected {src_member.resolve()}, found {actual_source}"
                )
            actual_weight = pointer.get("source_posterior_weight")
            if wv is not None and (actual_weight is None or not np.isclose(float(actual_weight), float(wv))):
                raise RuntimeError(f"Existing posterior source weight mismatch for {tgt_member.name}")
            continue

        # Create minimal target member dir; avoid duplicating large files.
        tgt_member.mkdir(parents=True, exist_ok=True)

        # Write a member-level source pointer for downstream tools (portable)
        (tgt_member / MEMBER_SOURCE_POINTER).write_text(
            json.dumps(
                {
                    "member_dir": str(src_member.resolve()),
                    "source_posterior_weight": (float(wv) if wv is not None else None),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Ensure meteo/ exists for compatibility; results/ will be created by the launcher
        (tgt_member / "meteo").mkdir(exist_ok=True)

        # Create a state pointer instead of copying the state file
        src_state = None
        # Try exact file, then glob
        exact = (src_member / "results" / str(patt))
        if exact.exists() and exact.is_file():
            src_state = exact
        else:
            matches = list((src_member / "results").glob(str(patt)))
            if matches:
                matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                src_state = matches[0]
        if src_state is not None:
            # Write pointer at member root (portable across layouts)
            src_resolved = src_state.resolve()
            try:
                rel = src_resolved.relative_to(tgt_member)
                out = {"path": str(rel)}
            except Exception:
                out = {"path": str(src_resolved)}
            (tgt_member / STATE_POINTER_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return pairs


def _resampling_output_paths(out_dir: Path, weights_csv: Path) -> tuple[Path, Path]:
    stem = weights_csv.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "weights" and len(parts[-1]) == 8 and parts[-1].isdigit():
        label = parts[-1]
    else:
        label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
        if not label:
            raise ValueError(f"Could not derive a stable resampling label from {weights_csv}")
    return out_dir / f"resample_indices_{label}.csv", out_dir / f"resample_manifest_{label}.json"


def _write_manifest(
    *,
    out_dir: Path,
    weights_csv: Path,
    alg: str,
    ess: float,
    n: int,
    seed: int,
    derived_seed: int,
    skipped: bool,
    pairs: list[tuple[str, str, float | None]],
    ess_threshold: float,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Write resampling indices CSV and a small JSON manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_csv, man_json = _resampling_output_paths(out_dir, weights_csv)

    # Indices CSV: posterior_id, source_id, weight
    expected_mapping = pd.DataFrame(pairs, columns=["posterior_member_id", "source_member_id", "weight"])
    if idx_csv.exists() != man_json.exists():
        raise RuntimeError(f"Incomplete resampling provenance under {out_dir}")
    if idx_csv.is_file() and man_json.is_file() and not overwrite:
        existing_manifest = load_manifest(man_json)
        if existing_manifest is None or existing_manifest.get("resampling_schema_version") != 1:
            raise RuntimeError(f"Unsupported resampling manifest: {man_json}")
        expected_fields = {
            "algorithm": alg,
            "n": int(n),
            "seed": int(seed),
            "derived_seed": int(derived_seed),
            "rng_scheme": RNG_SCHEME,
            "skipped": bool(skipped),
                "weights_sha256": _weights_digest(weights_csv),
        }
        mismatches = {
            key: (existing_manifest.get(key), value)
            for key, value in expected_fields.items()
            if existing_manifest.get(key) != value
        }
        if not np.isclose(float(existing_manifest.get("ess", np.nan)), float(ess)):
            mismatches["ess"] = (existing_manifest.get("ess"), float(ess))
        if not np.isclose(float(existing_manifest.get("ess_threshold", np.nan)), float(ess_threshold)):
            mismatches["ess_threshold"] = (existing_manifest.get("ess_threshold"), float(ess_threshold))
        if existing_manifest.get("mapping_sha256") != sha256_file(idx_csv):
            mismatches["mapping_sha256"] = (existing_manifest.get("mapping_sha256"), sha256_file(idx_csv))
        existing_mapping = pd.read_csv(idx_csv)
        if list(existing_mapping.columns) != list(expected_mapping.columns):
            mismatches["mapping_columns"] = (list(existing_mapping.columns), list(expected_mapping.columns))
        elif not existing_mapping[["posterior_member_id", "source_member_id"]].equals(
            expected_mapping[["posterior_member_id", "source_member_id"]]
        ) or not np.allclose(
            pd.to_numeric(existing_mapping["weight"], errors="raise"),
            pd.to_numeric(expected_mapping["weight"], errors="raise"),
        ):
            mismatches["mapping"] = ("existing", "expected")
        if mismatches:
            raise RuntimeError(f"Existing resampling provenance does not match current inputs: {mismatches}")
        return idx_csv, man_json

    expected_mapping.to_csv(idx_csv, index=False)

    manifest = {
        "resampling_schema_version": 1,
        "status": "complete",
        "algorithm": alg,
        "ess": float(ess),
        "n": int(n),
        "seed": int(seed),
        "derived_seed": int(derived_seed),
        "rng_scheme": RNG_SCHEME,
        "skipped": bool(skipped),
        "ess_threshold": float(ess_threshold),
        "weights_csv": str(weights_csv),
        "weights_sha256": _weights_digest(weights_csv),
        "mapping_csv": str(idx_csv),
        "mapping_sha256": sha256_file(idx_csv),
    }
    write_manifest_atomic(man_json, manifest)
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
) -> dict:
    """Core API: read weights, compute ESS, and materialize target ensemble.

    Returns a small dict with summary stats and output paths.
    """
    # Load weights
    df = _load_weights(weights_csv)
    w = normalize_weights(np.asarray(df["weight"], dtype=float))
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
        raise ValueError("A configured resampling seed is required")
    if int(seed) < 0:
        raise ValueError("Resampling seed must be non-negative")
    event_key = weights_csv.stem
    derived_seed = keyed_seed(int(seed), "resampling", event_key)

    indices: Optional[np.ndarray]
    if do_resample:
        rng = keyed_rng(int(seed), "resampling", event_key)
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
            "Skipping resampling | ESS={:.1f} >= thr_abs={:.1f} (ensemble healthy; mirroring source->target; ess_ratio={:.3f})",
            ess,
            thr_abs,
            (ess / float(n) if n > 0 else 0.0),
        )

    # Materialize posterior
    target_root = Path(step_dir) / "ensembles" / target_ensemble
    existing_targets = sorted(target_root.glob(f"{MEMBER_PREFIX}*")) if target_root.is_dir() else []
    idx_path, manifest_path = _resampling_output_paths(Path(step_dir) / "assim", weights_csv)
    if existing_targets and not overwrite and (not idx_path.is_file() or not manifest_path.is_file()):
        raise RuntimeError(
            f"Existing posterior ensemble lacks complete versioned resampling provenance under {Path(step_dir) / 'assim'}"
        )
    pairs = _mirror_or_resample(
        step_dir=step_dir,
        source_ensemble=source_ensemble,
        target_ensemble=target_ensemble,
        members_order=src_members,
        draw_indices=indices,
        overwrite=overwrite,
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
        derived_seed=derived_seed,
        skipped=(indices is None),
        pairs=pairs,
        ess_threshold=thr_abs,
        overwrite=overwrite,
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
        "storage_output_paths": [
            str(path)
            for posterior_member, _source_member, _weight in pairs
            for path in (
                Path(step_dir)
                / "ensembles"
                / target_ensemble
                / posterior_member
                / MEMBER_SOURCE_POINTER,
                Path(step_dir)
                / "ensembles"
                / target_ensemble
                / posterior_member
                / STATE_POINTER_JSON,
            )
            if path.is_file()
        ],
        "unique_sources": int(unique_sources),
        "unique_fraction": (float(unique_sources) / float(n) if n > 0 else 0.0),
    }


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-resample", description="Systematic resampling to form a posterior ensemble (no duplication; uses pointers for state)")
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"), help="Source ensemble")
    p.add_argument("--weights", required=True, type=Path, help="Path to weights CSV (single date)")
    p.add_argument("--target", default="posterior", choices=("posterior",), help="Target ensemble name")
    p.add_argument("--seed", type=int, help="Random seed for resampling")
    p.add_argument("--ess-threshold", type=float, help="Absolute threshold; if 0<val<=1 treated as ratio")
    p.add_argument("--ess-threshold-ratio", type=float, help="Ratio threshold in (0,1]; overrides --ess-threshold if set")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing target members")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    configure_cli_logger(args.log_level)

    # Defaults from project YAML inferred from --step-dir.
    rs_cfg = _read_resampling_from_project(infer_project_dir(Path(args.step_dir)))
    seed = int(args.seed) if args.seed is not None else (rs_cfg.seed if rs_cfg.seed is not None else None)
    # Parse thresholds with precedence: CLI ratio > CLI abs > config ratio > config abs
    cli_ratio = float(args.ess_threshold_ratio) if getattr(args, "ess_threshold_ratio", None) is not None else None
    cli_abs = float(args.ess_threshold) if args.ess_threshold is not None else None
    ess_thr_ratio = cli_ratio if cli_ratio is not None else rs_cfg.ess_threshold_ratio
    ess_thr_abs = cli_abs if cli_abs is not None else rs_cfg.ess_threshold

    try:
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

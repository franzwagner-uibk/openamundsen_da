"""openamundsen_da.methods.pf.rejuvenate

Create a rejuvenated prior ensemble for the next step from a posterior ensemble
without duplicating large state files.

Behavior
- Reads rejuvenation params from project YAML (data_assimilation.rejuvenation):
  - sigma_t: additive temperature noise
  - sigma_p: multiplicative precipitation noise (lognormal with configured mu_p)
  - sigma_rh: additive dew-point temperature noise
  - sigma_sw: multiplicative shortwave noise (lognormal with mu=0)
- For each posterior member in the previous step:
  - Determine its source member directory via MEMBER_SOURCE_POINTER
    (or fall back to the posterior member itself if missing)
  - Read station CSVs from that source meteo directory, filter to the next
    step time window, apply perturbations, and write into the next step prior
    member meteo directory
  - Copy stations.csv unchanged
  - Copy the state pointer file (STATE_POINTER_JSON) from the posterior
    member's results into the next step prior member results directory
- Writes a compact manifest JSON under next_step/assim.

This avoids copying large state files and keeps ensembles light.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import (
    DA_BLOCK,
    REJUVENATION_BLOCK,
    REJ_SIGMA_T,
    REJ_SIGMA_P,
    REJ_SIGMA_RH,
    REJ_SIGMA_SW,
    DA_MU_P,
    DA_SIGMA_RH,
    DA_SIGMA_SW,
    DEFAULT_TIME_COL,
    MEMBER_PREFIX,
    STATE_POINTER_JSON,
    STATE_DEFAULT_NAME,
    MEMBER_SOURCE_POINTER,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    list_member_dirs,
    meteo_dir_for_member,
    default_results_dir,
    find_step_yaml,
    find_project_yaml,
    infer_project_dir,
    open_loop_dir,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    recursive_files,
    write_manifest_atomic,
)
from openamundsen_da.util.keyed_rng import RNG_SCHEME, keyed_rng, keyed_seed
from openamundsen_da.util.parallel import pick_max_workers, run_tasks_with_pool
from openamundsen_da.util.meteo import filter_and_write_meteo
from openamundsen_da.methods.pf.weights import prior_weight_paths


@dataclass
class RejuvenationParams:
    sigma_t: float
    mu_p: float
    sigma_p: float
    sigma_rh: float
    sigma_sw: float
    seed: int


def _read_rejuvenation_params(project_dir: Path) -> RejuvenationParams:
    """Read rejuvenation params; reuse prior_forcing sigmas by default.

    If rejuvenation distribution parameters are provided, they override;
    otherwise they reuse the prior-forcing values. The stage seed is required.
    """
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    da = cfg.get(DA_BLOCK)
    if not isinstance(da, dict):
        raise ValueError(f"Missing required configuration mapping: {DA_BLOCK}")
    rj = da.get(REJUVENATION_BLOCK)
    if not isinstance(rj, dict):
        raise ValueError(f"Missing required configuration mapping: {DA_BLOCK}.{REJUVENATION_BLOCK}")
    prior = da.get("prior_forcing")
    if not isinstance(prior, dict):
        raise ValueError(f"Missing required configuration mapping: {DA_BLOCK}.prior_forcing")
    _reject_removed_humidity_method_option(prior, f"{DA_BLOCK}.prior_forcing")
    _reject_removed_humidity_method_option(rj, f"{DA_BLOCK}.{REJUVENATION_BLOCK}")
    if "rebase_open_loop" in rj:
        raise ValueError(
            f"{DA_BLOCK}.{REJUVENATION_BLOCK}.rebase_open_loop is unsupported; "
            "rejuvenation always rebuilds forcing from the unmodified setup forcing"
        )
    # Defaults: reuse prior_forcing
    sigma_t = float(rj.get(REJ_SIGMA_T, prior.get("sigma_t", 0.0)))
    mu_p = float(rj.get(DA_MU_P, prior.get(DA_MU_P, 0.0)))
    sigma_p = float(rj.get(REJ_SIGMA_P, prior.get("sigma_p", 0.0)))
    try:
        sigma_rh = float(rj.get(REJ_SIGMA_RH, prior[DA_SIGMA_RH]))
        sigma_sw = float(rj.get(REJ_SIGMA_SW, prior[DA_SIGMA_SW]))
    except KeyError as exc:
        raise ValueError(
            f"Missing required configuration key: {DA_BLOCK}.prior_forcing.{exc.args[0]}"
        ) from exc
    if "seed" not in rj:
        raise ValueError(f"Missing required configuration key: {DA_BLOCK}.{REJUVENATION_BLOCK}.seed")
    seed = int(rj["seed"])
    if seed < 0:
        raise ValueError(f"{DA_BLOCK}.{REJUVENATION_BLOCK}.seed must be non-negative")
    return RejuvenationParams(
        sigma_t=sigma_t,
        mu_p=mu_p,
        sigma_p=sigma_p,
        sigma_rh=sigma_rh,
        sigma_sw=sigma_sw,
        seed=seed,
    )


def _reject_removed_humidity_method_option(block: dict, path: str) -> None:
    if "humidity_perturbation_method" not in block:
        return
    raise ValueError(
        f"{path}.humidity_perturbation_method was removed; "
        f"{REJ_SIGMA_RH} always applies an additive dew-point temperature perturbation"
    )


def _strip_timezone(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert tz-aware timestamps to UTC and drop tz info for safe comparisons."""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC")
    try:
        return ts.tz_localize(None)
    except Exception:
        return ts


def _normalize_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex (converted from UTC if originally tz-aware)."""
    dt_idx = pd.to_datetime(idx, errors="coerce")
    if getattr(dt_idx, "tz", None) is not None:
        dt_idx = dt_idx.tz_convert("UTC").tz_localize(None)
    return dt_idx


def _read_next_step_dates(next_step_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    step_yaml = find_step_yaml(next_step_dir)
    step_cfg = _read_yaml_file(step_yaml) or {}
    try:
        start = pd.to_datetime(step_cfg["start_date"])  # type: ignore[index]
    except Exception as e:
        raise ValueError(f"Missing or invalid start_date in {step_yaml}") from e
    # Prefer project end; fallback to step end_date.
    try:
        project_yaml = find_project_yaml(infer_project_dir(next_step_dir))
        project_cfg = _read_yaml_file(project_yaml) or {}
        end = pd.to_datetime(project_cfg["end_date"])  # type: ignore[index]
    except Exception:
        try:
            end = pd.to_datetime(step_cfg["end_date"])  # type: ignore[index]
        except Exception as e:
            raise ValueError("Could not determine end_date from project/step config") from e
    return start, end


def _inclusive_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start = _strip_timezone(start)
    end = _strip_timezone(end)
    dt_idx = _normalize_datetime_index(df.index)
    mask = (dt_idx >= start) & (dt_idx <= end)
    out = df.loc[mask].copy()
    out.index = dt_idx[mask]
    return out


def _write_csv(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def _source_member_dir(posterior_member: Path) -> Path:
    ptr = posterior_member / MEMBER_SOURCE_POINTER
    if ptr.exists():
        try:
            d = json.loads(ptr.read_text(encoding="utf-8")) or {}
            md = d.get("member_dir")
            if md:
                p = Path(md)
                if not p.is_absolute():
                    p = (posterior_member / p).resolve()
                return p
        except Exception:
            pass
    return posterior_member


def _rejuvenate_member_task(
    member_idx: int,
    post_member: Path,
    src_member: Path,
    tgt_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dT: float,
    fP: float,
    dTd: float,
    fSW: float,
    setup_dir: Path,
    source_meteo_dir: Optional[Path],
) -> dict:
    """Worker: rebase one member's meteo and copy state pointer."""
    member_name = f"{MEMBER_PREFIX}{member_idx:03d}"
    tgt_member = tgt_root / member_name
    tgt_meteo = meteo_dir_for_member(tgt_member)
    tgt_meteo.mkdir(parents=True, exist_ok=True)

    # Choose meteo source and write perturbed window
    src_meteo = Path(source_meteo_dir) if source_meteo_dir is not None else (Path(setup_dir) / "meteo")
    filter_and_write_meteo(
        src_dir=src_meteo,
        dst_dir=tgt_meteo,
        start=start,
        end=end,
        delta_t=dT,
        f_p=fP,
        delta_rh=dTd,
        f_sw=fSW,
    )

    # Copy state pointer if present (support root or results location)
    post_ptr_root = post_member / STATE_POINTER_JSON
    post_ptr_results = default_results_dir(post_member) / STATE_POINTER_JSON
    post_ptr = post_ptr_root if post_ptr_root.exists() else post_ptr_results
    copied_ptr = False
    if post_ptr.exists():
        try:
            data = json.loads(post_ptr.read_text(encoding="utf-8")) or {}
            target = data.get("path") or data.get("state_path")
        except Exception:
            data = None
            target = None
        if target:
            q = Path(target)
            if not q.is_absolute():
                q = (post_ptr.parent / q).resolve()
            try:
                rel = q.relative_to(tgt_member)
                out = {"path": str(rel)}
            except Exception:
                out = {"path": str(q)}
            (tgt_member / STATE_POINTER_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")
            copied_ptr = True

    return {
        "member": member_name,
        "source_member": src_member.name,
        "delta_T": dT,
        "f_p": fP,
        "delta_dew_point": dTd,
        "f_sw": fSW,
        "copied_state_pointer": copied_ptr,
        "rebase_open_loop": True,
    }


def rejuvenate(
    *,
    setup_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    source_ensemble: str = "posterior",
    target_ensemble: str = "prior",
    source_meteo_dir: Optional[Path] = None,
) -> dict:
    project_dir = infer_project_dir(next_step_dir)
    params = _read_rejuvenation_params(project_dir)
    start, end = _read_next_step_dates(next_step_dir)
    src_members = list_member_dirs(Path(prev_step_dir) / "ensembles", source_ensemble)
    if not src_members:
        raise RuntimeError(f"No members under {prev_step_dir}/ensembles/{source_ensemble}")

    tgt_root = Path(next_step_dir) / "ensembles" / target_ensemble
    tgt_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    event_key = Path(next_step_dir).name
    for i, post_member in enumerate(src_members, start=1):
        src_member = _source_member_dir(post_member)
        member_key = post_member.name
        dT = (
            float(keyed_rng(params.seed, "rejuvenation", event_key, member_key, "temperature").normal(0.0, params.sigma_t))
            if params.sigma_t
            else 0.0
        )
        fP = (
            float(
                keyed_rng(params.seed, "rejuvenation", event_key, member_key, "precipitation").lognormal(
                    mean=params.mu_p,
                    sigma=params.sigma_p,
                )
            )
            if params.sigma_p
            else float(math.exp(params.mu_p)) if params.mu_p else 1.0
        )
        dTd = (
            float(keyed_rng(params.seed, "rejuvenation", event_key, member_key, "dew_point").normal(0.0, params.sigma_rh))
            if params.sigma_rh
            else 0.0
        )
        fSW = (
            float(keyed_rng(params.seed, "rejuvenation", event_key, member_key, "shortwave").lognormal(0.0, params.sigma_sw))
            if params.sigma_sw
            else 1.0
        )
        tasks.append(
            (i, post_member, src_member, tgt_root, start, end, dT, fP, dTd, fSW, Path(setup_dir), source_meteo_dir)
        )

    if not tasks:
        return {"members": 0, "copied_state_pointers": 0}

    workers = pick_max_workers(None, fallback=len(tasks), limit=len(tasks))
    logger.info("Rejuvenating {} member(s) with max_workers={}", len(tasks), workers)

    rows = run_tasks_with_pool(
        _rejuvenate_member_task,
        tasks,
        max_workers=workers,
        fallback_workers=len(tasks),
        label="rejuvenate",
    )
    copied_pointers = sum(int(r.get("copied_state_pointer")) for r in rows)
    for res in rows:
        logger.info(
            "[{m}] dT={dt:+.3f} f_p={fp:.3f} dTd={dtd:+.3f} f_sw={fsw:.3f} state_ptr={sp} rebase=True",
            m=res["member"],
            dt=res["delta_T"],
            fp=res["f_p"],
            dtd=res["delta_dew_point"],
            fsw=res["f_sw"],
            sp=res["copied_state_pointer"],
        )

    # Also prepare open_loop for the next step and copy state pointer
    try:
        _copy_open_loop_to_next(Path(setup_dir), Path(prev_step_dir), Path(next_step_dir), start=start, end=end)
    except Exception as e:
        logger.warning("Could not prepare open_loop for next step: {}", e)

    out_dir = Path(next_step_dir) / "assim"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["member"])
    manifest_inputs = _rejuvenation_input_inventory(
        setup_dir=Path(setup_dir),
        prev_step_dir=Path(prev_step_dir),
        next_step_dir=Path(next_step_dir),
        source_ensemble=source_ensemble,
    )
    manifest_outputs = _rejuvenation_output_inventory(
        setup_dir=Path(setup_dir),
        next_step_dir=Path(next_step_dir),
        target_ensemble=target_ensemble,
    )
    manifest = {
        "rejuvenation_schema_version": 1,
        "status": "complete",
        "source_step": str(prev_step_dir),
        "target_step": str(next_step_dir),
        "source_ensemble": source_ensemble,
        "target_ensemble": target_ensemble,
        "sigma_t": params.sigma_t,
        "mu_p": params.mu_p,
        "sigma_p": params.sigma_p,
        "sigma_rh": params.sigma_rh,
        "sigma_sw": params.sigma_sw,
        "seed": int(params.seed),
        "rng_scheme": RNG_SCHEME,
        "event_key": event_key,
        "event_seed": keyed_seed(params.seed, "rejuvenation", event_key),
        "copied_state_pointers": int(copied_pointers),
        "members": rows_sorted,
        "input_inventory": manifest_inputs,
        "input_inventory_sha256": inventory_digest(manifest_inputs),
        "output_inventory": manifest_outputs,
        "output_inventory_sha256": inventory_digest(manifest_outputs),
    }
    write_manifest_atomic(out_dir / "rejuvenate_manifest.json", manifest)

    return {"members": len(rows_sorted), "copied_state_pointers": copied_pointers}


def _rejuvenation_input_inventory(
    *,
    setup_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    source_ensemble: str,
) -> list[dict[str, object]]:
    project_dir = infer_project_dir(next_step_dir)
    files: list[Path] = []
    files.extend(recursive_files(setup_dir / "meteo"))
    files.extend(project_dir.glob("*.yml"))
    files.extend(project_dir.glob("*.yaml"))
    files.extend(next_step_dir.glob("*.yml"))
    files.extend(next_step_dir.glob("*.yaml"))
    files.extend(recursive_files(prev_step_dir / "assim"))
    files.extend(path for path in prior_weight_paths(next_step_dir) if path.is_file())
    for member in list_member_dirs(prev_step_dir / "ensembles", source_ensemble):
        for pointer_name in (MEMBER_SOURCE_POINTER, STATE_POINTER_JSON):
            pointer = member / pointer_name
            if pointer.is_file():
                files.append(pointer)
    return file_inventory(root=setup_dir, files=files)


def _rejuvenation_output_inventory(
    *,
    setup_dir: Path,
    next_step_dir: Path,
    target_ensemble: str,
) -> list[dict[str, object]]:
    files: list[Path] = []
    target_root = next_step_dir / "ensembles" / target_ensemble
    for member in list_member_dirs(next_step_dir / "ensembles", target_ensemble):
        files.extend(recursive_files(member / "meteo"))
        pointer = member / STATE_POINTER_JSON
        if pointer.is_file():
            files.append(pointer)
    open_loop = target_root / "open_loop"
    files.extend(recursive_files(open_loop / "meteo"))
    if (open_loop / STATE_POINTER_JSON).is_file():
        files.append(open_loop / STATE_POINTER_JSON)
    return file_inventory(root=setup_dir, files=files)


def validate_rejuvenation_manifest(
    *,
    setup_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    source_ensemble: str = "posterior",
    target_ensemble: str = "prior",
) -> dict:
    """Validate resume provenance and generated process-noise forcing."""
    manifest_path = Path(next_step_dir) / "assim" / "rejuvenate_manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Missing rejuvenation manifest: {manifest_path}")
    if manifest.get("rejuvenation_schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"Unsupported or incomplete rejuvenation manifest: {manifest_path}")
    params = _read_rejuvenation_params(infer_project_dir(next_step_dir))
    expected = {
        "source_step": str(prev_step_dir),
        "target_step": str(next_step_dir),
        "source_ensemble": source_ensemble,
        "target_ensemble": target_ensemble,
        "sigma_t": params.sigma_t,
        "mu_p": params.mu_p,
        "sigma_p": params.sigma_p,
        "sigma_rh": params.sigma_rh,
        "sigma_sw": params.sigma_sw,
        "seed": int(params.seed),
        "rng_scheme": RNG_SCHEME,
        "event_key": Path(next_step_dir).name,
        "event_seed": keyed_seed(params.seed, "rejuvenation", Path(next_step_dir).name),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    inputs = _rejuvenation_input_inventory(
        setup_dir=Path(setup_dir),
        prev_step_dir=Path(prev_step_dir),
        next_step_dir=Path(next_step_dir),
        source_ensemble=source_ensemble,
    )
    outputs = _rejuvenation_output_inventory(
        setup_dir=Path(setup_dir),
        next_step_dir=Path(next_step_dir),
        target_ensemble=target_ensemble,
    )
    if manifest.get("input_inventory_sha256") != inventory_digest(inputs):
        mismatches["input_inventory_sha256"] = (manifest.get("input_inventory_sha256"), inventory_digest(inputs))
    if manifest.get("output_inventory_sha256") != inventory_digest(outputs):
        mismatches["output_inventory_sha256"] = (manifest.get("output_inventory_sha256"), inventory_digest(outputs))
    if mismatches:
        raise RuntimeError(f"Rejuvenation resume provenance mismatch: {mismatches}")
    return manifest


def _copy_open_loop_to_next(
    setup_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    *,
    start: "pd.Timestamp",
    end: "pd.Timestamp",
) -> None:
    """Prepare next-step open_loop meteo and write a state pointer.

    - Filters previous step open_loop meteo to [start..end] and writes into
      next step prior open_loop/meteo.
    - Writes next step prior open_loop/STATE_POINTER_JSON pointing to the
      previous step open_loop state file.
    """
    prev_ol = open_loop_dir(prev_step_dir)
    next_ol = open_loop_dir(next_step_dir)
    start = _strip_timezone(start)
    end = _strip_timezone(end)

    # Base meteo comes from project-level meteo directory
    met_prev = Path(setup_dir) / "meteo"
    met_next = next_ol / "meteo"
    met_next.mkdir(parents=True, exist_ok=True)

    stations_csv = met_prev / "stations.csv"
    if stations_csv.exists():
        (met_next / "stations.csv").write_bytes(stations_csv.read_bytes())

    for src in sorted(met_prev.glob("*.csv")):
        if src.name.lower() == "stations.csv":
            continue
        df = pd.read_csv(src, parse_dates=True, index_col=0)
        time_col = df.index.name or DEFAULT_TIME_COL
        df = _inclusive_filter(df, start, end)
        df.index = _normalize_datetime_index(df.index)
        idx_col_name = df.index.name or "index"
        df_out = df.reset_index().rename(columns={idx_col_name: time_col})
        _write_csv(df_out, met_next / src.name)

    # Copy a pointer to the previous step's open_loop state file
    res_prev = prev_ol / "results"
    cand = res_prev / STATE_DEFAULT_NAME
    if not cand.exists():
        picks = sorted(res_prev.glob("*.pickle.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
        cand = picks[0] if picks else None
    if cand and cand.exists():
        next_ol.mkdir(parents=True, exist_ok=True)
        try:
            rel = cand.resolve().relative_to(next_ol)
            out = {"path": str(rel)}
        except Exception:
            out = {"path": str(cand.resolve())}
        (next_ol / STATE_POINTER_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-rejuvenate", description="Rejuvenate posterior into prior for next step (rebase on open_loop; no state duplication)")
    p.add_argument("--setup-dir", required=True, type=Path)
    p.add_argument("--prev-step-dir", required=True, type=Path)
    p.add_argument("--next-step-dir", required=True, type=Path)
    p.add_argument("--source-meteo-dir", type=Path, help="Explicit meteo source directory (stations.csv + per-station CSVs). Overrides rebase/compound base selection")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    configure_cli_logger(args.log_level)

    try:
        summary = rejuvenate(
            setup_dir=Path(args.setup_dir),
            prev_step_dir=Path(args.prev_step_dir),
            next_step_dir=Path(args.next_step_dir),
            source_meteo_dir=(Path(args.source_meteo_dir) if args.source_meteo_dir is not None else None),
        )
        logger.info("Rejuvenated prior | members={} state_ptrs={}", summary.get("members"), summary.get("copied_state_pointers"))
        return 0
    except Exception as e:
        logger.error(f"Rejuvenation failed: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

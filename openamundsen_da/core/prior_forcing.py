from __future__ import annotations
"""
openamundsen_da.core.prior_forcing

Build a prior ensemble of meteorological forcings (perturbed CSVs) for
openAMUNDSEN. Creates an open-loop copy and N member_XXX sets under the step
directory, matching the discovery used by the ensemble launcher.

Design
- Inputs: explicit input meteo dir, project dir, and step dir
- Dates: inclusive [start_date..end_date] read from the step YAML
- Params: read from project YAML under data_assimilation.prior_forcing
- Perturbations: additive temperature and humidity-state offsets plus
  multiplicative precipitation and shortwave factors, constant per member
  across stations and time
- Schema: first column must be datetime (name is flexible); 'temp', 'precip',
  'rel_hum', and 'sw_in' are optional per station file
- Precip negatives: if 'precip' exists and contains negatives, abort
- Output: <step_dir>/ensembles/prior/{open_loop,member_XXX}/{meteo,results}
"""

import argparse
import concurrent.futures as cf
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from loguru import logger

from openamundsen_da.core.env import (
    apply_env_from_project,
    ensure_gdal_proj_from_conda,
    apply_numeric_thread_defaults,
    _read_yaml_file,
)

from openamundsen_da.core.constants import (
    DA_BLOCK,
    DA_PRIOR_BLOCK,
    DA_ENSEMBLE_SIZE,
    DA_RANDOM_SEED,
    DA_SIGMA_T,
    DA_MU_P,
    DA_SIGMA_P,
    DA_SIGMA_RH,
    DA_SIGMA_SW,
    ENSEMBLE_PRIOR,
    START_DATE,
    END_DATE,
    LOGURU_FORMAT,
)
from openamundsen_da.util.stats import (
    sample_delta_rh,
    sample_delta_t,
    sample_precip_factor,
    sample_shortwave_factor,
)
from openamundsen_da.util.keyed_rng import RNG_SCHEME, keyed_rng, keyed_seed
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    recursive_files,
    write_manifest_atomic,
)
from openamundsen_da.util.storage_admission import (
    StorageAccountingSummary,
    accounting_summary_delta,
    accounting_summary_from_inventory,
    reused_accounting_summary,
)
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.meteo import filter_and_write_meteo
from openamundsen_da.io.paths import (
    find_setup_yaml,
    find_step_yaml,
    find_project_yaml,
    infer_project_dir,
    infer_setup_dir_from_project,
    meteo_dir_for_member,
    default_results_dir,
    prior_root as prior_root_dir,
    open_loop_dir as open_loop_dir_for_step,
    member_dir_for_index,
)


@dataclass
class PriorParams:
    """Prior configuration read from project YAML."""
    ensemble_size: int
    random_seed: int
    sigma_t: float
    mu_p: float
    sigma_p: float
    sigma_rh: float
    sigma_sw: float


PRIOR_FORCING_MANIFEST = "prior_forcing_manifest.json"


def _read_prior_params(project_dir: Path) -> PriorParams:
    """Load prior configuration from project YAML > data_assimilation.prior_forcing."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    da = (cfg.get(DA_BLOCK) or {}).get(DA_PRIOR_BLOCK) or {}
    _reject_removed_humidity_method_option(da, f"{DA_BLOCK}.{DA_PRIOR_BLOCK}")
    try:
        seed = int(da[DA_RANDOM_SEED])
        if seed < 0:
            raise ValueError(f"{DA_BLOCK}.{DA_PRIOR_BLOCK}.{DA_RANDOM_SEED} must be non-negative")
        return PriorParams(
            ensemble_size=int(da[DA_ENSEMBLE_SIZE]),
            random_seed=seed,
            sigma_t=float(da[DA_SIGMA_T]),
            mu_p=float(da[DA_MU_P]),
            sigma_p=float(da[DA_SIGMA_P]),
            sigma_rh=float(da[DA_SIGMA_RH]),
            sigma_sw=float(da[DA_SIGMA_SW]),
        )
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(
            f"Missing prior parameter in project YAML:{' ' + missing} under "
            f"{DA_BLOCK}.{DA_PRIOR_BLOCK}"
        ) from e


def _reject_removed_humidity_method_option(block: dict, path: str) -> None:
    if "humidity_perturbation_method" not in block:
        return
    raise ValueError(
        f"{path}.humidity_perturbation_method was removed; "
        f"{DA_SIGMA_RH} always applies an additive dew-point temperature perturbation"
    )


def _read_step_window(step_dir: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Read the inclusive forcing window from the consuming step YAML."""
    step_yaml = find_step_yaml(step_dir)
    step_cfg = _read_yaml_file(step_yaml) or {}
    try:
        start = pd.to_datetime(step_cfg[START_DATE])
    except KeyError as e:
        raise ValueError(f"Missing required key '{START_DATE}' in {step_yaml}") from e
    if pd.isna(start):
        raise ValueError(f"Invalid {START_DATE} in {step_yaml}")
    try:
        end = pd.to_datetime(step_cfg[END_DATE])
    except KeyError as e:
        raise ValueError(f"Missing required key '{END_DATE}' in {step_yaml}") from e
    if pd.isna(end):
        raise ValueError(f"Invalid {END_DATE} in {step_yaml}")
    if end < start:
        raise ValueError(f"Invalid step window in {step_yaml}: {END_DATE} precedes {START_DATE}")
    return start, end


def _make_member_dirs(root: Path) -> Tuple[Path, Path]:
    """Create meteo and results subdirs under the given member/open_loop root."""
    meteo = meteo_dir_for_member(root)
    results = default_results_dir(root)
    meteo.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return meteo, results


def _write_info(
    member_root: Path,
    name: str,
    seed: int,
    delta_t: float,
    f_p: float,
    delta_rh: float,
    f_sw: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    input_dir: Path,
) -> None:
    """Write a compact INFO.txt summarizing the member perturbations and context."""
    info = member_root / "INFO.txt"
    lines = [
        f"Member: {name}",
        f"Random seed: {seed}",
        "",
        "Perturbations (constant per member):",
        f"  delta_T (additive): {delta_t:+.3f}",
        f"  precip factor f_p:  {f_p:.3f}",
        f"  delta_Td (dew point additive): {delta_rh:+.3f}",
        f"  shortwave factor f_sw: {f_sw:.3f}",
        "",
        "Date filter (inclusive):",
        f"  start_date: {start}",
        f"  end_date:   {end}",
        "",
        "Schema:",
        "  required: first column = datetime (name flexible)",
        "  optional: temp, precip, rel_hum, sw_in",
        "",
        "Input:",
        f"  meteo dir: {input_dir}",
        "",
        "Layout:",
        f"  {member_root / 'meteo'}",
        f"  {member_root / 'results'}",
    ]
    info.write_text("\n".join(lines), encoding="utf-8")


def _build_member(
    member_idx: int,
    member_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    delta_t: float,
    f_p: float,
    delta_rh: float,
    f_sw: float,
    random_seed: int,
    input_meteo_dir: Path,
) -> None:
    """Worker: write perturbed meteo for one member."""
    meteo_dir, _ = _make_member_dirs(member_root)
    filter_and_write_meteo(
        src_dir=input_meteo_dir,
        dst_dir=meteo_dir,
        start=start,
        end=end,
        delta_t=delta_t,
        f_p=f_p,
        delta_rh=delta_rh,
        f_sw=f_sw,
    )
    _write_info(
        member_root,
        f"member_{member_idx:03d}",
        seed=random_seed,
        delta_t=delta_t,
        f_p=f_p,
        delta_rh=delta_rh,
        f_sw=f_sw,
        start=start,
        end=end,
        input_dir=input_meteo_dir,
    )


def _prior_forcing_inventory(*, root: Path, paths: list[Path]) -> list[dict[str, object]]:
    return file_inventory(root=root, files=paths)


def _prior_forcing_input_inventory(
    *,
    setup_dir: Path,
    input_meteo_dir: Path,
    project_dir: Path,
    step_dir: Path,
) -> list[dict[str, object]]:
    files = recursive_files(input_meteo_dir)
    files.extend(project_dir.glob("*.yml"))
    files.extend(project_dir.glob("*.yaml"))
    files.extend(step_dir.glob("*.yml"))
    files.extend(step_dir.glob("*.yaml"))
    return _prior_forcing_inventory(root=setup_dir, paths=files)


def _prior_forcing_output_inventory(*, setup_dir: Path, step_dir: Path) -> list[dict[str, object]]:
    files: list[Path] = []
    root = prior_root_dir(step_dir)
    for member in sorted(root.glob("member_*")):
        files.extend(recursive_files(member / "meteo"))
        info = member / "INFO.txt"
        if info.is_file():
            files.append(info)
    files.extend(recursive_files(open_loop_dir_for_step(step_dir) / "meteo"))
    return _prior_forcing_inventory(root=setup_dir, paths=files)


def _prior_forcing_manifest_path(step_dir: Path) -> Path:
    return Path(step_dir) / "assim" / PRIOR_FORCING_MANIFEST


def validate_prior_forcing_manifest(
    *,
    input_meteo_dir: Path,
    project_dir: Path,
    step_dir: Path,
) -> dict:
    """Validate initial process-noise forcing before reusing an existing stage."""
    manifest_path = _prior_forcing_manifest_path(step_dir)
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Missing prior-forcing manifest: {manifest_path}")
    if manifest.get("prior_forcing_schema_version") != 2 or manifest.get("status") != "complete":
        raise ValueError(f"Unsupported or incomplete prior-forcing manifest: {manifest_path}")
    params = _read_prior_params(project_dir)
    start, end = _read_step_window(step_dir)
    setup_dir = project_dir.parent.parent.resolve()
    expected = {
        "ensemble_size": params.ensemble_size,
        "random_seed": params.random_seed,
        "sigma_t": params.sigma_t,
        "mu_p": params.mu_p,
        "sigma_p": params.sigma_p,
        "sigma_rh": params.sigma_rh,
        "sigma_sw": params.sigma_sw,
        "rng_scheme": RNG_SCHEME,
        "event_key": step_dir.name,
        "event_seed": keyed_seed(params.random_seed, "initial_forcing", step_dir.name),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    inputs = _prior_forcing_input_inventory(
        setup_dir=setup_dir,
        input_meteo_dir=input_meteo_dir,
        project_dir=project_dir,
        step_dir=step_dir,
    )
    outputs = _prior_forcing_output_inventory(setup_dir=setup_dir, step_dir=step_dir)
    if manifest.get("input_inventory_sha256") != inventory_digest(inputs):
        mismatches["input_inventory_sha256"] = (manifest.get("input_inventory_sha256"), inventory_digest(inputs))
    if manifest.get("output_inventory_sha256") != inventory_digest(outputs):
        mismatches["output_inventory_sha256"] = (manifest.get("output_inventory_sha256"), inventory_digest(outputs))
    if mismatches:
        raise RuntimeError(f"Prior-forcing resume provenance mismatch: {mismatches}")
    return manifest


def build_prior_ensemble(
    input_meteo_dir: Path | str,
    project_dir: Path | str,
    step_dir: Path | str,
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build open-loop and prior ensemble member meteo directories for a step.

    Parameters
    - input_meteo_dir: Path to original, long-span station CSV directory
      (contains stations.csv and <station_id>.csv files)
    - project_dir: Path to project directory under setup/projects
    - step_dir: Path to step directory containing step_XX.yml with dates
    - max_workers: Optional worker cap for member creation (defaults to
      min(CPU, ensemble_size, MAX_WORKERS env))
    - overwrite: If False, skip existing open_loop/member directories; if True,
      (re)write outputs
    """
    input_meteo_dir = Path(input_meteo_dir)
    project_dir = Path(project_dir)
    step_dir = Path(step_dir)

    # Read configuration
    inferred_project_dir = infer_project_dir(step_dir)
    if inferred_project_dir != project_dir:
        raise ValueError(
            f"step_dir belongs to {inferred_project_dir}, but --project-dir is {project_dir}"
        )
    params = _read_prior_params(project_dir)
    start, end = _read_step_window(step_dir)
    manifest_path = _prior_forcing_manifest_path(step_dir)
    prior_root = prior_root_dir(step_dir)
    before_accounting = accounting_summary_from_inventory(
        completed_step=step_dir.name,
        inventory=_prior_forcing_output_inventory(
            setup_dir=project_dir.parent.parent.resolve(),
            step_dir=step_dir,
        ),
        source="prior_forcing_before",
    )
    if manifest_path.is_file() and not overwrite:
        validated = validate_prior_forcing_manifest(
            input_meteo_dir=input_meteo_dir,
            project_dir=project_dir,
            step_dir=step_dir,
        )
        logger.info("Validated prior-forcing manifest -> {}", manifest_path)
        accounting = validated.get("storage_accounting")
        if not isinstance(accounting, dict):
            accounting = accounting_summary_from_inventory(
                completed_step=step_dir.name,
                inventory=list(validated.get("output_inventory") or []),
                source="prior_forcing_reconciliation",
            ).as_dict()
        return reused_accounting_summary(
            StorageAccountingSummary.from_dict(accounting),
            source="prior_forcing_reused",
        ).as_dict()
    if prior_root.exists() and any(prior_root.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Existing prior forcing under {prior_root} lacks a compatible versioned manifest; "
            "use --overwrite to rebuild it"
        )

    logger.info(
        "Building prior ensemble -> ensemble={ens}  N={n}  seed={seed}",
        ens=ENSEMBLE_PRIOR, n=params.ensemble_size, seed=params.random_seed,
    )
    logger.info("Dates (inclusive): {s} .. {e}", s=str(start), e=str(end))

    # Prepare open_loop
    open_loop_root = open_loop_dir_for_step(step_dir)
    if open_loop_root.exists() and not overwrite:
        logger.info("Open-loop exists -> skipping (use --overwrite to rebuild)")
    else:
        meteo_ol, _ = _make_member_dirs(open_loop_root)
        filter_and_write_meteo(
            src_dir=input_meteo_dir,
            dst_dir=meteo_ol,
            start=start,
            end=end,
            delta_t=0.0,
            f_p=1.0,
            delta_rh=0.0,
            f_sw=1.0,
        )
        logger.info("Open-loop written: {p}", p=str(open_loop_root))

    # Prepare member tasks (skip existing unless overwrite)
    tasks = []
    perturbations: list[dict[str, object]] = []
    for i in range(1, params.ensemble_size + 1):
        member_name = f"member_{i:03d}"
        member_root = member_dir_for_index(step_dir, i)
        if member_root.exists() and not overwrite:
            logger.info(f"[{member_name}] exists -> skipping (use --overwrite)")
            continue
        event_key = step_dir.name
        delta_t = sample_delta_t(
            keyed_rng(params.random_seed, "initial_forcing", event_key, member_name, "temperature"),
            params.sigma_t,
        )
        f_p = sample_precip_factor(
            keyed_rng(params.random_seed, "initial_forcing", event_key, member_name, "precipitation"),
            params.mu_p,
            params.sigma_p,
        )
        delta_rh = sample_delta_rh(
            keyed_rng(params.random_seed, "initial_forcing", event_key, member_name, "dew_point"),
            params.sigma_rh,
        )
        f_sw = sample_shortwave_factor(
            keyed_rng(params.random_seed, "initial_forcing", event_key, member_name, "shortwave"),
            params.sigma_sw,
        )
        logger.info(
            "[{m}] delta_T={dt:+.3f}  f_p={fp:.3f}  delta_Td={drh:+.3f}  f_sw={fsw:.3f}",
            m=member_name,
            dt=delta_t,
            fp=f_p,
            drh=delta_rh,
            fsw=f_sw,
        )
        perturbations.append(
            {
                "member": member_name,
                "delta_T": delta_t,
                "f_p": f_p,
                "delta_dew_point": delta_rh,
                "f_sw": f_sw,
            }
        )
        tasks.append((i, member_root, delta_t, f_p, delta_rh, f_sw, input_meteo_dir))

    if not tasks:
        logger.info("No members to build (all exist and overwrite is False).")
        logger.info("Prior ensemble completed under: {root}", root=str(prior_root))
        outputs = _prior_forcing_output_inventory(
            setup_dir=project_dir.parent.parent.resolve(),
            step_dir=step_dir,
        )
        return reused_accounting_summary(accounting_summary_from_inventory(
            completed_step=step_dir.name,
            inventory=outputs,
            source="prior_forcing_reconciliation",
        ), source="prior_forcing_reused").as_dict()

    workers = pick_max_workers(max_workers, fallback=params.ensemble_size, limit=len(tasks))
    logger.info("Building {} member(s) with max_workers={}", len(tasks), workers)

    if workers <= 1:
        for i, member_root, delta_t, f_p, delta_rh, f_sw, src_dir in tasks:
            _build_member(
                member_idx=i,
                member_root=member_root,
                start=start,
                end=end,
                delta_t=delta_t,
                f_p=f_p,
                delta_rh=delta_rh,
                f_sw=f_sw,
                random_seed=params.random_seed,
                input_meteo_dir=src_dir,
            )
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _build_member,
                    i,
                    member_root,
                    start,
                    end,
                    delta_t,
                    f_p,
                    delta_rh,
                    f_sw,
                    params.random_seed,
                    src_dir,
                ): f"member_{i:03d}"
                for i, member_root, delta_t, f_p, delta_rh, f_sw, src_dir in tasks
            }
            for fut in cf.as_completed(futs):
                name = futs[fut]
                try:
                    fut.result()
                    logger.info("[{}] written", name)
                except Exception as e:
                    logger.error("[{}] failed: {}", name, e)
                    raise

    logger.info("Prior ensemble completed under: {root}", root=str(prior_root))
    setup_dir = project_dir.parent.parent.resolve()
    inputs = _prior_forcing_input_inventory(
        setup_dir=setup_dir,
        input_meteo_dir=input_meteo_dir,
        project_dir=project_dir,
        step_dir=step_dir,
    )
    outputs = _prior_forcing_output_inventory(setup_dir=setup_dir, step_dir=step_dir)
    storage_accounting = accounting_summary_delta(
        before=before_accounting,
        after=accounting_summary_from_inventory(
            completed_step=step_dir.name,
            inventory=outputs,
            source="prior_forcing_after",
        ),
        source="prior_forcing",
    ).as_dict()
    write_manifest_atomic(
        manifest_path,
        {
            "prior_forcing_schema_version": 2,
            "status": "complete",
            "ensemble_size": params.ensemble_size,
            "random_seed": params.random_seed,
            "sigma_t": params.sigma_t,
            "mu_p": params.mu_p,
            "sigma_p": params.sigma_p,
            "sigma_rh": params.sigma_rh,
            "sigma_sw": params.sigma_sw,
            "rng_scheme": RNG_SCHEME,
            "event_key": step_dir.name,
            "event_seed": keyed_seed(params.random_seed, "initial_forcing", step_dir.name),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "members": perturbations,
            "input_inventory": inputs,
            "input_inventory_sha256": inventory_digest(inputs),
            "output_inventory": outputs,
            "output_inventory_sha256": inventory_digest(outputs),
            "storage_accounting": storage_accounting,
        },
    )
    return storage_accounting


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prior meteo ensemble (CSV)")
    p.add_argument("--input-meteo-dir", required=True, type=Path)
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--max-workers", type=int, default=None, help="Max workers for member build (overrides MAX_WORKERS env)")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        # Configure logger to match launcher formatting (green timestamp | level | message)
        logger.remove()
        logger.add(
            sys.stderr,
            level=args.log_level,
            enqueue=True,
            colorize=True,
            format=LOGURU_FORMAT,
        )

        # Apply environment (suppress GDAL warnings by setting GDAL_DATA/PROJ_LIB where possible)
        try:
            setup_dir = infer_setup_dir_from_project(args.project_dir)
            setup_yaml = find_setup_yaml(setup_dir)
            apply_env_from_project(setup_yaml)
        except Exception:
            # Best-effort: continue even if project env section is missing
            pass
        ensure_gdal_proj_from_conda()
        apply_numeric_thread_defaults()
        # Quiet GDAL debug output; avoid importing osgeo to prevent early warnings
        os.environ.setdefault("CPL_DEBUG", "OFF")

        build_prior_ensemble(
            args.input_meteo_dir,
            args.project_dir,
            args.step_dir,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Persistent importance-weight ledgers for sequential particle filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    recursive_files,
    sha256_file,
    write_manifest_atomic,
)
from openamundsen_da.util.stats import effective_sample_size, logsumexp, normalize_weights


PRIOR_WEIGHTS_FILENAME = "prior_weights.csv"
PRIOR_WEIGHTS_MANIFEST_FILENAME = "prior_weights_manifest.json"
WEIGHT_LEDGER_SCHEMA_VERSION = 1
EVENT_WEIGHTS_SCHEMA_VERSION = 1


def prior_weight_paths(step_dir: Path) -> tuple[Path, Path]:
    """Return the CSV and manifest paths for one step's prior weights."""
    assim_dir = Path(step_dir) / "assim"
    return assim_dir / PRIOR_WEIGHTS_FILENAME, assim_dir / PRIOR_WEIGHTS_MANIFEST_FILENAME


def event_weights_manifest_path(weights_csv: Path) -> Path:
    """Return the versioned manifest path paired with an event weights CSV."""
    weights_csv = Path(weights_csv)
    return weights_csv.with_name(f"{weights_csv.stem}_manifest.json")


def _event_input_files(*, project_dir: Path, step_dir: Path) -> list[Path]:
    files: list[Path] = []
    files.extend(Path(project_dir).glob("*.yml"))
    files.extend(Path(project_dir).glob("*.yaml"))
    files.extend(Path(step_dir).glob("*.yml"))
    files.extend(Path(step_dir).glob("*.yaml"))
    files.extend(recursive_files(Path(step_dir) / "obs"))
    prior_csv, prior_manifest = prior_weight_paths(step_dir)
    files.extend((prior_csv, prior_manifest))
    for member_results in sorted((Path(step_dir) / "ensembles" / "prior").glob("member_*/results")):
        files.extend(
            path
            for path in recursive_files(member_results)
            if path.name == "member_run.json"
            or path.name == "output_grids.nc"
            or path.name.startswith("point_")
            or path.suffix.lower() in {".tif", ".tiff"}
        )
    return files


def write_event_weights(
    weights_csv: Path,
    weights: pd.DataFrame,
    *,
    project_dir: Path,
    step_dir: Path,
) -> Path:
    """Write event weights and a hash-bound resume manifest."""
    required = {
        "member_id",
        "prior_log_weight",
        "prior_weight",
        "log_likelihood",
        "log_weight",
        "weight",
        "prior_ess",
        "posterior_ess",
        "resampling_threshold",
        "resampled",
    }
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"Event weights missing required columns: {sorted(missing)}")
    weights_csv = Path(weights_csv)
    weights_csv.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(weights_csv, index=False)
    root = Path(project_dir).resolve()
    inventory = file_inventory(
        root=root,
        files=_event_input_files(project_dir=project_dir, step_dir=step_dir),
    )
    manifest = {
        "event_weights_schema_version": EVENT_WEIGHTS_SCHEMA_VERSION,
        "status": "complete",
        "step": Path(step_dir).name,
        "weights_csv": str(weights_csv.resolve().relative_to(root)),
        "weights_sha256": sha256_file(weights_csv),
        "input_inventory": inventory,
        "input_inventory_sha256": inventory_digest(inventory),
    }
    return write_manifest_atomic(event_weights_manifest_path(weights_csv), manifest)


def load_event_weights(
    weights_csv: Path,
    *,
    project_dir: Path,
    step_dir: Path,
) -> pd.DataFrame:
    """Load event weights only when their config, inputs and ancestry still match."""
    weights_csv = Path(weights_csv)
    manifest_path = event_weights_manifest_path(weights_csv)
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Missing event-weights manifest: {manifest_path}")
    if manifest.get("event_weights_schema_version") != EVENT_WEIGHTS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported event-weights manifest: {manifest_path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete event-weights manifest: {manifest_path}")
    if not weights_csv.is_file() or manifest.get("weights_sha256") != sha256_file(weights_csv):
        raise ValueError(f"Event-weights hash mismatch: {weights_csv}")
    root = Path(project_dir).resolve()
    current_inventory = file_inventory(
        root=root,
        files=_event_input_files(project_dir=project_dir, step_dir=step_dir),
    )
    if manifest.get("input_inventory_sha256") != inventory_digest(current_inventory):
        raise ValueError(
            f"Event-weights inputs changed for {weights_csv}; rerun assimilation with overwrite enabled"
        )
    return pd.read_csv(weights_csv)


def _member_ids(member_ids: Iterable[str]) -> list[str]:
    values = [str(member_id) for member_id in member_ids]
    if not values:
        raise ValueError("A PF weight ledger requires at least one member")
    if len(values) != len(set(values)):
        raise ValueError("PF member IDs must be unique")
    return values


def _normalized_log_weights(weights: np.ndarray) -> np.ndarray:
    normalized = normalize_weights(weights)
    with np.errstate(divide="ignore"):
        return np.log(normalized)


def write_prior_weights(
    step_dir: Path,
    *,
    member_ids: Iterable[str],
    weights: Iterable[float] | None,
    mode: str,
    source_step: Path | None = None,
    source_weights: Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write a normalized prior-weight ledger and its ancestry manifest."""
    ids = _member_ids(member_ids)
    if mode not in {"initialized_uniform", "carried_posterior", "resampled_uniform"}:
        raise ValueError(f"Unsupported prior-weight ledger mode: {mode!r}")
    if weights is None:
        normalized = np.full(len(ids), 1.0 / len(ids), dtype=float)
    else:
        normalized = normalize_weights(np.asarray(list(weights), dtype=float))
        if normalized.size != len(ids):
            raise ValueError("Prior weights and member IDs must have the same length")

    csv_path, manifest_path = prior_weight_paths(step_dir)
    if (csv_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"Prior-weight ledger already exists for {step_dir}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "member_id": ids,
            "log_weight": _normalized_log_weights(normalized),
            "weight": normalized,
        }
    )
    frame.to_csv(csv_path, index=False)
    manifest = {
        "weight_ledger_schema_version": WEIGHT_LEDGER_SCHEMA_VERSION,
        "status": "complete",
        "mode": mode,
        "step": Path(step_dir).name,
        "source_step": Path(source_step).name if source_step is not None else None,
        "source_weights": str(Path(source_weights)) if source_weights is not None else None,
        "member_count": len(ids),
        "ess": effective_sample_size(normalized),
        "weights_sha256": sha256_file(csv_path),
    }
    write_manifest_atomic(manifest_path, manifest)
    return frame


def initialize_prior_weights(
    step_dir: Path,
    member_ids: Iterable[str],
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Initialize the first PF step with uniform importance weights."""
    return write_prior_weights(
        step_dir,
        member_ids=member_ids,
        weights=None,
        mode="initialized_uniform",
        overwrite=overwrite,
    )


def load_prior_weights(step_dir: Path, member_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Load and strictly validate one step's prior-weight ledger."""
    csv_path, manifest_path = prior_weight_paths(step_dir)
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing PF prior-weight ledger for {step_dir}; initialize a new chain explicitly "
            "or run the preceding resampling stage"
        )
    manifest = load_manifest(manifest_path)
    if manifest is None or manifest.get("weight_ledger_schema_version") != WEIGHT_LEDGER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported PF prior-weight ledger in {manifest_path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete PF prior-weight ledger in {manifest_path}")
    if manifest.get("weights_sha256") != sha256_file(csv_path):
        raise ValueError(f"PF prior-weight ledger hash mismatch in {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"member_id", "log_weight", "weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"PF prior-weight ledger missing columns {sorted(required - set(frame.columns))}: {csv_path}")
    if frame["member_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate member IDs in PF prior-weight ledger: {csv_path}")
    frame["member_id"] = frame["member_id"].astype(str)
    frame["weight"] = normalize_weights(pd.to_numeric(frame["weight"], errors="raise").to_numpy(dtype=float))
    frame["log_weight"] = pd.to_numeric(frame["log_weight"], errors="raise")
    positive = frame["weight"].to_numpy(dtype=float) > 0.0
    if not np.allclose(
        frame.loc[positive, "log_weight"].to_numpy(dtype=float),
        np.log(frame.loc[positive, "weight"].to_numpy(dtype=float)),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(f"Inconsistent log_weight and weight columns in {csv_path}")

    if member_ids is not None:
        expected = _member_ids(member_ids)
        actual = frame["member_id"].tolist()
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(f"PF prior-weight member mismatch in {csv_path}: missing={missing}, extra={extra}")
        frame = frame.set_index("member_id").loc[expected].reset_index()
    return frame


def combine_event_weights(
    event_frame: pd.DataFrame,
    *,
    step_dir: Path,
    initialize: bool = False,
) -> pd.DataFrame:
    """Combine event log likelihoods with the persisted sequential prior."""
    if "member_id" not in event_frame.columns:
        raise ValueError("Event weights must contain member_id")
    result = event_frame.copy()
    result["member_id"] = result["member_id"].astype(str)
    if result["member_id"].duplicated().any():
        raise ValueError("Event weights contain duplicate member IDs")
    if "log_likelihood" in result.columns:
        log_likelihood = pd.to_numeric(result["log_likelihood"], errors="raise").to_numpy(dtype=float)
    elif "log_weight" in result.columns:
        log_likelihood = pd.to_numeric(result["log_weight"], errors="raise").to_numpy(dtype=float)
    else:
        raise ValueError("Event weights must contain log_likelihood")
    if not np.all(np.isfinite(log_likelihood)):
        raise ValueError("Event log likelihoods must be finite")

    member_ids = result["member_id"].tolist()
    csv_path, _ = prior_weight_paths(step_dir)
    if initialize and not csv_path.is_file():
        initialize_prior_weights(step_dir, member_ids)
    prior = load_prior_weights(step_dir, member_ids)
    prior_log = prior["log_weight"].to_numpy(dtype=float)
    unnormalized = prior_log + log_likelihood
    normalizer = logsumexp(unnormalized)
    posterior_log = unnormalized - normalizer
    posterior = np.exp(posterior_log)
    posterior = normalize_weights(posterior)

    result["prior_log_weight"] = prior_log
    result["prior_weight"] = prior["weight"].to_numpy(dtype=float)
    result["log_likelihood"] = log_likelihood
    result["log_weight"] = posterior_log
    result["weight"] = posterior
    result["prior_ess"] = effective_sample_size(result["prior_weight"].to_numpy(dtype=float))
    result["posterior_ess"] = effective_sample_size(posterior)
    result["ess"] = result["posterior_ess"]
    return result


def carry_weights_to_next_step(
    *,
    current_step_dir: Path,
    next_step_dir: Path,
    event_weights: pd.DataFrame,
    mapping: pd.DataFrame,
    resampled: bool,
    source_weights: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create the next-step prior ledger from a resampling member mapping."""
    required_mapping = {"posterior_member_id", "source_member_id"}
    if not required_mapping.issubset(mapping.columns):
        raise ValueError(f"Resampling mapping missing columns: {sorted(required_mapping - set(mapping.columns))}")
    target_ids = mapping["posterior_member_id"].astype(str).tolist()
    if resampled:
        weights = None
        mode = "resampled_uniform"
    else:
        source = event_weights.copy()
        source["member_id"] = source["member_id"].astype(str)
        source_weights_by_id = source.set_index("member_id")["weight"]
        try:
            weights = [float(source_weights_by_id.loc[str(member_id)]) for member_id in mapping["source_member_id"]]
        except KeyError as exc:
            raise ValueError(f"Resampling mapping references unknown source member {exc.args[0]!r}") from exc
        mode = "carried_posterior"
    return write_prior_weights(
        next_step_dir,
        member_ids=target_ids,
        weights=weights,
        mode=mode,
        source_step=current_step_dir,
        source_weights=source_weights,
        overwrite=overwrite,
    )


__all__ = [
    "PRIOR_WEIGHTS_FILENAME",
    "PRIOR_WEIGHTS_MANIFEST_FILENAME",
    "WEIGHT_LEDGER_SCHEMA_VERSION",
    "EVENT_WEIGHTS_SCHEMA_VERSION",
    "carry_weights_to_next_step",
    "combine_event_weights",
    "event_weights_manifest_path",
    "initialize_prior_weights",
    "load_prior_weights",
    "load_event_weights",
    "prior_weight_paths",
    "write_prior_weights",
    "write_event_weights",
]

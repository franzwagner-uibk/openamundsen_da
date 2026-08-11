"""Fixed, conservative disk-admission policy for project steps."""

from __future__ import annotations

import shutil
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError


SOFT_USED_FRACTION = 0.80
EMERGENCY_USED_FRACTION = 0.90
OPERATIONAL_RESERVE_FRACTION = 0.05


@dataclass(frozen=True)
class DiskBudgetSnapshot:
    filesystem_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    estimated_growth_bytes: int
    operational_reserve_bytes: int

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.total_bytes

    @property
    def projected_used_fraction(self) -> float:
        return (
            self.used_bytes
            + self.estimated_growth_bytes
            + self.operational_reserve_bytes
        ) / self.total_bytes


def _parse_csv_timestamp(raw: str) -> datetime | None:
    value = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _station_file_bounds(path: Path) -> tuple[datetime, datetime] | None:
    """Read first/last timestamps without loading a station file into memory."""
    with path.open("rb") as stream:
        header_raw = stream.readline().decode("utf-8-sig", errors="strict")
        first_raw = stream.readline().decode("utf-8", errors="strict")
        if not first_raw:
            return None
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 65536))
        tail = stream.read().decode("utf-8", errors="strict").splitlines()
    header = next(csv.reader([header_raw]))
    first = next(csv.reader([first_raw]))
    if not header or not first:
        return None
    try:
        date_idx = header.index("date")
    except ValueError as exc:
        raise ValueError(f"Forcing CSV has no date column: {path}") from exc
    first_ts = _parse_csv_timestamp(first[date_idx])
    last_ts = None
    for line in reversed(tail):
        if not line.strip() or line == header_raw.rstrip("\r\n"):
            continue
        row = next(csv.reader([line]))
        if row:
            last_ts = _parse_csv_timestamp(row[date_idx])
            if last_ts is not None:
                break
    if first_ts is None or last_ts is None:
        return None
    return first_ts, last_ts


def estimate_step_forcing_bytes(
    meteo_dir: str | Path,
    *,
    start: datetime,
    end: datetime,
    ensemble_size: int,
) -> int:
    """Estimate generated forcing bytes from source size and temporal coverage."""
    meteo_dir = Path(meteo_dir)
    if ensemble_size < 1 or end < start:
        raise ValueError("Invalid forcing estimate inputs")
    station_files = sorted(
        path for path in meteo_dir.glob("*.csv") if path.name != "stations.csv" and path.is_file()
    )
    if not station_files:
        raise FileNotFoundError(f"No station forcing CSV files found in {meteo_dir}")
    first_times: list[datetime] = []
    last_times: list[datetime] = []
    payload_bytes = 0
    for path in station_files:
        payload_bytes += path.stat().st_size
        bounds = _station_file_bounds(path)
        if bounds is not None:
            first_times.append(bounds[0])
            last_times.append(bounds[1])
    if not first_times:
        raise ValueError(f"Could not read forcing time coverage in {meteo_dir}")
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)
    coverage_seconds = max(1.0, (max(last_times) - min(first_times)).total_seconds())
    window_seconds = max(1.0, (end - start).total_seconds())
    fraction = min(1.0, window_seconds / coverage_seconds)
    metadata_bytes = (meteo_dir / "stations.csv").stat().st_size if (meteo_dir / "stations.csv").is_file() else 0
    # CSV formatting and uneven station coverage make exact byte prediction
    # impossible before generation. Keep a 35% conservative serialization
    # margin while scaling only the requested step window.
    per_copy = int(payload_bytes * fraction * 1.35) + metadata_bytes
    return per_copy * (ensemble_size + 1)


def estimate_compact_timeseries_bytes(project_dir: str | Path) -> int:
    """Conservatively reserve one raw-byte equivalent for compact exports."""
    project_dir = Path(project_dir).resolve()
    patterns = (
        "steps/step_*/ensembles/*/*/results/point_*.csv",
        "steps/step_*/ensembles/*/*/meteo/*.csv",
    )
    paths = {
        path.resolve()
        for pattern in patterns
        for path in project_dir.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    # Compression normally makes the NetCDF smaller than the CSV source. Keep
    # ten percent for metadata, temporary files and sparse point variables.
    return int(sum(path.stat().st_size for path in paths) * 1.10)


def check_step_admission(
    project_dir: str | Path,
    *,
    estimated_growth_bytes: int = 0,
    allow_existing_step_drain: bool = False,
    usage: shutil._ntuple_diskusage | None = None,
) -> DiskBudgetSnapshot:
    """Refuse a new step when fixed project-filesystem limits are exceeded."""
    project_dir = Path(project_dir).resolve()
    if estimated_growth_bytes < 0:
        raise ValueError("estimated_growth_bytes must be non-negative")
    current = usage if usage is not None else shutil.disk_usage(project_dir)
    if current.total <= 0:
        raise RuntimeError(f"Could not determine filesystem capacity for {project_dir}")
    snapshot = DiskBudgetSnapshot(
        filesystem_path=project_dir,
        total_bytes=int(current.total),
        used_bytes=int(current.used),
        free_bytes=int(current.free),
        estimated_growth_bytes=int(estimated_growth_bytes),
        operational_reserve_bytes=int(current.total * OPERATIONAL_RESERVE_FRACTION),
    )
    if snapshot.used_fraction >= EMERGENCY_USED_FRACTION:
        raise LowDiskEmergencyError(
            f"Project filesystem is at or above the fixed 90% emergency limit "
            f"({snapshot.used_fraction:.1%} used): {project_dir}"
        )
    if snapshot.used_fraction >= SOFT_USED_FRACTION and not allow_existing_step_drain:
        raise LowDiskPauseError(
            f"Project filesystem is at or above the fixed 80% step-admission limit "
            f"({snapshot.used_fraction:.1%} used): {project_dir}"
        )
    if snapshot.projected_used_fraction >= EMERGENCY_USED_FRACTION:
        raise LowDiskPauseError(
            "Step completion estimate would reach the fixed 90% emergency limit: "
            f"current={snapshot.used_fraction:.1%}, projected={snapshot.projected_used_fraction:.1%}, "
            f"estimated_growth={snapshot.estimated_growth_bytes} bytes"
        )
    return snapshot


__all__ = [
    "DiskBudgetSnapshot",
    "EMERGENCY_USED_FRACTION",
    "OPERATIONAL_RESERVE_FRACTION",
    "SOFT_USED_FRACTION",
    "check_step_admission",
    "estimate_compact_timeseries_bytes",
    "estimate_step_forcing_bytes",
]

"""Command-scoped immutable catalog for storage-planning source files."""

from __future__ import annotations

import bisect
import csv
import hashlib
import math
import os
import stat as stat_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _ForcingIndex:
    header_bytes: int
    timestamps: tuple[datetime, ...]
    prefix_row_bytes: tuple[int, ...]

    def selected_bytes(self, *, start: datetime, end: datetime) -> int:
        first = bisect.bisect_left(self.timestamps, start)
        last = bisect.bisect_right(self.timestamps, end)
        payload = self.prefix_row_bytes[last] - self.prefix_row_bytes[first]
        return self.header_bytes + math.ceil(payload * 1.35)


@dataclass(frozen=True)
class _ForcingDirectory:
    station_indexes: tuple[_ForcingIndex, ...]
    metadata_bytes: int


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_csv_timestamp(raw: str) -> datetime | None:
    value = raw.strip().replace("Z", "+00:00")
    try:
        return _normalized_datetime(datetime.fromisoformat(value))
    except ValueError:
        return None


class SourceCatalog:
    """Read each immutable source inode at most once during one command.

    Logical aliases remain distinct for output-size arithmetic, while hashing
    and forcing-row parsing are cached by stable inode identity. Every source
    read is bracketed by descriptor metadata checks so concurrent mutation
    fails closed instead of mixing generations.
    """

    def __init__(self, *, trusted_root: str | Path) -> None:
        self.trusted_root = Path(trusted_root).resolve(strict=True)
        self._digests: dict[_FileIdentity, str] = {}
        self._forcing_indexes: dict[_FileIdentity, _ForcingIndex] = {}
        self._forcing_directories: dict[Path, _ForcingDirectory] = {}
        self._logical_identities: dict[Path, _FileIdentity] = {}
        self._resolved_paths: dict[Path, Path] = {}
        self._hashed_bytes = 0
        self._forcing_bytes = 0
        self._forcing_window_queries = 0
        self._payload_reads: set[_FileIdentity] = set()

    @staticmethod
    def _identity(metadata: os.stat_result) -> _FileIdentity:
        return _FileIdentity(
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            size=int(metadata.st_size),
            mtime_ns=int(metadata.st_mtime_ns),
            ctime_ns=int(metadata.st_ctime_ns),
        )

    def _open_source(self, path: str | Path) -> tuple[int, Path, _FileIdentity]:
        logical = Path(path).absolute()
        resolved = logical.resolve(strict=True)
        try:
            resolved.relative_to(self.trusted_root)
        except ValueError as exc:
            raise ValueError(
                f"Catalog source escapes trusted root {self.trusted_root}: {logical}"
            ) from exc
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        fd = os.open(resolved, flags)
        try:
            metadata = os.fstat(fd)
            if not stat_module.S_ISREG(metadata.st_mode):
                raise ValueError(f"Catalog source is not a regular file: {logical}")
            identity = self._identity(metadata)
            previous = self._logical_identities.get(logical)
            if previous is not None and previous != identity:
                raise RuntimeError(f"Catalog source changed during preflight: {logical}")
            self._logical_identities[logical] = identity
            previous_resolved = self._resolved_paths.get(logical)
            if previous_resolved is not None and previous_resolved != resolved:
                raise RuntimeError(f"Catalog source target changed during preflight: {logical}")
            self._resolved_paths[logical] = resolved
            return fd, resolved, identity
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _verify_fd(fd: int, identity: _FileIdentity, path: Path) -> None:
        if SourceCatalog._identity(os.fstat(fd)) != identity:
            raise RuntimeError(f"Catalog source changed while it was read: {path}")

    def sha256_file(self, path: str | Path) -> str:
        """Return a stable digest while hashing each physical file once."""
        fd, resolved, identity = self._open_source(path)
        try:
            cached = self._digests.get(identity)
            if cached is not None:
                return cached
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1024 * 1024):
                digest.update(chunk)
            self._verify_fd(fd, identity, resolved)
            value = digest.hexdigest()
            self._digests[identity] = value
            self._hashed_bytes += identity.size
            self._payload_reads.add(identity)
            return value
        finally:
            os.close(fd)

    def file_size(self, path: str | Path) -> int:
        """Return one stable contained source size without reading its payload."""
        fd, resolved, identity = self._open_source(path)
        try:
            self._verify_fd(fd, identity, resolved)
            return identity.size
        finally:
            os.close(fd)

    def _forcing_index(self, path: Path) -> _ForcingIndex:
        fd, resolved, identity = self._open_source(path)
        try:
            cached = self._forcing_indexes.get(identity)
            if cached is not None:
                return cached
            with os.fdopen(fd, "rb", closefd=False) as stream:
                header_raw = stream.readline()
                if not header_raw:
                    raise ValueError(f"Forcing CSV is empty: {path}")
                digest = hashlib.sha256(header_raw)
                header = next(
                    csv.reader([header_raw.decode("utf-8-sig", errors="strict")])
                )
                try:
                    date_idx = header.index("date")
                except ValueError as exc:
                    raise ValueError(f"Forcing CSV has no date column: {path}") from exc
                rows: list[tuple[datetime, int]] = []
                for line_number, raw_line in enumerate(stream, start=2):
                    digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    row = next(csv.reader([raw_line.decode("utf-8", errors="strict")]))
                    if date_idx >= len(row):
                        raise ValueError(f"Forcing row {line_number} has no date in {path}")
                    timestamp = _parse_csv_timestamp(row[date_idx])
                    if timestamp is None:
                        raise ValueError(
                            f"Invalid forcing timestamp on row {line_number} in {path}"
                        )
                    rows.append((timestamp, len(raw_line)))
            self._verify_fd(fd, identity, resolved)
            rows.sort(key=lambda item: item[0])
            timestamps = tuple(item[0] for item in rows)
            prefix = [0]
            for _timestamp, row_bytes in rows:
                prefix.append(prefix[-1] + row_bytes)
            index = _ForcingIndex(
                header_bytes=len(header_raw),
                timestamps=timestamps,
                prefix_row_bytes=tuple(prefix),
            )
            self._forcing_indexes[identity] = index
            self._forcing_bytes += identity.size
            if identity not in self._digests:
                self._digests[identity] = digest.hexdigest()
                self._hashed_bytes += identity.size
            self._payload_reads.add(identity)
            return index
        finally:
            os.close(fd)

    def _forcing_directory(self, meteo_dir: str | Path) -> _ForcingDirectory:
        logical_dir = Path(meteo_dir).absolute()
        cached = self._forcing_directories.get(logical_dir)
        if cached is not None:
            return cached
        resolved_dir = logical_dir.resolve(strict=True)
        try:
            resolved_dir.relative_to(self.trusted_root)
        except ValueError as exc:
            raise ValueError(
                f"Forcing directory escapes trusted root {self.trusted_root}: {logical_dir}"
            ) from exc
        station_files = sorted(
            path
            for path in logical_dir.glob("*.csv")
            if path.name != "stations.csv" and path.is_file()
        )
        if not station_files:
            raise FileNotFoundError(f"No station forcing CSV files found in {logical_dir}")
        metadata_path = logical_dir / "stations.csv"
        directory = _ForcingDirectory(
            station_indexes=tuple(self._forcing_index(path) for path in station_files),
            metadata_bytes=self.file_size(metadata_path) if metadata_path.is_file() else 0,
        )
        self._forcing_directories[logical_dir] = directory
        return directory

    def estimate_step_forcing_bytes(
        self,
        meteo_dir: str | Path,
        *,
        start: datetime,
        end: datetime,
        ensemble_size: int,
    ) -> int:
        """Estimate one window without rereading indexed forcing rows."""
        if ensemble_size < 1 or end < start:
            raise ValueError("Invalid forcing estimate inputs")
        start = _normalized_datetime(start)
        end = _normalized_datetime(end)
        directory = self._forcing_directory(meteo_dir)
        self._forcing_window_queries += 1
        payload = sum(
            index.selected_bytes(start=start, end=end)
            for index in directory.station_indexes
        )
        return (payload + directory.metadata_bytes) * (ensemble_size + 1)

    def summary(self) -> dict[str, Any]:
        """Return compact audit counters without exposing the row index."""
        return {
            "unique_source_files": len(set(self._logical_identities.values())),
            "logical_source_paths": len(self._logical_identities),
            "unique_hashed_files": len(self._digests),
            "unique_hashed_bytes": self._hashed_bytes,
            "forcing_files_parsed": len(self._forcing_indexes),
            "forcing_bytes_parsed": self._forcing_bytes,
            "forcing_directories": len(self._forcing_directories),
            "forcing_window_queries": self._forcing_window_queries,
            "unique_payload_bytes_read": sum(
                identity.size for identity in self._payload_reads
            ),
        }

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return the immutable stat identity used for cheap phase validation."""
        return tuple(
            {
                "logical_path": str(logical),
                "resolved_path": str(self._resolved_paths[logical]),
                "device": identity.device,
                "inode": identity.inode,
                "size": identity.size,
                "mtime_ns": identity.mtime_ns,
                "ctime_ns": identity.ctime_ns,
            }
            for logical, identity in sorted(
                self._logical_identities.items(),
                key=lambda item: str(item[0]),
            )
        )

    @staticmethod
    def verify_snapshot(
        snapshot: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        *,
        trusted_root: str | Path,
    ) -> None:
        """Fail closed when any cataloged logical source changed identity."""
        trusted = Path(trusted_root).resolve(strict=True)
        for record in snapshot:
            logical = Path(str(record["logical_path"]))
            resolved = logical.resolve(strict=True)
            try:
                resolved.relative_to(trusted)
            except ValueError as exc:
                raise RuntimeError(
                    f"Catalog source escapes trusted root after preflight: {logical}"
                ) from exc
            if str(resolved) != str(record["resolved_path"]):
                raise RuntimeError(f"Catalog source target changed after preflight: {logical}")
            metadata = resolved.stat()
            identity = SourceCatalog._identity(metadata)
            expected = _FileIdentity(
                device=int(record["device"]),
                inode=int(record["inode"]),
                size=int(record["size"]),
                mtime_ns=int(record["mtime_ns"]),
                ctime_ns=int(record["ctime_ns"]),
            )
            if identity != expected:
                raise RuntimeError(f"Catalog source changed after preflight: {logical}")

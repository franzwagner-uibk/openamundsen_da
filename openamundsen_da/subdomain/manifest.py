"""Manifest helpers for sub-domain DA workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class WindowSpec:
    """Spatial window of a sub-domain relative to the global grid."""

    row_off: int
    col_off: int
    height: int
    width: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WindowSpec":
        return cls(
            row_off=int(data["row_off"]),
            col_off=int(data["col_off"]),
            height=int(data["height"]),
            width=int(data["width"]),
        )


@dataclass
class SubdomainMeta:
    """Metadata for one prepared sub-domain setup."""

    id: str
    label: str
    setup_dir: Path
    setup_yaml: Path
    project_dir: Path
    project_yaml: Path
    project_name: str
    grids_dir: Path
    meteo_dir: Path
    obs_stations_dir: Path
    roi_raster_path: Path
    roi_vector_path: Path
    window: WindowSpec
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    crs: Optional[str]
    status: str = "pending"
    run_manifest: Optional[Path] = None
    dropped_events: list[dict] = field(default_factory=list)
    station_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "setup_dir",
            "setup_yaml",
            "project_dir",
            "project_yaml",
            "grids_dir",
            "meteo_dir",
            "obs_stations_dir",
            "roi_raster_path",
            "roi_vector_path",
            "run_manifest",
        ):
            if data.get(key) is not None:
                data[key] = str(data[key])
        data["transform"] = list(self.transform)
        data["bounds"] = list(self.bounds)
        data["window"] = self.window.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SubdomainMeta":
        return cls(
            id=str(data["id"]),
            label=str(data.get("label", data["id"])),
            setup_dir=Path(data["setup_dir"]),
            setup_yaml=Path(data["setup_yaml"]),
            project_dir=Path(data["project_dir"]),
            project_yaml=Path(data["project_yaml"]),
            project_name=str(data["project_name"]),
            grids_dir=Path(data["grids_dir"]),
            meteo_dir=Path(data["meteo_dir"]),
            obs_stations_dir=Path(data["obs_stations_dir"]),
            roi_raster_path=Path(data["roi_raster_path"]),
            roi_vector_path=Path(data["roi_vector_path"]),
            window=WindowSpec.from_dict(data["window"]),
            transform=tuple(float(x) for x in data["transform"]),
            bounds=tuple(float(x) for x in data["bounds"]),
            crs=(data.get("crs") if data.get("crs") not in (None, "None") else None),
            status=str(data.get("status", "pending")),
            run_manifest=(Path(data["run_manifest"]) if data.get("run_manifest") else None),
            dropped_events=list(data.get("dropped_events") or []),
            station_counts={str(k): int(v) for k, v in (data.get("station_counts") or {}).items()},
        )


@dataclass
class SubdomainManifest:
    """Top-level manifest describing sub-domain preparation and execution."""

    run_mode: str
    setup_dir: Path
    project_dir: Path
    project_name: str
    setup_yaml: Path
    project_yaml: Path
    subdomain_root: Path
    regions_path: Path
    id_field: str
    crs: Optional[str]
    grid_rows: int
    grid_cols: int
    grid_transform: tuple[float, float, float, float, float, float]
    grid_resolution: float
    grid_domain: str
    clip_mode: str
    station_buffer_m: float
    roi_buffer_m: float
    grid_buffer_m: float
    raw_snowcover_dir: Path
    raw_wetsnow_dir: Path
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    subdomains: Dict[str, SubdomainMeta] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_mode": self.run_mode,
            "setup_dir": str(self.setup_dir),
            "project_dir": str(self.project_dir),
            "project_name": self.project_name,
            "setup_yaml": str(self.setup_yaml),
            "project_yaml": str(self.project_yaml),
            "subdomain_root": str(self.subdomain_root),
            "regions_path": str(self.regions_path),
            "id_field": self.id_field,
            "crs": self.crs,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "grid_transform": list(self.grid_transform),
            "grid_resolution": self.grid_resolution,
            "grid_domain": self.grid_domain,
            "clip_mode": self.clip_mode,
            "station_buffer_m": self.station_buffer_m,
            "roi_buffer_m": self.roi_buffer_m,
            "grid_buffer_m": self.grid_buffer_m,
            "raw_snowcover_dir": str(self.raw_snowcover_dir),
            "raw_wetsnow_dir": str(self.raw_wetsnow_dir),
            "created_at": self.created_at,
            "subdomains": {k: v.to_dict() for k, v in self.subdomains.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubdomainManifest":
        subs = {k: SubdomainMeta.from_dict(v) for k, v in (data.get("subdomains") or {}).items()}
        return cls(
            run_mode=str(data.get("run_mode", "subdomain")),
            setup_dir=Path(data["setup_dir"]),
            project_dir=Path(data["project_dir"]),
            project_name=str(data["project_name"]),
            setup_yaml=Path(data["setup_yaml"]),
            project_yaml=Path(data["project_yaml"]),
            subdomain_root=Path(data["subdomain_root"]),
            regions_path=Path(data["regions_path"]),
            id_field=str(data["id_field"]),
            crs=data.get("crs"),
            grid_rows=int(data["grid_rows"]),
            grid_cols=int(data["grid_cols"]),
            grid_transform=tuple(float(x) for x in data["grid_transform"]),
            grid_resolution=float(data["grid_resolution"]),
            grid_domain=str(data["grid_domain"]),
            clip_mode=str(data.get("clip_mode", "window")),
            station_buffer_m=float(data.get("station_buffer_m", 0.0)),
            roi_buffer_m=float(data.get("roi_buffer_m", 0.0)),
            grid_buffer_m=float(data.get("grid_buffer_m", 0.0)),
            raw_snowcover_dir=Path(data["raw_snowcover_dir"]),
            raw_wetsnow_dir=Path(data["raw_wetsnow_dir"]),
            created_at=str(data.get("created_at", "")),
            subdomains=subs,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "SubdomainManifest":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

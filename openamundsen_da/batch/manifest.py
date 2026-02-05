"""Manifest helpers for batch subregion runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class WindowSpec:
    """Spatial window of a subregion relative to the global grid."""

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
class SubregionMeta:
    """Metadata for a prepared subregion setup."""

    id: str
    label: str
    setup_dir: Path
    config_path: Path
    grids_dir: Path
    meteo_dir: Path
    obs_dir: Path
    results_dir: Path
    roi_path: Path
    window: WindowSpec
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    crs: Optional[str]
    status: str = "pending"
    run_manifest: Optional[Path] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert Path objects to strings for JSON serialization
        for key in ("setup_dir", "config_path", "grids_dir", "meteo_dir", "obs_dir", "results_dir", "roi_path", "run_manifest"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        data["transform"] = list(self.transform)
        data["bounds"] = list(self.bounds)
        data["window"] = self.window.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SubregionMeta":
        return cls(
            id=str(data["id"]),
            label=str(data.get("label", data["id"])),
            setup_dir=Path(data["setup_dir"]),
            config_path=Path(data["config_path"]),
            grids_dir=Path(data["grids_dir"]),
            meteo_dir=Path(data["meteo_dir"]),
            obs_dir=Path(data["obs_dir"]),
            results_dir=Path(data["results_dir"]),
            roi_path=Path(data["roi_path"]),
            window=WindowSpec.from_dict(data["window"]),
            transform=tuple(float(x) for x in data["transform"]),
            bounds=tuple(float(x) for x in data["bounds"]),
            crs=(data.get("crs") if data.get("crs") not in (None, "None") else None),
            status=str(data.get("status", "pending")),
            run_manifest=(Path(data["run_manifest"]) if data.get("run_manifest") else None),
        )


@dataclass
class BatchManifest:
    """Top-level manifest describing a batch run and its subregions."""

    batch_name: str
    base_config: Path
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
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    subregions: Dict[str, SubregionMeta] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "batch_name": self.batch_name,
            "base_config": str(self.base_config),
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
            "created_at": self.created_at,
            "subregions": {k: v.to_dict() for k, v in self.subregions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchManifest":
        subs = {k: SubregionMeta.from_dict(v) for k, v in (data.get("subregions") or {}).items()}
        return cls(
            batch_name=str(data["batch_name"]),
            base_config=Path(data["base_config"]),
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
            created_at=str(data.get("created_at", "")),
            subregions=subs,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "BatchManifest":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

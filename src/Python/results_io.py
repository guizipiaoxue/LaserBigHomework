"""Portable result serialization for simulations and the report."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def copy_parameter_file(source: Path, destination: Path) -> None:
    """Store the exact YAML used for a run beside its derived results."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in keys})


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_directory(results_root: Path) -> Path:
    """Create a collision-safe timestamped directory under ``results/runs``."""
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp()
    candidate = runs_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = runs_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def publish_latest_manifest(results_root: Path, run_directory: Path,
                            run_manifest: dict) -> dict:
    """Merge a run into the root manifest using paths relative to results/."""
    import json

    root_manifest_path = results_root / "manifest.json"
    if root_manifest_path.exists():
        try:
            manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    else:
        manifest = {}
    manifest["generated_at_utc"] = run_manifest.get("generated_at_utc", utc_timestamp())
    manifest["config"] = run_manifest.get("config", {})
    manifest["latest_run"] = run_directory.relative_to(results_root).as_posix()
    # These aliases belonged to the removed browser viewer.  Keep the root
    # manifest focused on the canonical ``results`` entries and remove them
    # when an older manifest is reused.
    for obsolete_key in ("baseline_json", "baseline_images", "tilt_csv",
                         "tilt_angles_mrad", "tilt_images"):
        manifest.pop(obsolete_key, None)
    manifest.setdefault("results", {})
    run_prefix = run_directory.relative_to(results_root)
    for name, entry in run_manifest.get("results", {}).items():
        published = jsonable(entry)
        files = published.get("files") if isinstance(published, dict) else None
        if isinstance(files, dict):
            published["files"] = {
                key: (run_prefix / value).as_posix() if value else value
                for key, value in files.items()
            }
        manifest["results"][name] = published

    # Preserve the configuration that produced each result group. Older
    # manifests can recover it from that run's own manifest.json.
    for entry in manifest.get("results", {}).values():
        if not isinstance(entry, dict) or entry.get("parameters"):
            continue
        files = entry.get("files", {})
        for value in files.values() if isinstance(files, dict) else ():
            parts = Path(str(value)).parts
            if len(parts) < 3 or parts[0] != "runs":
                continue
            run_manifest_path = results_root.joinpath(*parts[:2], "manifest.json")
            try:
                historical = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if historical.get("config"):
                entry["parameters"] = historical["config"]
                break

    write_json(root_manifest_path, manifest)
    return manifest


def history_to_rows(history: list[dict]) -> list[dict]:
    return history


def save_field_npz(path: Path, *, x: np.ndarray, field_m1: np.ndarray,
                   field_m2: np.ndarray, y: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if y is None:
        np.savez_compressed(path, x=x, field_m1=field_m1, field_m2=field_m2)
    else:
        np.savez_compressed(path, x=x, y=y, field_m1=field_m1, field_m2=field_m2)

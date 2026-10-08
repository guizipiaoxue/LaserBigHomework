"""Run the required 1-D and 2-D baseline simulations.

Examples (from the repository root):
  python src/Python/run_baseline.py --mode 1d
  python src/Python/run_baseline.py --mode 2d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Python.cavity import Cavity1D, Cavity2D
    from Python.config_loader import DEFAULT_PARAMETER_PATH, load_parameters
    from Python.fox_li import make_initial_field_1d, make_initial_field_2d, run_fox_li
    from Python.grid import make_grid_1d, make_grid_2d
    from Python.metrics import hg00_1d, hg00_2d, metrics_1d, metrics_2d
    from Python.plots import plot_1d_fields, plot_2d_fields, plot_convergence
    from Python.propagation import fresnel_propagate_1d, fresnel_propagate_2d
    from Python.results_io import (create_run_directory, history_to_rows,
                                   publish_latest_manifest, save_field_npz,
                                   utc_timestamp, write_csv, write_json)
else:
    from .cavity import Cavity1D, Cavity2D
    from .config_loader import DEFAULT_PARAMETER_PATH, load_parameters
    from .fox_li import make_initial_field_1d, make_initial_field_2d, run_fox_li
    from .grid import make_grid_1d, make_grid_2d
    from .metrics import hg00_1d, hg00_2d, metrics_1d, metrics_2d
    from .plots import plot_1d_fields, plot_2d_fields, plot_convergence
    from .propagation import fresnel_propagate_1d, fresnel_propagate_2d
    from .results_io import (create_run_directory, history_to_rows,
                             publish_latest_manifest, save_field_npz,
                             utc_timestamp, write_csv, write_json)


def _summary_common(result) -> dict:
    return {
        "gamma_real": float(result.gamma.real),
        "gamma_imag": float(result.gamma.imag),
        "gamma_abs": float(abs(result.gamma)),
        "gamma_phase_rad": float(np.angle(result.gamma)),
        "rho": float(result.rho),
        "power_retention": float(abs(result.gamma) ** 2),
        "round_trip_loss": float(max(0.0, 1.0 - abs(result.gamma) ** 2)),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "stop_reason": result.stop_reason,
    }


def run_1d(parameters: dict, output_root: Path) -> dict:
    grid = make_grid_1d(parameters)
    cavity = Cavity1D(parameters, grid)
    seed_value = int(parameters["fox_li"]["random_seed"])
    initial = make_initial_field_1d(grid, parameters, seed_value)
    result = run_fox_li(cavity.round_trip, initial, parameters,
                        area_element=grid.dx)
    field_m1 = result.field
    field_m2 = cavity.mirror2(fresnel_propagate_1d(
        field_m1, grid, cavity.wavelength, cavity.length))
    reference = hg00_1d(grid.x, cavity.wavelength, cavity.length)
    summary = _summary_common(result)
    summary.update({"dimension": "1d", "n": grid.n, "dx_m": grid.dx,
                    "window_half_width_m": grid.half_width,
                    "aperture_half_width_m": grid.aperture_half_width,
                    "seed": seed_value,
                    "m1": metrics_1d(field_m1, grid.x, reference),
                    "m2": metrics_1d(field_m2, grid.x, reference)})
    out = output_root / "baseline" / "1d"
    out.mkdir(parents=True, exist_ok=True)
    save_field_npz(out / "fields.npz", x=grid.x, field_m1=field_m1, field_m2=field_m2)
    write_json(out / "summary.json", summary)
    write_csv(out / "history.csv", history_to_rows(result.history))
    plot_1d_fields(out / "fields.png", grid.x, field_m1, field_m2,
                   (f"1-D | lambda={cavity.wavelength * 1e9:.0f} nm, L=R1=R2={cavity.length:.3g} m, "
                    f"a={grid.aperture_half_width * 1e3:.3g} mm, Xmax={grid.half_width * 1e3:.3g} mm, "
                    f"N={grid.n}, theta=0, seed={seed_value}"),
                   float(parameters["fox_li"]["intensity_floor"]))
    plot_convergence(
        out / "convergence.png", result.history,
        (f"1-D Fox-Li convergence | lambda={cavity.wavelength * 1e9:.0f} nm, "
         f"L={cavity.length:.3g} m, R1={cavity.r1:.3g} m, R2={cavity.r2:.3g} m, "
         f"a={grid.aperture_half_width * 1e3:.3g} mm, Xmax={grid.half_width * 1e3:.3g} mm, "
         f"N={grid.n}, theta=0 mrad, seed={seed_value}"))
    return {"summary": summary, "parameters": parameters,
            "files": {"fields": "baseline/1d/fields.npz", "summary": "baseline/1d/summary.json",
                      "history": "baseline/1d/history.csv", "fields_png": "baseline/1d/fields.png",
                      "convergence_png": "baseline/1d/convergence.png"}}


def run_2d(parameters: dict, output_root: Path) -> dict:
    grid = make_grid_2d(parameters)
    cavity = Cavity2D(parameters, grid)
    seed_value = int(parameters["fox_li"]["random_seed"])
    initial = make_initial_field_2d(grid, parameters, seed_value)
    result = run_fox_li(cavity.round_trip, initial, parameters,
                        area_element=grid.dx * grid.dy)
    field_m1 = result.field
    field_m2 = cavity.mirror2(fresnel_propagate_2d(
        field_m1, grid, cavity.wavelength, cavity.length))
    reference = hg00_2d(grid.x, grid.y, cavity.wavelength, cavity.length)
    summary = _summary_common(result)
    summary.update({"dimension": "2d", "n": grid.n, "dx_m": grid.dx, "dy_m": grid.dy,
                    "window_half_width_m": grid.half_width,
                    "aperture_half_width_m": grid.aperture_half_width,
                    "seed": seed_value,
                    "m1": metrics_2d(field_m1, grid.x, grid.y, reference),
                    "m2": metrics_2d(field_m2, grid.x, grid.y, reference)})
    out = output_root / "baseline" / "2d"
    out.mkdir(parents=True, exist_ok=True)
    save_field_npz(out / "fields.npz", x=grid.x, y=grid.y, field_m1=field_m1, field_m2=field_m2)
    write_json(out / "summary.json", summary)
    write_csv(out / "history.csv", history_to_rows(result.history))
    plot_2d_fields(out / "fields.png", grid.x, grid.y, field_m1, field_m2,
                   (f"2-D | lambda={cavity.wavelength * 1e9:.0f} nm, L=R1=R2={cavity.length:.3g} m, "
                    f"a={grid.aperture_half_width * 1e3:.3g} mm, Xmax={grid.half_width * 1e3:.3g} mm, "
                    f"N={grid.n}, theta=0, seed={seed_value}"),
                   float(parameters["fox_li"]["intensity_floor"]))
    plot_convergence(
        out / "convergence.png", result.history,
        (f"2-D Fox-Li convergence | lambda={cavity.wavelength * 1e9:.0f} nm, "
         f"L={cavity.length:.3g} m, R1={cavity.r1:.3g} m, R2={cavity.r2:.3g} m, "
         f"a={grid.aperture_half_width * 1e3:.3g} mm, Xmax={grid.half_width * 1e3:.3g} mm, "
         f"N={grid.n}, theta=0 mrad, seed={seed_value}"))
    return {"summary": summary, "parameters": parameters,
            "files": {"fields": "baseline/2d/fields.npz", "summary": "baseline/2d/summary.json",
                      "history": "baseline/2d/history.csv", "fields_png": "baseline/2d/fields.png",
                      "convergence_png": "baseline/2d/convergence.png"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("1d", "2d", "both"), default="both")
    args = parser.parse_args()
    parameters = load_parameters()
    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / str(parameters["output"]["root"])
    output_root = create_run_directory(results_root)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    copyfile(DEFAULT_PARAMETER_PATH, output_root / "parameter_used.yaml")
    manifest = {"generated_at_utc": utc_timestamp(), "config": parameters, "results": {}}
    if args.mode in ("1d", "both"):
        print("Running 1-D Fox-Li baseline ...", flush=True)
        manifest["results"]["baseline_1d"] = run_1d(parameters, output_root)
    if args.mode in ("2d", "both"):
        print("Running 2-D Fox-Li baseline ...", flush=True)
        manifest["results"]["baseline_2d"] = run_2d(parameters, output_root)
    write_json(output_root / "manifest.json", manifest)
    publish_latest_manifest(results_root.resolve(), output_root, manifest)
    print(f"Results written to {output_root}")


if __name__ == "__main__":
    main()

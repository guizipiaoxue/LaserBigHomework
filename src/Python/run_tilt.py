"""Run the advanced single-mirror tilt scan configured in parameter.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Python.cavity import Cavity2D
    from Python.config_loader import DEFAULT_PARAMETER_PATH, load_parameters
    from Python.fox_li import make_initial_field_2d, run_fox_li
    from Python.grid import make_grid_2d
    from Python.metrics import hg00_2d, metrics_2d
    from Python.plots import plot_2d_fields, plot_tilt_scan, plot_convergence
    from Python.propagation import fresnel_propagate_2d
    from Python.results_io import (create_run_directory, publish_latest_manifest,
                                   save_field_npz, utc_timestamp, write_csv, write_json)
else:
    from .cavity import Cavity2D
    from .config_loader import DEFAULT_PARAMETER_PATH, load_parameters
    from .fox_li import make_initial_field_2d, run_fox_li
    from .grid import make_grid_2d
    from .metrics import hg00_2d, metrics_2d
    from .plots import plot_2d_fields, plot_tilt_scan, plot_convergence
    from .propagation import fresnel_propagate_2d
    from .results_io import (create_run_directory, publish_latest_manifest,
                             save_field_npz, utc_timestamp, write_csv, write_json)


def run_tilt(parameters: dict, output_root: Path) -> dict:
    grid = make_grid_2d(parameters)
    project = parameters["project"]
    wavelength = float(project["wavelength_m"])
    length = float(project["cavity_length_m"])
    angles = [float(v) for v in parameters["tilt"]["angles_mrad"]]
    if not angles:
        raise ValueError("tilt angle list must not be empty")
    k = 2.0 * np.pi / wavelength
    max_angle_mrad = max(abs(float(value)) for value in angles)
    max_phase_step = 2.0 * k * grid.dy * abs(np.sin(max_angle_mrad * 1e-3))
    # The mirror phase must be sampled with at least two points per 2*pi cycle.
    if max_phase_step >= np.pi:
        raise ValueError(
            "tilt phase is undersampled: 2*k*dy*|sin(theta_max)| must be < pi "
            f"(got {max_phase_step:.3g})")
    seed_value = int(parameters["fox_li"]["random_seed"])
    reference = hg00_2d(grid.x, grid.y, wavelength, length)
    previous_field = None
    zero_reference = None
    zero_reference_m2 = None
    rows: list[dict] = []
    histories: list[dict] = []
    fields_by_angle: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def geometric_initial(theta: float, mirror: str) -> np.ndarray:
        """Seed the first tilted solve with the paraxial fixed-point Gaussian.

        A centered zero-angle field is a poor seed in a confocal cavity: the
        tilted operator can exchange two nearly degenerate modes, producing a
        harmless but very slow two-cycle in the complex field.  The ray fixed
        point gives M1 displacement ``L*theta`` for M2 tilt and the opposite
        reference-plane arrangement for M1 tilt.
        """
        xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
        wm = np.sqrt(wavelength * length / np.pi)
        signed = float(np.sin(theta))
        if mirror == "M2":
            y0, alpha = length * signed, -signed
        else:
            y0, alpha = 0.0, signed
        radius2 = xx**2 + (yy - y0)**2
        field = (np.exp(-radius2 / wm**2) *
                 np.exp(-1j * k * radius2 / (2.0 * length)) *
                 np.exp(1j * k * alpha * yy))
        norm = np.linalg.norm(field.ravel())
        return field / norm

    out = output_root / "tilt"
    out.mkdir(parents=True, exist_ok=True)
    for index, theta_mrad in enumerate(angles):
        theta = float(theta_mrad) * 1e-3
        cavity = Cavity2D(parameters, grid, theta=theta)
        previous_angle = float(angles[index - 1]) if index else None
        needs_geometry_seed = (previous_field is None or
                               (previous_angle is not None and
                                abs(previous_angle) < 1e-15 and
                                abs(float(theta_mrad)) >= 1e-15))
        if (previous_field is not None and
                bool(parameters["tilt"]["warm_start"]) and
                not needs_geometry_seed):
            initial = previous_field
            initialization = "warm_start_previous_angle"
            actual_seed = ""
        elif abs(float(theta_mrad)) >= 1e-15:
            initial = geometric_initial(theta, str(parameters["tilt"]["mirror"]))
            initialization = "paraxial_geometric_fixed_point"
            actual_seed = ""
        else:
            actual_seed = seed_value + index
            initial = make_initial_field_2d(grid, parameters, actual_seed)
            initialization = str(parameters["fox_li"]["initializer"])
        result = run_fox_li(cavity.round_trip, initial, parameters,
                            area_element=grid.dx * grid.dy)
        previous_field = result.field
        field_m1 = result.field
        field_m2 = cavity.mirror2(fresnel_propagate_2d(
            field_m1, grid, wavelength, length))
        fields_by_angle[f"{theta_mrad:g}"] = (field_m1, field_m2)
        if abs(float(theta_mrad)) < 1e-15:
            zero_reference = field_m1.copy()
            zero_reference_m2 = field_m2.copy()
        # Zero-angle references may not exist yet when a custom scan starts
        # at a nonzero angle; eta_0 is filled in a second pass below.
        m1 = metrics_2d(field_m1, grid.x, grid.y, reference)
        m2 = metrics_2d(field_m2, grid.x, grid.y, reference)
        row = {
            "theta_mrad": float(theta_mrad),
            "theta_rad": theta,
            "seed": actual_seed,
            "initialization": initialization,
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "gamma_abs_M1": float(abs(result.gamma)),
            "gamma_phase_M1_rad": float(np.angle(result.gamma)),
            "loss_M1": float(max(0.0, 1.0 - abs(result.gamma) ** 2)),
            # The round-trip eigenvalue is invariant to the reference plane;
            # retaining the same value for M2 makes the table explicit while
            # the two fields still expose plane-dependent centroids and widths.
            "loss_M2": float(max(0.0, 1.0 - abs(result.gamma) ** 2)),
            "y_centroid_M1_mm": m1["y_centroid_m"] * 1e3,
            "y_centroid_M2_mm": m2["y_centroid_m"] * 1e3,
            "x_centroid_M1_mm": m1["x_centroid_m"] * 1e3,
            "x_centroid_M2_mm": m2["x_centroid_m"] * 1e3,
            "mode_radius_y_M1_mm": m1["mode_radius_y_m"] * 1e3,
            "mode_radius_y_M2_mm": m2["mode_radius_y_m"] * 1e3,
            "hg00_overlap_M1": m1["hg00_overlap"],
            "hg00_overlap_M2": m2["hg00_overlap"],
            "zero_angle_overlap_M1": "",
            "zero_angle_overlap_M2": "",
            "asymmetry_M1": m1["y_asymmetry"],
            "asymmetry_M2": m2["y_asymmetry"],
            "stop_reason": result.stop_reason,
        }
        rows.append(row)
        histories.extend({"theta_mrad": float(theta_mrad), **history_row}
                         for history_row in result.history)
        save_field_npz(out / f"fields_theta_{theta_mrad:g}mrad.npz", x=grid.x, y=grid.y,
                       field_m1=field_m1, field_m2=field_m2)
        if index in (0, len(angles) // 2, len(angles) - 1):
            plot_2d_fields(out / f"fields_theta_{theta_mrad:g}mrad.png", grid.x, grid.y,
                           field_m1, field_m2,
                           (f"Tilt | {parameters['tilt']['mirror']} tilted, lambda={wavelength * 1e9:.0f} nm, L=R1=R2={length:.3g} m, "
                            f"a={grid.aperture_half_width * 1e3:.3g} mm, Xmax={grid.half_width * 1e3:.3g} mm, "
                            f"N={grid.n}, theta={theta_mrad:g} mrad, init={initialization}"),
                           float(parameters["fox_li"]["intensity_floor"]))
            plot_convergence(out / f"convergence_theta_{theta_mrad:g}mrad.png",
                             result.history,
                             (f"Fox-Li convergence | lambda={wavelength * 1e9:.0f} nm, "
                              f"L={length:.3g} m, R1={project['mirror1_radius_m']:.3g} m, "
                              f"R2={project['mirror2_radius_m']:.3g} m, "
                              f"a={grid.aperture_half_width * 1e3:.3g} mm, "
                              f"Xmax={grid.half_width * 1e3:.3g} mm, N={grid.n}, "
                              f"theta={theta_mrad:g} mrad, "
                              f"seed={actual_seed if actual_seed != '' else 'geometric init'}"))
    # A custom angle list may place zero after the first point.  Fill the
    # reference-overlap column once the zero-angle eigenfield is available.
    if zero_reference is None and any(abs(float(a)) < 1e-15 for a in angles):
        zero_fields = fields_by_angle.get("0")
        if zero_fields is not None:
            zero_reference, zero_reference_m2 = zero_fields
    if zero_reference is not None:
        for row in rows:
            key = f"{float(row['theta_mrad']):g}"
            fields = fields_by_angle.get(key)
            if fields is None:
                continue
            row["zero_angle_overlap_M1"] = metrics_2d(
                fields[0], grid.x, grid.y, reference, zero_reference)["zero_angle_overlap"]
            row["zero_angle_overlap_M2"] = metrics_2d(
                fields[1], grid.x, grid.y, reference, zero_reference_m2)["zero_angle_overlap"]
    write_csv(out / "scan.csv", rows)
    write_csv(out / "history.csv", histories)
    fit_limit = float(parameters["tilt"]["linear_fit_max_mrad"])
    fit = {}
    for plane in ("M1", "M2"):
        key = f"y_centroid_{plane}_mm"
        small = []
        for row in rows:
            try:
                theta_value = float(row["theta_mrad"])
                centroid = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if (bool(row["converged"]) and abs(theta_value) <= fit_limit
                    and np.isfinite(theta_value) and np.isfinite(centroid)):
                small.append((theta_value, centroid))
        if len(small) < 2:
            continue
        x = np.array([point[0] for point in small])
        y = np.array([point[1] for point in small])
        if np.unique(x).size < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        scale = max(1.0, float(np.max(np.abs(y))))
        r_squared = (float(1.0 - ss_res / ss_tot)
                     if ss_tot > np.finfo(float).eps * scale**2 else None)
        fit[plane] = {"slope_mm_per_mrad": float(slope),
                      "intercept_mm": float(intercept),
                      "r2": r_squared,
                      "max_abs_theta_mrad": fit_limit,
                      "n": len(small)}
    plot_tilt_scan(out / "scan.png", rows, "Single-mirror tilt scan", fit=fit)
    summary = {"dimension": "2d", "mirror": str(parameters["tilt"]["mirror"]),
               "angles_mrad": angles, "n": grid.n, "seed": seed_value,
               "phase_step_max_rad": float(max_phase_step), "linear_fit": fit, "rows": rows}
    write_json(out / "parameters_used.json", {
        "parameters": parameters,
        "effective_seed": seed_value,
        "effective_angles_mrad": [float(value) for value in angles],
        "phase_step_max_rad": float(max_phase_step),
    })
    write_json(out / "summary.json", summary)
    return {"summary": summary, "parameters": parameters,
            "files": {"scan": "tilt/scan.csv", "history": "tilt/history.csv",
                      "summary": "tilt/summary.json", "plot": "tilt/scan.png"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parameters = load_parameters()
    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / str(parameters["output"]["root"])
    output_root = create_run_directory(results_root)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    copyfile(DEFAULT_PARAMETER_PATH, output_root / "parameter_used.yaml")
    print("Running 2-D single-mirror tilt scan ...", flush=True)
    result = run_tilt(parameters, output_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"generated_at_utc": utc_timestamp(), "config": parameters, "results": {}}
    manifest["config"] = parameters
    manifest.setdefault("results", {})["tilt"] = result
    write_json(manifest_path, manifest)
    publish_latest_manifest(results_root.resolve(), output_root, manifest)
    print(f"Results written to {output_root}")


if __name__ == "__main__":
    main()

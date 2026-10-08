"""Load the single YAML parameter source used by every simulation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PARAMETER_PATH = Path(__file__).resolve().parent / "config" / "parameter.yaml"


def load_parameters() -> dict[str, Any]:
    """Read and validate the YAML parameter file.

    Keeping this validation close to the loader makes a run fail early when a
    physical value is missing, while still leaving all values editable in YAML.
    """
    parameter_path = DEFAULT_PARAMETER_PATH
    if not parameter_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {parameter_path}")
    with parameter_path.open("r", encoding="utf-8") as stream:
        parameters = yaml.safe_load(stream) or {}
    for section in ("project", "grid", "fox_li", "tilt", "output"):
        if section not in parameters:
            raise ValueError(f"Missing section '{section}' in {parameter_path}")
    required = {
        "project": ("name", "wavelength_m", "cavity_length_m",
                    "mirror1_radius_m", "mirror2_radius_m"),
        "grid": ("one_d", "two_d", "propagation_padding_factor"),
        "fox_li": ("max_iterations", "convergence_threshold",
                   "eigenvalue_relative_threshold", "consecutive_converged",
                   "intensity_floor", "initializer", "random_seed"),
        "tilt": ("mirror", "angles_mrad", "warm_start", "linear_fit_max_mrad"),
        "output": ("root",),
    }
    for section, keys in required.items():
        missing = [key for key in keys if key not in parameters[section]]
        if missing:
            raise ValueError(f"Missing {section} parameter(s): {', '.join(missing)}")
    project = parameters["project"]
    for key in ("wavelength_m", "cavity_length_m", "mirror1_radius_m", "mirror2_radius_m"):
        if float(project[key]) <= 0:
            raise ValueError(f"project.{key} must be positive")
    length = float(project["cavity_length_m"])
    for key in ("mirror1_radius_m", "mirror2_radius_m"):
        if abs(float(project[key]) - length) > 1e-9 * length:
            raise ValueError(f"project.{key} must equal cavity_length_m for this confocal model")
    if int(parameters["fox_li"]["max_iterations"]) < 1:
        raise ValueError("fox_li.max_iterations must be at least one")
    if float(parameters["fox_li"]["convergence_threshold"]) <= 0:
        raise ValueError("fox_li.convergence_threshold must be positive")
    if float(parameters["fox_li"]["eigenvalue_relative_threshold"]) <= 0:
        raise ValueError("fox_li.eigenvalue_relative_threshold must be positive")
    if int(parameters["fox_li"]["consecutive_converged"]) < 1:
        raise ValueError("fox_li.consecutive_converged must be at least one")
    if str(parameters["fox_li"]["initializer"]).lower() not in ("gaussian", "random"):
        raise ValueError("fox_li.initializer must be 'gaussian' or 'random'")
    for section in ("one_d", "two_d"):
        grid = parameters["grid"][section]
        for key in ("n", "window_half_width_factor", "aperture_half_width_factor"):
            if key not in grid:
                raise ValueError(f"Missing grid.{section}.{key}")
        if int(grid["n"]) < 8:
            raise ValueError(f"grid.{section}.n must be at least 8")
        if float(grid["window_half_width_factor"]) <= 0:
            raise ValueError(f"grid.{section}.window_half_width_factor must be positive")
        if float(grid["aperture_half_width_factor"]) <= 0:
            raise ValueError(f"grid.{section}.aperture_half_width_factor must be positive")
    if int(parameters["grid"]["propagation_padding_factor"]) < 1:
        raise ValueError("grid.propagation_padding_factor must be at least one")
    if not parameters["tilt"]["angles_mrad"]:
        raise ValueError("tilt.angles_mrad must not be empty")
    mirror = str(parameters["tilt"]["mirror"]).upper()
    if mirror not in ("M1", "M2"):
        raise ValueError("tilt.mirror must be 'M1' or 'M2'")
    parameters["tilt"]["mirror"] = mirror
    if float(parameters["tilt"]["linear_fit_max_mrad"]) < 0:
        raise ValueError("tilt.linear_fit_max_mrad must be non-negative")
    return parameters

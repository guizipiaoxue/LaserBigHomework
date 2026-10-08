"""Fox-Li power iteration with phase-aware convergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class FoxLiResult:
    field: np.ndarray
    gamma: complex
    rho: float
    iterations: int
    converged: bool
    stop_reason: str
    history: list[dict]


def _normalise(field: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(field.ravel()))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Fox-Li field became zero or non-finite")
    return field / norm


def _initial_field(shape: tuple[int, ...], coordinates: tuple[np.ndarray, ...],
                   parameters: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    initializer = str(parameters["fox_li"]["initializer"]).lower()
    if initializer == "random":
        field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    else:
        # Seed with the analytically matched confocal HG00 field.  A purely
        # real broad Gaussian has components in several nearly degenerate
        # confocal modes, so power iteration can appear not to converge.
        radius2 = sum(np.asarray(c) ** 2 for c in coordinates)
        wavelength = float(parameters["project"]["wavelength_m"])
        length = float(parameters["project"]["cavity_length_m"])
        wm = np.sqrt(wavelength * length / np.pi)
        k = 2.0 * np.pi / wavelength
        field = np.exp(-radius2 / wm**2) * np.exp(-1j * k * radius2 / (2.0 * length))
        field = field.astype(complex)
    return _normalise(field)


def run_fox_li(round_trip: Callable[[np.ndarray], np.ndarray],
               initial_field: np.ndarray, parameters: dict,
               *, area_element: float = 1.0) -> FoxLiResult:
    """Iterate a round-trip operator and return the converged normalized field.

    ``area_element`` appears in inner products, but cancels in Rayleigh ratios;
    it is retained to make the reported definitions match the continuous model.
    """
    settings = parameters["fox_li"]
    max_iterations = int(settings["max_iterations"])
    threshold = float(settings["convergence_threshold"])
    eig_threshold = float(settings["eigenvalue_relative_threshold"])
    required = int(settings["consecutive_converged"])
    epsilon = np.finfo(float).eps

    field = _normalise(np.asarray(initial_field, dtype=complex))
    history: list[dict] = []
    previous_gamma: complex | None = None
    stable_count = 0
    gamma = complex(np.nan, np.nan)
    rho = float("nan")
    stop_reason = "maximum_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        raw_next = round_trip(field)
        raw_norm = float(np.linalg.norm(raw_next.ravel()))
        old_norm = float(np.linalg.norm(field.ravel()))
        gamma = complex(np.vdot(field, raw_next) * area_element /
                        (np.vdot(field, field) * area_element))
        rho = raw_norm / max(old_norm, epsilon)
        next_field = _normalise(raw_next)

        overlap = np.vdot(field, next_field) * area_element
        phase_factor = overlap / abs(overlap) if abs(overlap) > epsilon else 1.0 + 0.0j
        delta = float(np.linalg.norm(next_field - phase_factor * field) /
                      max(np.linalg.norm(next_field), epsilon))
        old_intensity = np.abs(field) ** 2
        new_intensity = np.abs(next_field) ** 2
        d_intensity = float(np.linalg.norm((new_intensity - old_intensity).ravel()) /
                            max(np.linalg.norm(new_intensity.ravel()), epsilon))
        if previous_gamma is None:
            d_gamma = float("inf")
        else:
            d_gamma = float(abs(gamma - previous_gamma) /
                            max(abs(gamma), epsilon))
        row = {
            "iteration": iteration,
            "delta": delta,
            "intensity_difference": d_intensity,
            "gamma_real": float(gamma.real),
            "gamma_imag": float(gamma.imag),
            "gamma_abs": float(abs(gamma)),
            "gamma_phase": float(np.angle(gamma)),
            "rho": float(rho),
            "power_retention": float(abs(gamma) ** 2),
            "loss": float(max(0.0, 1.0 - abs(gamma) ** 2)),
            "eigenvalue_difference": d_gamma,
        }
        history.append(row)
        if delta < threshold and d_intensity < threshold and d_gamma < eig_threshold:
            stable_count += 1
        else:
            stable_count = 0
        field = next_field
        previous_gamma = gamma
        if stable_count >= required:
            stop_reason = f"all criteria met for {required} consecutive iterations"
            return FoxLiResult(field=field, gamma=gamma, rho=rho, iterations=iteration,
                               converged=True, stop_reason=stop_reason,
                               history=history)
    return FoxLiResult(field=field, gamma=gamma, rho=rho, iterations=max_iterations,
                       converged=False,
                       stop_reason=stop_reason, history=history)


def make_initial_field_1d(grid, parameters: dict, seed: int | None = None) -> np.ndarray:
    actual_seed = int(parameters["fox_li"]["random_seed"] if seed is None else seed)
    return _initial_field((grid.n,), (grid.x,), parameters, actual_seed)


def make_initial_field_2d(grid, parameters: dict, seed: int | None = None) -> np.ndarray:
    actual_seed = int(parameters["fox_li"]["random_seed"] if seed is None else seed)
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    return _initial_field((grid.n, grid.n), (xx, yy), parameters, actual_seed)

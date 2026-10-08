"""Mode metrics and Hermite-Gaussian reference fields."""

from __future__ import annotations

import numpy as np


def _normalised_overlap(a: np.ndarray, b: np.ndarray, area_element: float = 1.0) -> float:
    numerator = abs(np.vdot(a, b) * area_element) ** 2
    denominator = float(np.vdot(a, a).real * np.vdot(b, b).real * area_element**2)
    return float(numerator / denominator) if denominator else 0.0


def hg00_1d(x: np.ndarray, wavelength: float, length: float) -> np.ndarray:
    wm = np.sqrt(wavelength * length / np.pi)
    k = 2.0 * np.pi / wavelength
    return np.exp(-(x**2) / wm**2) * np.exp(-1j * k * x**2 / (2.0 * length))


def hg00_2d(x: np.ndarray, y: np.ndarray, wavelength: float, length: float) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    wm = np.sqrt(wavelength * length / np.pi)
    k = 2.0 * np.pi / wavelength
    return np.exp(-(xx**2 + yy**2) / wm**2) * np.exp(-1j * k * (xx**2 + yy**2) / (2.0 * length))


def mode_overlap(field: np.ndarray, reference: np.ndarray, area_element: float = 1.0) -> float:
    return _normalised_overlap(field, reference, area_element)


def metrics_1d(field: np.ndarray, x: np.ndarray, reference: np.ndarray | None = None) -> dict:
    intensity = np.abs(field) ** 2
    total = float(np.sum(intensity))
    centroid = float(np.sum(x * intensity) / total) if total else 0.0
    variance = float(np.sum((x - centroid) ** 2 * intensity) / total) if total else 0.0
    result = {"x_centroid_m": centroid, "sigma_x_m": float(np.sqrt(max(variance, 0.0))),
              "mode_radius_x_m": float(2.0 * np.sqrt(max(variance, 0.0)))}
    if reference is not None:
        result["hg00_overlap"] = mode_overlap(field, reference)
    return result


def metrics_2d(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                reference: np.ndarray | None = None,
                reference_zero: np.ndarray | None = None) -> dict:
    intensity = np.abs(field) ** 2
    total = float(np.sum(intensity))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xbar = float(np.sum(xx * intensity) / total) if total else 0.0
    ybar = float(np.sum(yy * intensity) / total) if total else 0.0
    sx = float(np.sqrt(max(np.sum((xx - xbar) ** 2 * intensity) / total, 0.0))) if total else 0.0
    sy = float(np.sqrt(max(np.sum((yy - ybar) ** 2 * intensity) / total, 0.0))) if total else 0.0
    # Coordinates use x[0] = -Xmax and x[N/2] = 0; reflect about y=0.
    flipped = np.roll(np.flip(intensity, axis=0), 1, axis=0)
    asymmetry = float(np.sum(np.abs(intensity - flipped)) / total) if total else 0.0
    result = {"x_centroid_m": xbar, "y_centroid_m": ybar, "sigma_x_m": sx,
              "sigma_y_m": sy, "mode_radius_x_m": 2.0 * sx, "mode_radius_y_m": 2.0 * sy,
              "y_asymmetry": asymmetry}
    if reference is not None:
        result["hg00_overlap"] = mode_overlap(field, reference)
    if reference_zero is not None:
        result["zero_angle_overlap"] = mode_overlap(field, reference_zero)
    return result

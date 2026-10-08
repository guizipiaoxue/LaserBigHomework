"""Coordinate and spatial-frequency grids."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid1D:
    x: np.ndarray
    fx: np.ndarray
    dx: float
    half_width: float
    aperture_half_width: float
    n: int
    padding_factor: int = 2


@dataclass(frozen=True)
class Grid2D:
    x: np.ndarray
    y: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    dx: float
    dy: float
    half_width: float
    aperture_half_width: float
    n: int
    padding_factor: int = 2


def _field_radius(parameters: dict) -> float:
    wavelength = float(parameters["project"]["wavelength_m"])
    length = float(parameters["project"]["cavity_length_m"])
    # At a confocal mirror w_m^2 = lambda L / pi.
    return float(np.sqrt(wavelength * length / np.pi))


def make_grid_1d(parameters: dict) -> Grid1D:
    settings = parameters["grid"]["one_d"]
    n = int(settings["n"])
    wm = _field_radius(parameters)
    half_width = float(settings["window_half_width_factor"]) * wm
    aperture = float(settings["aperture_half_width_factor"]) * wm
    dx = 2.0 * half_width / n
    x = (np.arange(n) - n / 2.0) * dx
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    padding = int(parameters["grid"]["propagation_padding_factor"])
    return Grid1D(x=x, fx=fx, dx=dx, half_width=half_width,
                  aperture_half_width=aperture, n=n, padding_factor=max(1, padding))


def make_grid_2d(parameters: dict) -> Grid2D:
    settings = parameters["grid"]["two_d"]
    n = int(settings["n"])
    wm = _field_radius(parameters)
    half_width = float(settings["window_half_width_factor"]) * wm
    aperture = float(settings["aperture_half_width_factor"]) * wm
    dx = 2.0 * half_width / n
    x = (np.arange(n) - n / 2.0) * dx
    y = x.copy()
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    padding = int(parameters["grid"]["propagation_padding_factor"])
    return Grid2D(x=x, y=y, fx=f, fy=f.copy(), dx=dx, dy=dx,
                  half_width=half_width, aperture_half_width=aperture, n=n,
                  padding_factor=max(1, padding))


def aperture_1d(grid: Grid1D) -> np.ndarray:
    return (np.abs(grid.x) <= grid.aperture_half_width).astype(float)


def aperture_2d(grid: Grid2D) -> np.ndarray:
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    return ((np.abs(xx) <= grid.aperture_half_width)
            & (np.abs(yy) <= grid.aperture_half_width)).astype(float)

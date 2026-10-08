"""Mirror and round-trip operators for the confocal cavity."""

from __future__ import annotations

import numpy as np

from .grid import Grid1D, Grid2D, aperture_1d, aperture_2d
from .propagation import fresnel_propagate_1d, fresnel_propagate_2d


class Cavity1D:
    def __init__(self, parameters: dict, grid: Grid1D, theta: float = 0.0):
        project = parameters["project"]
        self.wavelength = float(project["wavelength_m"])
        self.length = float(project["cavity_length_m"])
        self.r1 = float(project["mirror1_radius_m"])
        self.r2 = float(project["mirror2_radius_m"])
        self.k = 2.0 * np.pi / self.wavelength
        self.grid = grid
        self.mask = aperture_1d(grid)
        self.theta = float(theta)
        if abs(self.theta) > 0.0:
            raise ValueError(
                "The 1-D x-only model cannot represent rotation about x, "
                "which adds a phase ramp along y; use Cavity2D for mirror tilt."
            )

    def mirror1(self, field: np.ndarray) -> np.ndarray:
        return self.mask * np.exp(-1j * self.k * self.grid.x**2 / self.r1) * field

    def mirror2(self, field: np.ndarray) -> np.ndarray:
        # Rotation about x adds a phase ramp along y, absent from this 1-D model.
        return self.mask * np.exp(-1j * self.k * self.grid.x**2 / self.r2) * field

    def round_trip(self, field: np.ndarray) -> np.ndarray:
        return self.mirror1(fresnel_propagate_1d(self.mirror2(
            fresnel_propagate_1d(field, self.grid, self.wavelength, self.length)),
            self.grid, self.wavelength, self.length))


class Cavity2D:
    def __init__(self, parameters: dict, grid: Grid2D, theta: float = 0.0):
        project = parameters["project"]
        self.wavelength = float(project["wavelength_m"])
        self.length = float(project["cavity_length_m"])
        self.r1 = float(project["mirror1_radius_m"])
        self.r2 = float(project["mirror2_radius_m"])
        self.k = 2.0 * np.pi / self.wavelength
        self.grid = grid
        self.mask = aperture_2d(grid)
        self.theta = float(theta)
        self.tilt_mirror = str(parameters["tilt"]["mirror"]).upper()
        self.xx, self.yy = np.meshgrid(grid.x, grid.y, indexing="xy")

    def _tilt_factor(self) -> np.ndarray:
        return np.exp(1j * 2.0 * self.k * self.yy * np.sin(self.theta))

    def mirror1(self, field: np.ndarray) -> np.ndarray:
        phase = np.exp(-1j * self.k * (self.xx**2 + self.yy**2) / self.r1)
        if self.tilt_mirror == "M1":
            phase = phase * self._tilt_factor()
        return self.mask * phase * field

    def mirror2(self, field: np.ndarray) -> np.ndarray:
        phase = np.exp(-1j * self.k * (self.xx**2 + self.yy**2) / self.r2)
        if self.tilt_mirror == "M2":
            phase = phase * self._tilt_factor()
        return self.mask * phase * field

    def round_trip(self, field: np.ndarray) -> np.ndarray:
        after_m2 = self.mirror2(fresnel_propagate_2d(
            field, self.grid, self.wavelength, self.length))
        return self.mirror1(fresnel_propagate_2d(
            after_m2, self.grid, self.wavelength, self.length))

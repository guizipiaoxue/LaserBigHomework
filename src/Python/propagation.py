"""Fresnel propagation using an FFT transfer function."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .grid import Grid1D, Grid2D


def _centered_fft(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(values, axes=axes), axes=axes, norm="ortho"), axes=axes)


def _centered_ifft(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(values, axes=axes), axes=axes, norm="ortho"), axes=axes)


def fresnel_transfer_1d(grid: Grid1D, wavelength: float, length: float) -> np.ndarray:
    k = 2.0 * np.pi / wavelength
    return np.exp(1j * k * length) * np.exp(-1j * np.pi * wavelength * length * grid.fx**2)


def fresnel_transfer_2d(grid: Grid2D, wavelength: float, length: float) -> np.ndarray:
    k = 2.0 * np.pi / wavelength
    fx, fy = np.meshgrid(grid.fx, grid.fy, indexing="xy")
    return np.exp(1j * k * length) * np.exp(-1j * np.pi * wavelength * length * (fx**2 + fy**2))


@lru_cache(maxsize=16)
def _padded_transfer_1d(n: int, dx: float, wavelength: float, length: float) -> np.ndarray:
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    k = 2.0 * np.pi / wavelength
    return np.exp(1j * k * length) * np.exp(-1j * np.pi * wavelength * length * fx**2)


@lru_cache(maxsize=4)
def _padded_transfer_2d(n: int, dx: float, wavelength: float, length: float) -> np.ndarray:
    frequency = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    fx, fy = np.meshgrid(frequency, frequency, indexing="xy")
    k = 2.0 * np.pi / wavelength
    return np.exp(1j * k * length) * np.exp(
        -1j * np.pi * wavelength * length * (fx**2 + fy**2))


def fresnel_propagate_1d(field: np.ndarray, grid: Grid1D, wavelength: float, length: float) -> np.ndarray:
    # Zero-padding keeps the circular FFT convolution from wrapping the beam
    # around the finite computational window. The original dx is retained.
    pad_factor = max(1, int(getattr(grid, "padding_factor", 2)))
    n_pad = max(grid.n, int(pad_factor * grid.n))
    if n_pad == grid.n:
        transfer = fresnel_transfer_1d(grid, wavelength, length)
        return _centered_ifft(_centered_fft(field, axes=(0,)) * transfer, axes=(0,))
    padded = np.zeros(n_pad, dtype=complex)
    start = (n_pad - grid.n) // 2
    padded[start:start + grid.n] = field
    transfer = _padded_transfer_1d(n_pad, grid.dx, wavelength, length)
    propagated = _centered_ifft(_centered_fft(padded, axes=(0,)) * transfer, axes=(0,))
    return propagated[start:start + grid.n]


def fresnel_propagate_2d(field: np.ndarray, grid: Grid2D, wavelength: float, length: float) -> np.ndarray:
    pad_factor = max(1, int(getattr(grid, "padding_factor", 2)))
    n_pad = max(grid.n, int(pad_factor * grid.n))
    if n_pad == grid.n:
        transfer = fresnel_transfer_2d(grid, wavelength, length)
        return _centered_ifft(_centered_fft(field, axes=(0, 1)) * transfer, axes=(0, 1))
    padded = np.zeros((n_pad, n_pad), dtype=complex)
    start = (n_pad - grid.n) // 2
    padded[start:start + grid.n, start:start + grid.n] = field
    transfer = _padded_transfer_2d(n_pad, grid.dx, wavelength, length)
    propagated = _centered_ifft(_centered_fft(padded, axes=(0, 1)) * transfer, axes=(0, 1))
    return propagated[start:start + grid.n, start:start + grid.n]


def direct_fresnel_1d(field: np.ndarray, grid: Grid1D, wavelength: float, length: float) -> np.ndarray:
    """Small-grid O(N^2) reference implementation for algorithm checks."""
    k = 2.0 * np.pi / wavelength
    kernel = np.exp(1j * k * length) / np.sqrt(1j * wavelength * length)
    phase = np.exp(1j * k * (grid.x[:, None] - grid.x[None, :]) ** 2 / (2.0 * length))
    return kernel * (phase @ field) * grid.dx


def direct_fresnel_2d(field: np.ndarray, grid: Grid2D, wavelength: float,
                      length: float) -> np.ndarray:
    """Small-grid O(N^3) reference implementation of the 2-D Fresnel integral.

    The separable matrix products are algebraically identical to the dense
    O(N^4) double sum, while avoiding a four-dimensional temporary array.  It
    is intended for FFT cross-checks only; production propagation uses the
    padded transfer-function implementation above.
    """
    values = np.asarray(field, dtype=complex)
    if values.shape != (grid.n, grid.n):
        raise ValueError(f"expected field shape {(grid.n, grid.n)}, got {values.shape}")
    k = 2.0 * np.pi / wavelength
    kernel = np.exp(1j * k * length) / (1j * wavelength * length)
    phase_x = np.exp(1j * k * (grid.x[:, None] - grid.x[None, :]) ** 2 /
                     (2.0 * length))
    # x and y have the same sampling in Grid2D.  Keep the two products
    # explicit so a future rectangular-grid extension can use a separate
    # phase_y matrix without changing the normalization.
    phase_y = np.exp(1j * k * (grid.y[:, None] - grid.y[None, :]) ** 2 /
                     (2.0 * length))
    return kernel * (phase_y @ values @ phase_x.T) * grid.dx * grid.dy

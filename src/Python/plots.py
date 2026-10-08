"""Matplotlib figures generated for the report and result review."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_suptitle(fig: plt.Figure, title: str) -> None:
    """Set a readable, wrapped title for report figures.

    Long metadata strings (especially tilt scans) otherwise extend beyond the
    image boundary when Matplotlib centers the title.  Wrapping keeps every
    parameter visible in the PNG while leaving the plotting content unchanged.
    """
    wrapped = textwrap.fill(str(title), width=112, break_long_words=False,
                            break_on_hyphens=False)
    fig.suptitle(wrapped, fontsize=11)


def _is_converged(row: dict) -> bool:
    value = row.get("converged", False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _finite_segments(rows: list[dict], x_key: str, y_key: str) -> list[list[dict]]:
    segments: list[list[dict]] = []
    segment: list[dict] = []
    for row in rows:
        try:
            valid = (_is_converged(row) and
                     np.isfinite(float(row[x_key])) and
                     np.isfinite(float(row[y_key])))
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            segment.append(row)
        elif segment:
            segments.append(segment)
            segment = []
    if segment:
        segments.append(segment)
    return segments


def plot_1d_fields(path: Path, x: np.ndarray, field_m1: np.ndarray,
                   field_m2: np.ndarray, title: str, mask_fraction: float = 1e-3) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, field, label in ((axes[0, 0], field_m1, "M1"), (axes[0, 1], field_m2, "M2")):
        ax.plot(x * 1e3, np.abs(field), lw=1.2)
        ax.set_title(f"{label} amplitude")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("|u|")
        ax.grid(alpha=0.25)
    for ax, field, label in ((axes[1, 0], field_m1, "M1"), (axes[1, 1], field_m2, "M2")):
        magnitude = np.abs(field)
        phase = np.unwrap(np.angle(field))
        phase = np.where(magnitude >= mask_fraction * np.max(magnitude), phase, np.nan)
        ax.plot(x * 1e3, phase, lw=1.0)
        ax.set_title(f"{label} phase (masked)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("phase (rad)")
        ax.grid(alpha=0.25)
    _set_suptitle(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(path: Path, history: list[dict], title: str) -> None:
    iterations = np.array([row["iteration"] for row in history])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].semilogy(iterations, np.maximum([r["delta"] for r in history], 1e-18))
    axes[0].set_title("phase-aligned field error")
    axes[1].semilogy(iterations, np.maximum([r["intensity_difference"] for r in history], 1e-18), label="intensity")
    axes[1].semilogy(iterations, np.maximum([r["eigenvalue_difference"] for r in history], 1e-18), label="eigenvalue")
    axes[1].legend()
    axes[1].set_title("relative changes")
    axes[2].plot(iterations, [r["gamma_abs"] for r in history], label="|gamma|")
    axes[2].plot(iterations, [r["gamma_phase"] for r in history], label="arg(gamma)")
    axes[2].legend()
    axes[2].set_title("eigenvalue")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    _set_suptitle(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_2d_fields(path: Path, x: np.ndarray, y: np.ndarray,
                   field_m1: np.ndarray, field_m2: np.ndarray, title: str,
                   mask_fraction: float = 1e-3) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    xx = x * 1e3
    yy = y * 1e3
    values = [(np.abs(field_m1), "M1 amplitude", None),
              (np.abs(field_m2), "M2 amplitude", None)]
    max_amplitude = max(float(np.max(np.abs(field_m1))), float(np.max(np.abs(field_m2))))
    phase_mask = max_amplitude * mask_fraction
    values.extend([
        (np.ma.masked_where(np.abs(field_m1) < phase_mask, np.angle(field_m1)), "M1 phase", (-np.pi, np.pi)),
        (np.ma.masked_where(np.abs(field_m2) < phase_mask, np.angle(field_m2)), "M2 phase", (-np.pi, np.pi)),
    ])
    for ax, (value, label, limits) in zip(axes.ravel(), values):
        image = ax.pcolormesh(xx, yy, value, shading="auto",
                              cmap="magma" if "amplitude" in label else "twilight",
                              vmin=limits[0] if limits else None,
                              vmax=limits[1] if limits else None)
        fig.colorbar(image, ax=ax, shrink=0.82)
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    _set_suptitle(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tilt_scan(path: Path, rows: list[dict], title: str,
                   fit: dict | None = None) -> None:
    unconverged = [r for r in rows if not _is_converged(r)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)

    def valid_rows(source: list[dict], x_key: str, y_key: str) -> list[dict]:
        output = []
        for row in source:
            try:
                if np.isfinite(float(row[x_key])) and np.isfinite(float(row[y_key])):
                    output.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        return output

    for plane, marker in (("M1", "o"), ("M2", "s")):
        key = f"y_centroid_{plane}_mm"
        for index, segment in enumerate(_finite_segments(rows, "theta_mrad", key)):
            axes[0].plot([r["theta_mrad"] for r in segment], [r[key] for r in segment],
                         marker + "-", label=f"{plane} converged" if index == 0 else None)
        bad = valid_rows(unconverged, "theta_mrad", key)
        if bad:
            axes[0].scatter([r["theta_mrad"] for r in bad], [r[key] for r in bad],
                            marker=marker, facecolors="none", edgecolors="#d62728",
                            label=f"{plane} not converged")
    if fit:
        for plane, style in (("M1", "--"), ("M2", ":")):
            if plane not in fit:
                continue
            key = f"y_centroid_{plane}_mm"
            limit = float(fit[plane].get("max_abs_theta_mrad", np.inf))
            fitted_rows = [row for segment in _finite_segments(rows, "theta_mrad", key)
                           for row in segment
                           if abs(float(row["theta_mrad"])) <= limit]
            if not fitted_rows:
                continue
            x_fit = np.linspace(min(float(r["theta_mrad"]) for r in fitted_rows),
                                max(float(r["theta_mrad"]) for r in fitted_rows), 100)
            r2 = fit[plane].get("r2")
            r2_label = f"$R^2$={r2:.3f}" if r2 is not None else "$R^2$=n/a"
            axes[0].plot(x_fit, fit[plane]["slope_mm_per_mrad"] * x_fit + fit[plane]["intercept_mm"],
                         style, label=f"{plane} fit, {r2_label}")
    axes[0].set_ylabel("y centroid (mm)")
    axes[0].legend()
    for axis, series in ((axes[1], [("mode_radius_y_M1_mm", "o-", "M1 radius"),
                                    ("mode_radius_y_M2_mm", "s-", "M2 radius"),
                                    ("hg00_overlap_M1", "^-", "HG00 overlap M1")]),
                         (axes[2], [("loss_M1", "o-", "M1 loss"),
                                    ("loss_M2", "s-", "M2 loss")]),
                         (axes[3], [("iterations", "o-", "converged")])):
        for key, style, label in series:
            for index, segment in enumerate(_finite_segments(rows, "theta_mrad", key)):
                axis.plot([r["theta_mrad"] for r in segment], [r[key] for r in segment],
                          style, label=label if index == 0 else None)
            bad = valid_rows(unconverged, "theta_mrad", key)
            if bad:
                label = "not converged" if key.endswith("M1_mm") or key == "loss_M1" or axis is axes[3] else None
                if axis is axes[3]:
                    axis.scatter([r["theta_mrad"] for r in bad], [r[key] for r in bad],
                                 marker="x", color="#d62728", label=label)
                else:
                    axis.scatter([r["theta_mrad"] for r in bad], [r[key] for r in bad],
                                 marker="o", facecolors="none", edgecolors="#d62728",
                                 label=label)
    axes[1].legend()
    axes[1].set_ylabel("radius (mm) / overlap")
    axes[2].legend()
    axes[2].set_ylabel("round-trip loss")
    axes[3].legend()
    axes[3].set_ylabel("Fox-Li iterations")
    for ax in axes:
        ax.set_xlabel("tilt theta (mrad)")
        ax.grid(alpha=0.25)
    _set_suptitle(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)

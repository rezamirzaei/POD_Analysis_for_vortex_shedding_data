from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import reshape_field


def plot_energy_spectrum(
    singular_values: ArrayLike,
    cumulative_energy: ArrayLike,
    output_path: Path,
    max_modes: int = 80,
) -> Path:
    sigma = np.asarray(singular_values, dtype=np.float64)
    cumulative = np.asarray(cumulative_energy, dtype=np.float64)
    k = min(max_modes, sigma.size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].semilogy(np.arange(1, k + 1), sigma[:k], "o-", lw=1.8, markersize=3)
    axes[0].set_title("Singular Value Spectrum")
    axes[0].set_xlabel("Mode Number")
    axes[0].set_ylabel("Singular Value")
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(1, k + 1), cumulative[:k] * 100, "-", lw=2)
    for threshold, color in [(90, "green"), (95, "orange"), (99, "red")]:
        axes[1].axhline(threshold, color=color, ls="--", lw=1.2, alpha=0.8)
    axes[1].set_ylim(0, 102)
    axes[1].set_title("Cumulative Energy")
    axes[1].set_xlabel("Mode Number")
    axes[1].set_ylabel("Energy Captured (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_modes_grid(
    spatial_modes: NDArray[np.float64],
    energy_per_mode: NDArray[np.float64],
    grid_shape: Tuple[int, int],
    output_path: Path,
    max_modes: int = 12,
) -> Path:
    max_modes = min(max_modes, spatial_modes.shape[1])
    n_cols = 4
    n_rows = int(np.ceil(max_modes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx in range(max_modes):
        ax = axes_flat[idx]
        mode = reshape_field(spatial_modes[:, idx], grid_shape)
        vmax = float(np.max(np.abs(mode)))
        ax.imshow(mode, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Mode {idx + 1} ({energy_per_mode[idx] * 100:.1f}%)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[max_modes:]:
        ax.axis("off")

    fig.suptitle("Leading POD Spatial Modes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_snapshot_reconstructions(
    original: NDArray[np.float64],
    reconstructions: Mapping[int, NDArray[np.float64]],
    grid_shape: Tuple[int, int],
    output_path: Path,
) -> Path:
    mode_counts = sorted(reconstructions.keys())
    n_panels = len(mode_counts) + 1
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    original_field = reshape_field(original, grid_shape)
    vmax = float(np.percentile(np.abs(original_field), 99))

    axes_flat[0].imshow(
        original_field,
        cmap="RdBu_r",
        origin="lower",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    axes_flat[0].set_title("Original Snapshot", fontsize=10, fontweight="bold")
    axes_flat[0].set_xticks([])
    axes_flat[0].set_yticks([])

    for idx, n_modes in enumerate(mode_counts, start=1):
        ax = axes_flat[idx]
        rec_field = reshape_field(reconstructions[n_modes], grid_shape)
        ax.imshow(rec_field, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Reconstruction ({n_modes} modes)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("POD Snapshot Reconstructions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_temporal_coefficients(
    temporal_coeffs: NDArray[np.float64],
    energy_per_mode: NDArray[np.float64],
    output_path: Path,
    max_modes: int = 4,
) -> Path:
    k = min(max_modes, temporal_coeffs.shape[1])
    fig, axes = plt.subplots(k, 1, figsize=(10, 2.5 * k), sharex=True)
    axes_arr = np.atleast_1d(axes)

    for idx, ax in enumerate(axes_arr):
        ax.plot(temporal_coeffs[:, idx], lw=1.4, color=f"C{idx}")
        ax.set_ylabel(f"a{idx + 1}")
        ax.grid(alpha=0.3)
        ax.set_title(f"Mode {idx + 1} ({energy_per_mode[idx] * 100:.1f}% energy)")

    axes_arr[-1].set_xlabel("Snapshot Index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

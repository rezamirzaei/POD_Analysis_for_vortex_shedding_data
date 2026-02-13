from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfft, rfftfreq


@dataclass(frozen=True)
class SpectrumPeak:
    """Dominant frequency descriptor for a POD temporal coefficient."""

    mode_index: int
    frequency: float
    amplitude: float


def mode_count_for_energy(cumulative_energy: ArrayLike, target_energy: float) -> int:
    energy = np.asarray(cumulative_energy, dtype=np.float64)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("cumulative_energy must be a non-empty 1D array.")
    if not 0 < target_energy <= 1:
        raise ValueError("target_energy must be in the interval (0, 1].")
    return int(np.searchsorted(energy, target_energy, side="left") + 1)


def compression_ratio(
    n_snapshots: int,
    n_spatial_points: int,
    n_modes: int,
    include_mean: bool = True,
) -> float:
    """Return original-storage / reduced-storage ratio for POD representation."""
    if n_snapshots < 1 or n_spatial_points < 1:
        raise ValueError("n_snapshots and n_spatial_points must be >= 1.")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")

    original_storage = n_snapshots * n_spatial_points
    reduced_storage = (n_snapshots * n_modes) + (n_modes * n_spatial_points)
    if include_mean:
        reduced_storage += n_spatial_points

    return float(original_storage / reduced_storage)


def dominant_frequency(signal: ArrayLike, dt: float = 1.0) -> tuple[float, float]:
    """Return the strongest non-DC frequency and its amplitude."""
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 3:
        raise ValueError("signal must be a 1D array with at least 3 samples.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal contains NaN or infinite values.")

    centered = samples - np.mean(samples)
    spectrum = np.abs(rfft(centered))
    freqs = rfftfreq(samples.size, d=dt)
    if spectrum.size == 0:
        return 0.0, 0.0

    spectrum[0] = 0.0
    peak_idx = int(np.argmax(spectrum))
    return float(freqs[peak_idx]), float(spectrum[peak_idx])


def dominant_frequencies(
    temporal_coeffs: ArrayLike,
    dt: float = 1.0,
    max_modes: int = 4,
) -> List[SpectrumPeak]:
    coeffs = np.asarray(temporal_coeffs, dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[0] < 3 or coeffs.shape[1] < 1:
        raise ValueError("temporal_coeffs must be a 2D array with shape (n_samples>=3, n_modes>=1).")
    if max_modes < 1:
        raise ValueError("max_modes must be >= 1.")

    k = min(max_modes, coeffs.shape[1])
    peaks: List[SpectrumPeak] = []
    for idx in range(k):
        freq, amp = dominant_frequency(coeffs[:, idx], dt=dt)
        peaks.append(SpectrumPeak(mode_index=idx + 1, frequency=freq, amplitude=amp))
    return peaks

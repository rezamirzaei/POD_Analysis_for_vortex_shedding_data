from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import SnapshotDataset


@dataclass(frozen=True)
class OscillatoryDataset:
    """Synthetic dataset with known oscillatory frequencies."""

    snapshots: NDArray[np.float64]
    time: NDArray[np.float64]
    frequencies: Tuple[float, ...]
    amplitudes: Tuple[float, ...]
    spatial_grid: NDArray[np.float64]

    def to_snapshot_dataset(self) -> SnapshotDataset:
        return SnapshotDataset(
            snapshots=self.snapshots,
            time_values=self.time,
            metadata={
                "synthetic": True,
                "frequencies_hz": list(self.frequencies),
                "amplitudes": list(self.amplitudes),
            },
        )


def create_oscillatory_dataset(
    n_snapshots: int = 400,
    n_spatial: int = 128,
    dt: float = 0.01,
    frequencies: Sequence[float] = (2.0, 5.0),
    amplitudes: Sequence[float] = (1.0, 0.6),
    noise_level: float = 0.02,
    random_state: int | None = 42,
) -> OscillatoryDataset:
    """Create low-rank oscillatory snapshots for DMD/POD demos."""
    if n_snapshots < 10:
        raise ValueError("n_snapshots must be >= 10.")
    if n_spatial < 8:
        raise ValueError("n_spatial must be >= 8.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if len(frequencies) == 0:
        raise ValueError("frequencies must be non-empty.")
    if len(frequencies) != len(amplitudes):
        raise ValueError("frequencies and amplitudes must have the same length.")

    rng = np.random.default_rng(random_state)
    time = np.arange(n_snapshots, dtype=np.float64) * dt
    x = np.linspace(0, 2 * np.pi, n_spatial, endpoint=False)

    snapshots = np.zeros((n_snapshots, n_spatial), dtype=np.float64)
    for idx, (freq, amp) in enumerate(zip(frequencies, amplitudes), start=1):
        omega_t = 2 * np.pi * float(freq) * time
        # Use sine/cosine spatial-temporal pairs so the dynamics are first-order
        # representable and DMD can recover the target frequencies reliably.
        spatial_sin = np.sin(idx * x)
        spatial_cos = np.cos(idx * x)
        snapshots += float(amp) * (
            np.cos(omega_t)[:, None] * spatial_sin[None, :]
            + np.sin(omega_t)[:, None] * spatial_cos[None, :]
        )

    if noise_level > 0:
        snapshots += noise_level * np.std(snapshots) * rng.standard_normal(snapshots.shape)

    return OscillatoryDataset(
        snapshots=snapshots,
        time=time,
        frequencies=tuple(float(f) for f in frequencies),
        amplitudes=tuple(float(a) for a in amplitudes),
        spatial_grid=x,
    )


def create_multitone_dataset(
    n_snapshots: int = 2048,
    n_spatial: int = 96,
    dt: float = 0.001,
    frequencies: Sequence[float] = (20.0, 50.0, 100.0),
    amplitudes: Sequence[float] = (1.0, 0.7, 0.4),
    noise_level: float = 0.2,
    random_state: int | None = 42,
) -> OscillatoryDataset:
    """Create multi-frequency snapshots suitable for SPOD demonstrations."""
    return create_oscillatory_dataset(
        n_snapshots=n_snapshots,
        n_spatial=n_spatial,
        dt=dt,
        frequencies=frequencies,
        amplitudes=amplitudes,
        noise_level=noise_level,
        random_state=random_state,
    )

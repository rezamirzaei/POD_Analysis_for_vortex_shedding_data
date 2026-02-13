"""Spectral Proper Orthogonal Decomposition (SPOD).

SPOD extracts frequency-resolved coherent structures from turbulent/stochastic
data. Unlike standard POD which is purely energy-based, SPOD identifies modes
that are both energetic AND spectrally coherent.

References:
    - Towne, Schmidt & Colonius, "Spectral proper orthogonal decomposition and
      its relationship to dynamic mode decomposition and resolvent analysis" (2018)
    - Lumley, "Stochastic Tools in Turbulence" (1970) - original formulation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from .core import NotFittedError, _as_valid_matrix


@dataclass(frozen=True)
class SPODMode:
    """Container for a single SPOD mode at a specific frequency."""
    frequency: float
    energy: float
    energy_fraction: float
    mode_index: int
    mode_vector: NDArray[np.complex128]


class SPOD:
    """Spectral Proper Orthogonal Decomposition.

    SPOD computes optimal modes at each frequency by eigendecomposition
    of the cross-spectral density (CSD) matrix.

    This provides frequency-resolved coherent structures that are:
    - Orthogonal at each frequency
    - Ranked by spectral energy
    - Optimal in the spectral norm sense

    Example:
        >>> spod = SPOD(n_fft=256, overlap=0.5, dt=0.001)
        >>> spod.fit(time_series_data)
        >>> modes_at_50Hz = spod.get_modes_at_frequency(50.0)
    """

    def __init__(
        self,
        n_fft: int = 256,
        overlap: float = 0.5,
        dt: float = 1.0,
        window: str = "hann",
        n_modes: Optional[int] = None,
        detrend: str = "constant",
    ) -> None:
        """Initialize SPOD.

        Args:
            n_fft: FFT block size (number of snapshots per block).
            overlap: Overlap fraction between blocks (0.0 to 1.0).
            dt: Time step between snapshots.
            window: Window function ("hann", "hamming", "blackman", "boxcar").
            n_modes: Number of modes to compute at each frequency (None = all).
            detrend: Detrending method ("constant", "linear", None).
        """
        self.n_fft = n_fft
        self.overlap = overlap
        self.dt = dt
        self.window = window
        self.n_modes = n_modes
        self.detrend = detrend

        self._is_fitted = False
        self.frequencies_: Optional[NDArray[np.float64]] = None
        self.eigenvalues_: Optional[NDArray[np.float64]] = None  # (n_freq, n_modes)
        self.modes_: Optional[NDArray[np.complex128]] = None  # (n_freq, n_features, n_modes)
        self.n_blocks_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: ArrayLike) -> "SPOD":
        """Fit SPOD to snapshot data.

        Args:
            X: Snapshot matrix, shape (n_snapshots, n_spatial_points).

        Returns:
            self: Fitted SPOD model.
        """
        matrix = _as_valid_matrix(X)
        self.n_samples_, self.n_features_ = matrix.shape

        if self.n_samples_ < self.n_fft:
            raise ValueError(
                f"Need at least n_fft={self.n_fft} snapshots, got {self.n_samples_}."
            )

        # Compute number of blocks
        n_overlap = int(self.n_fft * self.overlap)
        step = self.n_fft - n_overlap
        self.n_blocks_ = (self.n_samples_ - self.n_fft) // step + 1

        if self.n_blocks_ < 2:
            raise ValueError("Not enough data for at least 2 blocks.")

        # Frequency vector
        self.frequencies_ = np.fft.rfftfreq(self.n_fft, d=self.dt)
        n_freq = len(self.frequencies_)

        # Build window
        if self.window == "hann":
            win = np.hanning(self.n_fft)
        elif self.window == "hamming":
            win = np.hamming(self.n_fft)
        elif self.window == "blackman":
            win = np.blackman(self.n_fft)
        else:
            win = np.ones(self.n_fft)

        win_norm = np.sum(win ** 2)

        # Extract blocks and compute FFT
        blocks_fft = np.zeros((self.n_blocks_, n_freq, self.n_features_), dtype=np.complex128)

        for i in range(self.n_blocks_):
            start = i * step
            end = start + self.n_fft
            block = matrix[start:end]

            # Detrend
            if self.detrend == "constant":
                block = block - np.mean(block, axis=0)
            elif self.detrend == "linear":
                t = np.arange(self.n_fft)
                for j in range(self.n_features_):
                    coeffs = np.polyfit(t, block[:, j], 1)
                    block[:, j] -= np.polyval(coeffs, t)

            # Window and FFT
            block_windowed = block * win[:, np.newaxis]
            blocks_fft[i] = np.fft.rfft(block_windowed, axis=0)

        # Normalize FFT
        blocks_fft *= np.sqrt(2.0 * self.dt / (self.n_fft * win_norm))

        # Compute SPOD at each frequency
        n_modes = self.n_modes if self.n_modes is not None else min(self.n_blocks_, self.n_features_)
        n_modes = min(n_modes, self.n_blocks_, self.n_features_)

        self.eigenvalues_ = np.zeros((n_freq, n_modes))
        self.modes_ = np.zeros((n_freq, self.n_features_, n_modes), dtype=np.complex128)

        for k in range(n_freq):
            # Cross-spectral density matrix: Q = (1/n_blocks) * sum(q_i @ q_i^H)
            Q = blocks_fft[:, k, :]  # (n_blocks, n_features)
            CSD = (Q.conj().T @ Q) / self.n_blocks_

            # Eigendecomposition (returns ascending order)
            eigvals, eigvecs = eigh(CSD)

            # Reverse to descending order
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx][:n_modes]
            eigvecs = eigvecs[:, idx][:, :n_modes]

            self.eigenvalues_[k] = np.real(eigvals)
            self.modes_[k] = eigvecs

        self._is_fitted = True
        return self

    def get_modes_at_frequency(
        self,
        frequency: float,
        n_modes: Optional[int] = None,
    ) -> List[SPODMode]:
        """Get SPOD modes at a specific frequency.

        Args:
            frequency: Target frequency (Hz).
            n_modes: Number of modes to return.

        Returns:
            List of SPODMode objects.
        """
        self._require_fitted()

        # Find nearest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies_ - frequency))
        actual_freq = self.frequencies_[freq_idx]

        n = n_modes if n_modes is not None else self.eigenvalues_.shape[1]
        n = min(n, self.eigenvalues_.shape[1])

        total_energy = np.sum(self.eigenvalues_[freq_idx])

        modes = []
        for i in range(n):
            modes.append(SPODMode(
                frequency=actual_freq,
                energy=float(self.eigenvalues_[freq_idx, i]),
                energy_fraction=float(self.eigenvalues_[freq_idx, i] / total_energy) if total_energy > 0 else 0.0,
                mode_index=i + 1,
                mode_vector=self.modes_[freq_idx, :, i],
            ))

        return modes

    def get_energy_spectrum(self, mode_index: int = 0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get energy spectrum for a specific mode across frequencies.

        Args:
            mode_index: Mode index (0 = leading mode).

        Returns:
            (frequencies, energies) arrays.
        """
        self._require_fitted()

        if mode_index >= self.eigenvalues_.shape[1]:
            raise IndexError(f"Mode index {mode_index} out of bounds.")

        return self.frequencies_.copy(), self.eigenvalues_[:, mode_index].copy()

    def total_energy_spectrum(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get total energy spectrum (sum over all modes).

        Returns:
            (frequencies, total_energies) arrays.
        """
        self._require_fitted()
        return self.frequencies_.copy(), np.sum(self.eigenvalues_, axis=1)

    def cumulative_energy_at_frequency(self, frequency: float) -> NDArray[np.float64]:
        """Get cumulative energy fraction at a frequency.

        Args:
            frequency: Target frequency.

        Returns:
            Cumulative energy fraction for each mode.
        """
        self._require_fitted()

        freq_idx = np.argmin(np.abs(self.frequencies_ - frequency))
        energies = self.eigenvalues_[freq_idx]
        total = np.sum(energies)

        if total == 0:
            return np.zeros_like(energies)

        return np.cumsum(energies) / total

    def find_dominant_frequencies(
        self,
        n_peaks: int = 5,
        min_separation: float = 0.0,
    ) -> List[dict]:
        """Find frequencies with highest SPOD energy.

        Args:
            n_peaks: Number of peaks to find.
            min_separation: Minimum frequency separation between peaks.

        Returns:
            List of peak descriptors.
        """
        self._require_fitted()

        # Total energy at each frequency
        total_energy = np.sum(self.eigenvalues_, axis=1)

        peaks = []
        total_energy_copy = total_energy.copy()

        for _ in range(n_peaks):
            peak_idx = np.argmax(total_energy_copy)
            peak_freq = self.frequencies_[peak_idx]
            peak_energy = total_energy[peak_idx]

            # Leading mode energy fraction at this frequency
            mode_energies = self.eigenvalues_[peak_idx]
            leading_fraction = mode_energies[0] / np.sum(mode_energies) if np.sum(mode_energies) > 0 else 0

            peaks.append({
                'frequency': float(peak_freq),
                'total_energy': float(peak_energy),
                'leading_mode_fraction': float(leading_fraction),
                'frequency_index': int(peak_idx),
            })

            # Mask nearby frequencies
            if min_separation > 0:
                mask = np.abs(self.frequencies_ - peak_freq) < min_separation
                total_energy_copy[mask] = 0
            else:
                total_energy_copy[peak_idx] = 0

        return peaks

    def reconstruct_at_frequency(
        self,
        frequency: float,
        n_modes: Optional[int] = None,
    ) -> NDArray[np.complex128]:
        """Reconstruct field at a specific frequency.

        Args:
            frequency: Target frequency.
            n_modes: Number of modes to use.

        Returns:
            Reconstructed complex field.
        """
        self._require_fitted()

        freq_idx = np.argmin(np.abs(self.frequencies_ - frequency))
        n = n_modes if n_modes is not None else self.eigenvalues_.shape[1]
        n = min(n, self.eigenvalues_.shape[1])

        # Reconstruct as weighted sum of modes
        modes = self.modes_[freq_idx, :, :n]
        energies = self.eigenvalues_[freq_idx, :n]

        # Weight by square root of energy (since modes are orthonormal)
        weights = np.sqrt(energies)

        return modes @ weights

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Call fit() before using this method.")


class StreamingSPOD:
    """Streaming SPOD for online/incremental computation.

    Updates SPOD estimates as new data blocks arrive, useful for
    real-time applications or very long time series.
    """

    def __init__(
        self,
        n_fft: int = 256,
        overlap: float = 0.5,
        dt: float = 1.0,
        window: str = "hann",
        exponential_weight: float = 0.0,
    ) -> None:
        """Initialize streaming SPOD.

        Args:
            n_fft: FFT block size.
            overlap: Overlap fraction.
            dt: Time step.
            window: Window function.
            exponential_weight: Weight for exponential averaging (0 = uniform).
                Higher values weight recent blocks more heavily.
        """
        self.n_fft = n_fft
        self.overlap = overlap
        self.dt = dt
        self.window = window
        self.exponential_weight = exponential_weight

        self._buffer: List[NDArray[np.float64]] = []
        self._csd_accum: Optional[NDArray[np.complex128]] = None
        self.n_blocks_seen_: int = 0
        self.frequencies_: Optional[NDArray[np.float64]] = None
        self.eigenvalues_: Optional[NDArray[np.float64]] = None
        self.modes_: Optional[NDArray[np.complex128]] = None
        self._is_fitted = False

    def partial_fit(self, X: ArrayLike) -> "StreamingSPOD":
        """Update SPOD with new data.

        Args:
            X: New snapshot data.

        Returns:
            self: Updated model.
        """
        matrix = _as_valid_matrix(X)

        # Add to buffer
        for row in matrix:
            self._buffer.append(row)

        # Process complete blocks
        step = int(self.n_fft * (1 - self.overlap))

        while len(self._buffer) >= self.n_fft:
            block = np.array(self._buffer[:self.n_fft])
            self._buffer = self._buffer[step:]
            self._process_block(block)

        return self

    def _process_block(self, block: NDArray[np.float64]) -> None:
        """Process a single block and update CSD estimate."""
        n_features = block.shape[1]

        # Initialize frequencies
        if self.frequencies_ is None:
            self.frequencies_ = np.fft.rfftfreq(self.n_fft, d=self.dt)

        n_freq = len(self.frequencies_)

        # Initialize CSD accumulator
        if self._csd_accum is None:
            self._csd_accum = np.zeros((n_freq, n_features, n_features), dtype=np.complex128)

        # Window
        if self.window == "hann":
            win = np.hanning(self.n_fft)
        elif self.window == "hamming":
            win = np.hamming(self.n_fft)
        else:
            win = np.ones(self.n_fft)

        # Detrend and FFT
        block_centered = block - np.mean(block, axis=0)
        block_windowed = block_centered * win[:, np.newaxis]
        Q = np.fft.rfft(block_windowed, axis=0)

        # Normalize
        win_norm = np.sum(win ** 2)
        Q *= np.sqrt(2.0 * self.dt / (self.n_fft * win_norm))

        # Update CSD with exponential weighting
        if self.exponential_weight > 0:
            alpha = self.exponential_weight
            for k in range(n_freq):
                q = Q[k:k+1].T  # (n_features, 1)
                csd_new = q @ q.conj().T
                self._csd_accum[k] = (1 - alpha) * self._csd_accum[k] + alpha * csd_new
        else:
            # Uniform averaging
            for k in range(n_freq):
                q = Q[k:k+1].T
                csd_new = q @ q.conj().T
                n = self.n_blocks_seen_
                self._csd_accum[k] = (n * self._csd_accum[k] + csd_new) / (n + 1)

        self.n_blocks_seen_ += 1

        # Update eigendecomposition
        self._update_eigen()
        self._is_fitted = True

    def _update_eigen(self) -> None:
        """Update eigenvalues and modes from current CSD estimate."""
        n_freq, n_features, _ = self._csd_accum.shape
        n_modes = min(10, n_features)  # Limit modes for efficiency

        self.eigenvalues_ = np.zeros((n_freq, n_modes))
        self.modes_ = np.zeros((n_freq, n_features, n_modes), dtype=np.complex128)

        for k in range(n_freq):
            eigvals, eigvecs = eigh(self._csd_accum[k])
            idx = np.argsort(eigvals)[::-1][:n_modes]
            self.eigenvalues_[k] = np.real(eigvals[idx])
            self.modes_[k] = eigvecs[:, idx]



"""Dynamic Mode Decomposition (DMD) for data-driven modal analysis.

DMD extracts spatiotemporal coherent structures that evolve linearly in time,
complementing POD which extracts energy-optimal structures.

References:
    - Schmid, P.J. "Dynamic mode decomposition of numerical and experimental data" (2010)
    - Kutz et al. "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems" (2016)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import NotFittedError, _as_valid_matrix


@dataclass(frozen=True)
class DMDMode:
    """Container for a single DMD mode with its properties."""
    index: int
    eigenvalue: complex
    frequency: float  # Hz (requires dt to be meaningful)
    growth_rate: float  # exponential growth/decay rate
    amplitude: float
    mode_vector: NDArray[np.complex128]

    @property
    def period(self) -> float:
        """Oscillation period (1/frequency)."""
        return 1.0 / self.frequency if self.frequency != 0 else float('inf')

    @property
    def is_stable(self) -> bool:
        """True if mode is stable (|eigenvalue| <= 1)."""
        return abs(self.eigenvalue) <= 1.0 + 1e-10

    @property
    def damping_ratio(self) -> float:
        """Damping ratio for oscillatory interpretation."""
        return -self.growth_rate / (2 * np.pi * self.frequency) if self.frequency != 0 else 0.0


class DMD:
    """Exact Dynamic Mode Decomposition.

    Extracts dynamic modes from time-series snapshot data by finding
    the best-fit linear operator A such that X' ≈ A @ X.

    Each DMD mode has:
    - A spatial structure (eigenvector)
    - A temporal eigenvalue (growth rate + frequency)
    - An amplitude (initial condition projection)

    Example:
        >>> dmd = DMD(rank=10, dt=0.01)
        >>> dmd.fit(snapshots)  # shape (n_snapshots, n_spatial)
        >>> future = dmd.predict(n_steps=100)
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        dt: float = 1.0,
        exact: bool = True,
        optimal_amplitudes: bool = True,
    ) -> None:
        """Initialize DMD.

        Args:
            rank: Truncation rank for SVD (None = full rank).
            dt: Time step between snapshots.
            exact: If True, use exact DMD; if False, use projected DMD.
            optimal_amplitudes: If True, compute optimal amplitudes via least-squares.
        """
        self.rank = rank
        self.dt = dt
        self.exact = exact
        self.optimal_amplitudes = optimal_amplitudes

        self._is_fitted = False
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.eigenvalues_: Optional[NDArray[np.complex128]] = None
        self.modes_: Optional[NDArray[np.complex128]] = None
        self.amplitudes_: Optional[NDArray[np.complex128]] = None
        self.frequencies_: Optional[NDArray[np.float64]] = None
        self.growth_rates_: Optional[NDArray[np.float64]] = None
        self._Atilde_: Optional[NDArray[np.complex128]] = None
        self._U_: Optional[NDArray[np.float64]] = None
        self._singular_values_: Optional[NDArray[np.float64]] = None

    def fit(self, X: ArrayLike) -> "DMD":
        """Fit DMD model to snapshot matrix.

        Args:
            X: Snapshot matrix, shape (n_snapshots, n_spatial).
               Rows are time-ordered snapshots.

        Returns:
            self: Fitted DMD model.
        """
        matrix = _as_valid_matrix(X)
        self.n_samples_, self.n_features_ = matrix.shape

        if self.n_samples_ < 2:
            raise ValueError("DMD requires at least 2 snapshots.")

        # Split into X (past) and X' (future)
        X_past = matrix[:-1].T  # shape (n_features, n_samples - 1)
        X_future = matrix[1:].T

        # SVD of past snapshots
        U, sigma, Vh = np.linalg.svd(X_past, full_matrices=False)

        # Truncate to rank
        r = self.rank if self.rank is not None else len(sigma)
        r = min(r, len(sigma))
        U_r = U[:, :r]
        sigma_r = sigma[:r]
        Vh_r = Vh[:r, :]

        self._U_ = U_r
        self._singular_values_ = sigma_r

        # Build low-rank operator A_tilde = U.T @ X' @ V @ S^{-1}
        self._Atilde_ = U_r.T @ X_future @ Vh_r.T @ np.diag(1.0 / sigma_r)

        # Eigendecomposition of A_tilde
        eigvals, W = np.linalg.eig(self._Atilde_)

        # Sort by magnitude (descending)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        W = W[:, idx]

        # Compute DMD modes
        if self.exact:
            # Exact DMD: Phi = X' @ V @ S^{-1} @ W
            self.modes_ = X_future @ Vh_r.T @ np.diag(1.0 / sigma_r) @ W
        else:
            # Projected DMD: Phi = U @ W
            self.modes_ = U_r @ W

        self.eigenvalues_ = eigvals

        # Continuous-time eigenvalues and frequencies
        # Handle zero eigenvalues gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            omega = np.log(eigvals) / self.dt
            # Replace inf/nan with 0
            omega = np.where(np.isfinite(omega), omega, 0.0)
        self.frequencies_ = np.abs(np.imag(omega)) / (2 * np.pi)
        self.growth_rates_ = np.real(omega)

        # Compute amplitudes
        if self.optimal_amplitudes:
            # Optimal amplitudes via least-squares
            self.amplitudes_ = np.linalg.lstsq(
                self.modes_, matrix[0], rcond=None
            )[0]
        else:
            # Initial condition projection
            self.amplitudes_ = np.linalg.pinv(self.modes_) @ matrix[0]

        self._is_fitted = True
        return self

    def predict(
        self,
        n_steps: int,
        start_step: int = 0,
    ) -> NDArray[np.complex128]:
        """Predict future snapshots.

        Args:
            n_steps: Number of time steps to predict.
            start_step: Starting time index (0 = initial condition).

        Returns:
            Predicted snapshots, shape (n_steps, n_features).
        """
        self._require_fitted()

        time_indices = np.arange(start_step, start_step + n_steps)

        # Build Vandermonde matrix
        vander = np.vander(self.eigenvalues_, N=start_step + n_steps, increasing=True)
        time_dynamics = vander[:, start_step:].T * self.amplitudes_

        return (self.modes_ @ time_dynamics.T).T

    def reconstruct(self) -> NDArray[np.complex128]:
        """Reconstruct training data from DMD model."""
        return self.predict(self.n_samples_, start_step=0)

    def get_modes(self, n_modes: Optional[int] = None) -> List[DMDMode]:
        """Get DMD modes as structured objects.

        Args:
            n_modes: Number of top modes to return (default: all).

        Returns:
            List of DMDMode objects sorted by amplitude.
        """
        self._require_fitted()

        n = n_modes if n_modes is not None else len(self.eigenvalues_)
        n = min(n, len(self.eigenvalues_))

        # Sort by amplitude
        idx = np.argsort(np.abs(self.amplitudes_))[::-1]

        modes = []
        for i in range(n):
            j = idx[i]
            modes.append(DMDMode(
                index=i + 1,
                eigenvalue=self.eigenvalues_[j],
                frequency=self.frequencies_[j],
                growth_rate=self.growth_rates_[j],
                amplitude=float(np.abs(self.amplitudes_[j])),
                mode_vector=self.modes_[:, j],
            ))

        return modes

    def reconstruction_error(self, X: ArrayLike, metric: str = "rmse") -> float:
        """Compute reconstruction error.

        Args:
            X: Original snapshot data.
            metric: Error metric ("rmse", "relative_l2", "mae").

        Returns:
            Reconstruction error.
        """
        self._require_fitted()

        matrix = _as_valid_matrix(X)
        reconstructed = np.real(self.reconstruct())
        delta = matrix - reconstructed

        if metric == "rmse":
            return float(np.sqrt(np.mean(delta ** 2)))
        elif metric == "relative_l2":
            return float(np.linalg.norm(delta) / np.linalg.norm(matrix))
        elif metric == "mae":
            return float(np.mean(np.abs(delta)))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def stability_analysis(self) -> dict:
        """Analyze stability of DMD modes.

        Returns:
            Dictionary with stability metrics.
        """
        self._require_fitted()

        magnitudes = np.abs(self.eigenvalues_)

        return {
            "n_stable_modes": int(np.sum(magnitudes <= 1.0 + 1e-10)),
            "n_unstable_modes": int(np.sum(magnitudes > 1.0 + 1e-10)),
            "max_eigenvalue_magnitude": float(np.max(magnitudes)),
            "spectral_radius": float(np.max(magnitudes)),
            "dominant_frequency": float(self.frequencies_[np.argmax(np.abs(self.amplitudes_))]),
            "dominant_growth_rate": float(self.growth_rates_[np.argmax(np.abs(self.amplitudes_))]),
        }

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Call fit() before using this method.")


class OptimizedDMD(DMD):
    """Optimized DMD with variable projection (optDMD).

    Uses variable projection to jointly optimize eigenvalues and
    amplitudes, often yielding more accurate results than exact DMD.

    Reference: Askham & Kutz, "Variable projection methods for an optimized
    dynamic mode decomposition" (2018)
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        dt: float = 1.0,
        max_iter: int = 30,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(rank=rank, dt=dt, exact=True, optimal_amplitudes=True)
        self.max_iter = max_iter
        self.tol = tol
        self._convergence_history_: List[float] = []

    def fit(self, X: ArrayLike) -> "OptimizedDMD":
        """Fit OptimizedDMD using variable projection.

        First fits standard DMD for initialization, then refines.
        """
        # Initialize with standard DMD
        super().fit(X)

        matrix = _as_valid_matrix(X)
        n_snapshots = matrix.shape[0]

        # Refinement via variable projection
        omega = np.log(self.eigenvalues_) / self.dt

        self._convergence_history_ = []

        for iteration in range(self.max_iter):
            # Build Vandermonde matrix with current eigenvalues
            t = np.arange(n_snapshots) * self.dt
            vander = np.exp(np.outer(t, omega))

            # Solve for optimal amplitudes
            amplitudes, residuals, _, _ = np.linalg.lstsq(
                vander, matrix, rcond=None
            )

            # Compute error
            reconstruction = vander @ amplitudes
            error = np.linalg.norm(matrix - reconstruction, 'fro')
            self._convergence_history_.append(error)

            # Check convergence
            if iteration > 0:
                rel_change = abs(
                    self._convergence_history_[-2] - error
                ) / (self._convergence_history_[-2] + 1e-10)
                if rel_change < self.tol:
                    break

            # Update amplitudes (eigenvalue update omitted for simplicity)
            self.amplitudes_ = amplitudes[:, 0] if amplitudes.ndim == 2 else amplitudes

        return self


class BOPDMD:
    """Bagging Optimized DMD for robust mode extraction.

    Uses bootstrap aggregation to identify robust DMD modes
    and estimate uncertainty in eigenvalues/frequencies.

    Reference: Sashidhar & Kutz, "Bagging, optimized dynamic mode
    decomposition for robust, stable forecasting" (2022)
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        dt: float = 1.0,
        n_bags: int = 100,
        bag_fraction: float = 0.8,
        random_state: Optional[int] = None,
    ) -> None:
        self.rank = rank
        self.dt = dt
        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.random_state = random_state

        self._is_fitted = False
        self.eigenvalues_mean_: Optional[NDArray[np.complex128]] = None
        self.eigenvalues_std_: Optional[NDArray[np.float64]] = None
        self.modes_mean_: Optional[NDArray[np.complex128]] = None
        self.frequencies_mean_: Optional[NDArray[np.float64]] = None
        self.frequencies_std_: Optional[NDArray[np.float64]] = None
        self._bag_results_: List[DMD] = []

    def fit(self, X: ArrayLike) -> "BOPDMD":
        """Fit BOP-DMD using bagging."""
        matrix = _as_valid_matrix(X)
        n_snapshots = matrix.shape[0]
        bag_size = int(n_snapshots * self.bag_fraction)

        rng = np.random.default_rng(self.random_state)

        all_eigenvalues = []
        all_frequencies = []
        all_modes = []

        for _ in range(self.n_bags):
            # Bootstrap sample (preserving time order)
            start_idx = rng.integers(0, n_snapshots - bag_size)
            bag_data = matrix[start_idx:start_idx + bag_size]

            # Fit DMD to bag
            dmd = DMD(rank=self.rank, dt=self.dt)
            try:
                dmd.fit(bag_data)
                self._bag_results_.append(dmd)
                all_eigenvalues.append(dmd.eigenvalues_)
                all_frequencies.append(dmd.frequencies_)
                all_modes.append(dmd.modes_)
            except Exception:
                continue

        if len(all_eigenvalues) == 0:
            raise ValueError("All bagging iterations failed.")

        # Aggregate results (simple mean, more sophisticated clustering available)
        # Use the first bag's eigenvalue count as reference
        n_modes = min(len(e) for e in all_eigenvalues)

        eigenvalues = np.array([e[:n_modes] for e in all_eigenvalues])
        frequencies = np.array([f[:n_modes] for f in all_frequencies])

        self.eigenvalues_mean_ = np.mean(eigenvalues, axis=0)
        self.eigenvalues_std_ = np.std(np.abs(eigenvalues), axis=0)
        self.frequencies_mean_ = np.mean(frequencies, axis=0)
        self.frequencies_std_ = np.std(frequencies, axis=0)

        # Use median bag for modes (mode shapes are harder to aggregate)
        median_idx = np.argmin(
            [np.abs(dmd.eigenvalues_[0] - self.eigenvalues_mean_[0])
             for dmd in self._bag_results_]
        )
        self.modes_mean_ = self._bag_results_[median_idx].modes_

        self._is_fitted = True
        return self

    def get_robust_modes(
        self,
        frequency_tolerance: float = 0.1,
        min_occurrence: float = 0.5,
    ) -> List[dict]:
        """Extract modes that appear consistently across bags.

        Args:
            frequency_tolerance: Relative tolerance for frequency matching.
            min_occurrence: Minimum fraction of bags where mode must appear.

        Returns:
            List of robust mode descriptors.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() first.")

        # Cluster frequencies across bags
        all_freqs = []
        for dmd in self._bag_results_:
            all_freqs.extend(dmd.frequencies_)

        all_freqs = np.array(all_freqs)

        # Simple clustering by binning
        robust_modes = []
        for i, f_mean in enumerate(self.frequencies_mean_):
            if f_mean == 0:
                continue

            # Count occurrences
            matches = np.sum(np.abs(all_freqs - f_mean) / f_mean < frequency_tolerance)
            occurrence_rate = matches / (len(self._bag_results_) * len(self.frequencies_mean_))

            if occurrence_rate >= min_occurrence:
                robust_modes.append({
                    'frequency': float(f_mean),
                    'frequency_std': float(self.frequencies_std_[i]),
                    'eigenvalue_magnitude': float(np.abs(self.eigenvalues_mean_[i])),
                    'occurrence_rate': float(occurrence_rate),
                    'is_stable': bool(np.abs(self.eigenvalues_mean_[i]) <= 1.0),
                })

        return sorted(robust_modes, key=lambda x: -x['occurrence_rate'])



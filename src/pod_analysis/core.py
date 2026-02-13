from __future__ import annotations

from enum import Enum
from typing import Optional, Union, Callable
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NotFittedError(RuntimeError):
    """Raised when a POD method requiring a fit is called before fit()."""


class SVDSolver(Enum):
    """Available SVD algorithms for POD decomposition."""
    FULL = "full"
    RANDOMIZED = "randomized"
    TRUNCATED = "truncated"
    AUTO = "auto"


def _as_valid_matrix(X: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, received {matrix.ndim}D.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Input matrix must be non-empty in both dimensions.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Input matrix contains NaN or infinite values.")
    return matrix


def _randomized_svd(
    matrix: NDArray[np.float64],
    n_components: int,
    n_oversamples: int = 10,
    n_power_iters: int = 2,
    random_state: Optional[int] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute truncated randomized SVD (Halko et al. 2011).

    Efficient for large matrices when only top-k components are needed.
    Complexity: O(mn*k) instead of O(mn*min(m,n)) for full SVD.
    """
    rng = np.random.default_rng(random_state)
    m, n = matrix.shape
    k = min(n_components + n_oversamples, min(m, n))

    # Random projection
    omega = rng.standard_normal((n, k))
    Y = matrix @ omega

    # Power iteration for better accuracy
    for _ in range(n_power_iters):
        Y = matrix @ (matrix.T @ Y)

    # Orthonormalize
    Q, _ = np.linalg.qr(Y)

    # Project to lower dimension and compute SVD
    B = Q.T @ matrix
    U_hat, sigma, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    # Truncate to requested components
    return U[:, :n_components], sigma[:n_components], Vt[:n_components]


class POD:
    """Proper Orthogonal Decomposition for snapshot matrices.

    Convention:
    - Input matrix X is shape (n_snapshots, n_spatial_points)
    - Spatial modes are columns in `modes_` with shape (n_spatial_points, n_modes)
    - Temporal coefficients are shape (n_snapshots, n_modes)

    Features:
    - Multiple SVD solvers: full, randomized (for large data), truncated
    - Optional weighted inner product for non-Euclidean norms
    - Incremental updates for streaming data
    - Out-of-sample projection

    Example:
        >>> pod = POD(svd_solver="auto", n_components=20)
        >>> pod.fit(snapshots)
        >>> reduced = pod.transform(new_data)
    """

    def __init__(
        self,
        center: bool = True,
        svd_solver: Union[str, SVDSolver] = "auto",
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
        weight_matrix: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize POD model.

        Args:
            center: If True, mean-center snapshots before decomposition.
            svd_solver: SVD algorithm - "full", "randomized", "truncated", or "auto".
                "auto" uses randomized for large matrices (n > 500 and n_samples > 10).
            n_components: Number of components for truncated/randomized SVD.
                If None, keeps min(n_snapshots, n_features) components.
            random_state: Random seed for reproducible randomized SVD.
            weight_matrix: Optional diagonal weight matrix W for weighted POD.
                The inner product becomes <x, y>_W = x^T W y.
                Pass as 1D array of diagonal entries for efficiency.
        """
        self.center = center
        self.svd_solver = SVDSolver(svd_solver) if isinstance(svd_solver, str) else svd_solver
        self.n_components = n_components
        self.random_state = random_state
        self._weight_matrix = None
        self._weight_sqrt = None
        if weight_matrix is not None:
            self._set_weight_matrix(weight_matrix)

        self._is_fitted = False
        self.mean_: Optional[NDArray[np.float64]] = None
        self.left_singular_vectors_: Optional[NDArray[np.float64]] = None
        self.singular_values_: Optional[NDArray[np.float64]] = None
        self.components_: Optional[NDArray[np.float64]] = None
        self.modes_: Optional[NDArray[np.float64]] = None
        self.temporal_coeffs_: Optional[NDArray[np.float64]] = None
        self.energy_per_mode_: Optional[NDArray[np.float64]] = None
        self.cumulative_energy_: Optional[NDArray[np.float64]] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_components_: Optional[int] = None
        self._noise_variance_: Optional[float] = None

    def _set_weight_matrix(self, weight_matrix: ArrayLike) -> None:
        """Set weight matrix for weighted POD."""
        W = np.asarray(weight_matrix, dtype=np.float64)
        if W.ndim != 1:
            raise ValueError("weight_matrix must be a 1D array of diagonal entries.")
        if np.any(W <= 0):
            raise ValueError("All weight entries must be positive.")
        self._weight_matrix = W
        self._weight_sqrt = np.sqrt(W)

    def _select_svd_solver(self, n_samples: int, n_features: int) -> SVDSolver:
        """Automatically select best SVD solver based on data dimensions."""
        if self.svd_solver != SVDSolver.AUTO:
            return self.svd_solver

        # Use randomized SVD for large matrices
        if n_features > 500 and n_samples > 10:
            return SVDSolver.RANDOMIZED
        return SVDSolver.FULL

    def fit(self, X: ArrayLike) -> "POD":
        """Fit POD model to snapshot matrix.

        Args:
            X: Snapshot matrix of shape (n_snapshots, n_spatial_points).

        Returns:
            self: Fitted POD model.
        """
        matrix = _as_valid_matrix(X)
        self.n_samples_, self.n_features_ = matrix.shape

        if self.center:
            self.mean_ = np.mean(matrix, axis=0)
            centered = matrix - self.mean_
        else:
            self.mean_ = np.zeros(self.n_features_, dtype=np.float64)
            centered = matrix

        # Apply weight matrix if provided (weighted POD)
        if self._weight_sqrt is not None:
            if self._weight_sqrt.size != self.n_features_:
                raise ValueError(
                    f"Weight matrix size {self._weight_sqrt.size} != n_features {self.n_features_}"
                )
            centered = centered * self._weight_sqrt

        # Select and apply SVD solver
        solver = self._select_svd_solver(self.n_samples_, self.n_features_)
        n_comp = self.n_components or min(self.n_samples_, self.n_features_)

        if solver == SVDSolver.RANDOMIZED:
            U, sigma, Vt = _randomized_svd(
                centered,
                n_components=n_comp,
                random_state=self.random_state,
            )
        elif solver == SVDSolver.TRUNCATED:
            # Use scipy's sparse SVD for truncated decomposition
            try:
                from scipy.sparse.linalg import svds
                # svds returns ascending order, so we reverse
                U, sigma, Vt = svds(centered, k=min(n_comp, min(self.n_samples_, self.n_features_) - 1))
                idx = np.argsort(sigma)[::-1]
                U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx]
            except ImportError:
                U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)
        else:  # FULL or AUTO->FULL
            U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)

        # Undo weight transformation for modes
        if self._weight_sqrt is not None:
            Vt = Vt / self._weight_sqrt

        self.left_singular_vectors_ = U
        self.singular_values_ = sigma
        self.components_ = Vt
        self.modes_ = Vt.T
        self.temporal_coeffs_ = U * sigma
        self.n_components_ = int(sigma.size)

        energy = sigma**2
        total_energy = float(np.sum(energy))
        if total_energy <= 0:
            raise ValueError("Total energy is zero; POD decomposition is not meaningful.")
        self.energy_per_mode_ = energy / total_energy
        self.cumulative_energy_ = np.cumsum(self.energy_per_mode_)

        # Estimate noise variance from residual (for probabilistic PCA interpretation)
        if self.n_components_ < min(self.n_samples_, self.n_features_):
            residual_var = (total_energy - np.sum(energy[:self.n_components_])) / (
                self.n_samples_ * self.n_features_ - self.n_components_
            )
            self._noise_variance_ = max(residual_var, 0.0)

        self._is_fitted = True
        return self

    def fit_transform(self, X: ArrayLike, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        self.fit(X)
        return self.temporal_coeffs_[:, : self._resolve_mode_count(n_modes)]

    def transform(self, X: ArrayLike, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        """Project new data onto POD modes.

        Args:
            X: New snapshot data, shape (n_new_snapshots, n_features).
            n_modes: Number of modes to use (default: all).

        Returns:
            Temporal coefficients for new data.
        """
        self._require_fitted()
        matrix = _as_valid_matrix(X)
        if matrix.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected input with {self.n_features_} features, received {matrix.shape[1]}."
            )

        centered = matrix - self.mean_
        k = self._resolve_mode_count(n_modes)
        return centered @ self.components_[:k].T

    def inverse_transform(
        self,
        coefficients: ArrayLike,
        n_modes: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Reconstruct snapshots from temporal coefficients."""
        self._require_fitted()
        coeffs = _as_valid_matrix(coefficients)
        k = self._resolve_mode_count(n_modes if n_modes is not None else coeffs.shape[1])
        if coeffs.shape[1] != k:
            raise ValueError(
                f"Expected coefficients with {k} columns, received {coeffs.shape[1]}."
            )
        return coeffs @ self.components_[:k] + self.mean_

    def reconstruct(self, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        """Reconstruct training data with specified number of modes."""
        self._require_fitted()
        k = self._resolve_mode_count(n_modes)
        return self.temporal_coeffs_[:, :k] @ self.components_[:k] + self.mean_

    def reconstruct_snapshot(self, index: int, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        """Reconstruct a single snapshot."""
        self._require_fitted()
        k = self._resolve_mode_count(n_modes)
        if index < 0 or index >= self.n_samples_:
            raise IndexError(
                f"Snapshot index {index} out of bounds for {self.n_samples_} snapshots."
            )
        return self.temporal_coeffs_[index, :k] @ self.components_[:k] + self.mean_

    def modes_for_energy(self, target_energy: float) -> int:
        """Return minimum number of modes to capture target energy fraction."""
        self._require_fitted()
        if not 0 < target_energy <= 1:
            raise ValueError("target_energy must be in the interval (0, 1].")
        return int(np.searchsorted(self.cumulative_energy_, target_energy, side="left") + 1)

    def reconstruction_error(
        self,
        X: ArrayLike,
        n_modes: Optional[int] = None,
        metric: str = "rmse",
    ) -> float:
        """Compute reconstruction error for given data.

        Args:
            X: Snapshot data to reconstruct.
            n_modes: Number of modes for reconstruction.
            metric: Error metric - "rmse", "mae", "relative_l2", or "max".

        Returns:
            Reconstruction error value.
        """
        self._require_fitted()
        matrix = _as_valid_matrix(X)
        coefficients = self.transform(matrix, n_modes=n_modes)
        reconstruction = self.inverse_transform(coefficients, n_modes=coefficients.shape[1])
        delta = matrix - reconstruction

        metric_name = metric.lower()
        if metric_name == "rmse":
            return float(np.sqrt(np.mean(delta**2)))
        if metric_name == "mae":
            return float(np.mean(np.abs(delta)))
        if metric_name == "relative_l2":
            baseline = float(np.linalg.norm(matrix))
            if baseline == 0:
                return 0.0
            return float(np.linalg.norm(delta) / baseline)
        if metric_name == "max":
            return float(np.max(np.abs(delta)))
        raise ValueError("metric must be one of: rmse, mae, relative_l2, max.")

    def score(self, X: ArrayLike, n_modes: Optional[int] = None) -> float:
        """Return R² score for reconstruction (sklearn-compatible).

        Returns 1.0 for perfect reconstruction, lower for worse fits.
        """
        self._require_fitted()
        matrix = _as_valid_matrix(X)
        reconstructed = self.inverse_transform(self.transform(matrix, n_modes), n_modes)

        ss_res = np.sum((matrix - reconstructed) ** 2)
        ss_tot = np.sum((matrix - np.mean(matrix, axis=0)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def get_feature_loadings(self) -> NDArray[np.float64]:
        """Return feature loadings (modes scaled by singular values).

        Useful for understanding feature importance in each mode.
        """
        self._require_fitted()
        return self.modes_ * self.singular_values_

    def explained_variance_ratio(self, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        """Return fraction of variance explained by each mode (sklearn-compatible)."""
        self._require_fitted()
        k = self._resolve_mode_count(n_modes)
        return self.energy_per_mode_[:k]

    def _resolve_mode_count(self, n_modes: Optional[int]) -> int:
        if n_modes is None:
            return int(self.n_components_)
        if not isinstance(n_modes, int):
            raise TypeError("n_modes must be an integer or None.")
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1.")
        if n_modes > self.n_components_:
            raise ValueError(f"n_modes={n_modes} exceeds available modes={self.n_components_}.")
        return n_modes

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Call fit() before using this method.")


class IncrementalPOD:
    """Incremental/Streaming POD for large or online datasets.

    Uses incremental SVD updates to process data in batches without
    loading the entire dataset into memory.

    Example:
        >>> ipod = IncrementalPOD(n_components=20)
        >>> for batch in data_loader:
        ...     ipod.partial_fit(batch)
        >>> modes = ipod.modes_
    """

    def __init__(
        self,
        n_components: int = 10,
        center: bool = True,
        batch_size: Optional[int] = None,
    ) -> None:
        """Initialize incremental POD.

        Args:
            n_components: Number of POD modes to compute.
            center: If True, incrementally compute and subtract mean.
            batch_size: Suggested batch size for partial_fit (informational).
        """
        self.n_components = n_components
        self.center = center
        self.batch_size = batch_size

        self._is_fitted = False
        self.n_samples_seen_: int = 0
        self.n_features_: Optional[int] = None
        self.mean_: Optional[NDArray[np.float64]] = None
        self.singular_values_: Optional[NDArray[np.float64]] = None
        self.components_: Optional[NDArray[np.float64]] = None
        self.modes_: Optional[NDArray[np.float64]] = None
        self.energy_per_mode_: Optional[NDArray[np.float64]] = None
        self.cumulative_energy_: Optional[NDArray[np.float64]] = None
        self._total_variance_: float = 0.0

    def partial_fit(self, X: ArrayLike) -> "IncrementalPOD":
        """Update POD with new batch of snapshots.

        Args:
            X: Batch of snapshots, shape (n_batch, n_features).

        Returns:
            self: Updated model.
        """
        batch = _as_valid_matrix(X)
        n_batch, n_features = batch.shape

        if self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(
                f"Batch has {n_features} features, expected {self.n_features_}."
            )

        # Incremental mean update
        if self.mean_ is None:
            self.mean_ = np.zeros(n_features, dtype=np.float64)

        if self.center:
            old_mean = self.mean_.copy()
            self.mean_ = (
                self.n_samples_seen_ * self.mean_ + np.sum(batch, axis=0)
            ) / (self.n_samples_seen_ + n_batch)
            batch_centered = batch - self.mean_

            # Correction for mean shift
            if self.n_samples_seen_ > 0 and self.components_ is not None:
                mean_correction = np.sqrt(
                    self.n_samples_seen_ * n_batch / (self.n_samples_seen_ + n_batch)
                ) * (old_mean - self.mean_)
                batch_centered = np.vstack([
                    batch_centered,
                    mean_correction.reshape(1, -1)
                ])
        else:
            batch_centered = batch

        # Update SVD incrementally
        if self.components_ is None:
            # First batch: standard SVD
            U, sigma, Vt = np.linalg.svd(batch_centered, full_matrices=False)
            k = min(self.n_components, sigma.size)
            self.singular_values_ = sigma[:k]
            self.components_ = Vt[:k]
        else:
            # Incremental SVD update using projection
            k = self.n_components

            # Project new data onto current basis
            projected = batch_centered @ self.components_.T
            residual = batch_centered - projected @ self.components_

            # QR of residual for orthogonal complement
            Q, R = np.linalg.qr(residual.T)

            # Build augmented matrix for SVD
            aug_top = np.diag(self.singular_values_)
            aug_top = np.hstack([aug_top, projected.T])
            aug_bottom = np.hstack([np.zeros((Q.shape[1], k)), R.T])
            augmented = np.vstack([aug_top, aug_bottom])

            # SVD of augmented matrix
            U_aug, sigma_aug, Vt_aug = np.linalg.svd(augmented, full_matrices=False)

            # Update components
            new_k = min(k, sigma_aug.size)
            self.singular_values_ = sigma_aug[:new_k]

            # Rotate basis
            old_components = np.vstack([self.components_, Q.T])
            self.components_ = (Vt_aug[:new_k] @ old_components)

        self.n_samples_seen_ += n_batch
        self.modes_ = self.components_.T if self.components_ is not None else None

        # Update energy statistics
        if self.singular_values_ is not None:
            energy = self.singular_values_ ** 2
            total = np.sum(energy)
            self.energy_per_mode_ = energy / total if total > 0 else energy
            self.cumulative_energy_ = np.cumsum(self.energy_per_mode_)

        self._is_fitted = True
        return self

    def fit(self, X: ArrayLike) -> "IncrementalPOD":
        """Fit model in batches (convenience method)."""
        matrix = _as_valid_matrix(X)
        batch_size = self.batch_size or max(100, matrix.shape[0] // 10)

        for start in range(0, matrix.shape[0], batch_size):
            end = min(start + batch_size, matrix.shape[0])
            self.partial_fit(matrix[start:end])

        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project data onto current POD modes."""
        if not self._is_fitted:
            raise NotFittedError("Call partial_fit or fit before transform.")

        matrix = _as_valid_matrix(X)
        if self.center:
            matrix = matrix - self.mean_
        return matrix @ self.components_.T

    def inverse_transform(self, coefficients: ArrayLike) -> NDArray[np.float64]:
        """Reconstruct from coefficients."""
        if not self._is_fitted:
            raise NotFittedError("Call partial_fit or fit before inverse_transform.")

        coeffs = np.asarray(coefficients, dtype=np.float64)
        result = coeffs @ self.components_
        if self.center:
            result = result + self.mean_
        return result


class MethodOfSnapshots:
    """POD using method of snapshots (efficient when n_spatial >> n_snapshots).

    Instead of SVD on X (n_t × n_x), computes eigendecomposition of
    the smaller correlation matrix C = X @ X.T (n_t × n_t).

    This is O(n_t³ + n_t² × n_x) vs O(n_t × n_x × min(n_t, n_x)) for full SVD.
    """

    def __init__(self, center: bool = True) -> None:
        self.center = center
        self._is_fitted = False
        self.mean_: Optional[NDArray[np.float64]] = None
        self.singular_values_: Optional[NDArray[np.float64]] = None
        self.modes_: Optional[NDArray[np.float64]] = None
        self.temporal_coeffs_: Optional[NDArray[np.float64]] = None
        self.energy_per_mode_: Optional[NDArray[np.float64]] = None
        self.cumulative_energy_: Optional[NDArray[np.float64]] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_components_: Optional[int] = None

    def fit(self, X: ArrayLike) -> "MethodOfSnapshots":
        """Fit POD using method of snapshots.

        Args:
            X: Snapshot matrix, shape (n_snapshots, n_spatial_points).
        """
        matrix = _as_valid_matrix(X)
        self.n_samples_, self.n_features_ = matrix.shape

        if self.center:
            self.mean_ = np.mean(matrix, axis=0)
            centered = matrix - self.mean_
        else:
            self.mean_ = np.zeros(self.n_features_, dtype=np.float64)
            centered = matrix

        # Correlation matrix C = X @ X.T
        correlation = centered @ centered.T

        # Eigendecomposition (returns ascending order)
        eigvals, eigvecs = np.linalg.eigh(correlation)

        # Reverse to descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Filter non-positive eigenvalues (numerical noise)
        positive = eigvals > np.finfo(np.float64).eps * eigvals[0]
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]

        # Singular values
        self.singular_values_ = np.sqrt(eigvals)

        # Compute spatial modes: Phi = X.T @ psi / sigma
        self.modes_ = (centered.T @ eigvecs) / self.singular_values_

        # Temporal coefficients
        self.temporal_coeffs_ = eigvecs * self.singular_values_

        self.n_components_ = int(self.singular_values_.size)

        # Energy
        energy = self.singular_values_ ** 2
        total_energy = np.sum(energy)
        self.energy_per_mode_ = energy / total_energy
        self.cumulative_energy_ = np.cumsum(self.energy_per_mode_)

        self._is_fitted = True
        return self

    def reconstruct(self, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        """Reconstruct snapshots with specified number of modes."""
        if not self._is_fitted:
            raise NotFittedError("Call fit() first.")

        k = n_modes if n_modes is not None else self.n_components_
        k = min(k, self.n_components_)

        return self.temporal_coeffs_[:, :k] @ self.modes_[:, :k].T + self.mean_

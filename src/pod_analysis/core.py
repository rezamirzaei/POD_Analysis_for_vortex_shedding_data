from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NotFittedError(RuntimeError):
    """Raised when a POD method requiring a fit is called before fit()."""


def _as_valid_matrix(X: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, received {matrix.ndim}D.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Input matrix must be non-empty in both dimensions.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Input matrix contains NaN or infinite values.")
    return matrix


class POD:
    """Proper Orthogonal Decomposition for snapshot matrices.

    Convention:
    - Input matrix X is shape (n_snapshots, n_spatial_points)
    - Spatial modes are columns in `modes_` with shape (n_spatial_points, n_modes)
    - Temporal coefficients are shape (n_snapshots, n_modes)
    """

    def __init__(self, center: bool = True) -> None:
        self.center = center
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

    def fit(self, X: ArrayLike) -> "POD":
        matrix = _as_valid_matrix(X)
        self.n_samples_, self.n_features_ = matrix.shape

        if self.center:
            self.mean_ = np.mean(matrix, axis=0)
            centered = matrix - self.mean_
        else:
            self.mean_ = np.zeros(self.n_features_, dtype=np.float64)
            centered = matrix

        U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)
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
        self._is_fitted = True
        return self

    def fit_transform(self, X: ArrayLike, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        self.fit(X)
        return self.temporal_coeffs_[:, : self._resolve_mode_count(n_modes)]

    def transform(self, X: ArrayLike, n_modes: Optional[int] = None) -> NDArray[np.float64]:
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
        self._require_fitted()
        coeffs = _as_valid_matrix(coefficients)
        k = self._resolve_mode_count(n_modes if n_modes is not None else coeffs.shape[1])
        if coeffs.shape[1] != k:
            raise ValueError(
                f"Expected coefficients with {k} columns, received {coeffs.shape[1]}."
            )
        return coeffs @ self.components_[:k] + self.mean_

    def reconstruct(self, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        self._require_fitted()
        k = self._resolve_mode_count(n_modes)
        return self.temporal_coeffs_[:, :k] @ self.components_[:k] + self.mean_

    def reconstruct_snapshot(self, index: int, n_modes: Optional[int] = None) -> NDArray[np.float64]:
        self._require_fitted()
        k = self._resolve_mode_count(n_modes)
        if index < 0 or index >= self.n_samples_:
            raise IndexError(
                f"Snapshot index {index} out of bounds for {self.n_samples_} snapshots."
            )
        return self.temporal_coeffs_[index, :k] @ self.components_[:k] + self.mean_

    def modes_for_energy(self, target_energy: float) -> int:
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
        raise ValueError("metric must be one of: rmse, mae, relative_l2.")

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

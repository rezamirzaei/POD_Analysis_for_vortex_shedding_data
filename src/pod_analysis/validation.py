from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

from .core import NotFittedError, POD


@dataclass(frozen=True)
class PODConsistencyReport:
    """Numerical checks that validate core POD identities."""

    orthonormality_fro_error: float
    projection_relative_error: float
    parseval_relative_error: float
    energy_sum_error: float
    energy_vector_relative_error: float
    cumulative_energy_is_monotonic: bool
    rmse_curve_is_monotonic: bool
    max_modes_checked: int
    passed: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "orthonormality_fro_error": self.orthonormality_fro_error,
            "projection_relative_error": self.projection_relative_error,
            "parseval_relative_error": self.parseval_relative_error,
            "energy_sum_error": self.energy_sum_error,
            "energy_vector_relative_error": self.energy_vector_relative_error,
            "cumulative_energy_is_monotonic": self.cumulative_energy_is_monotonic,
            "rmse_curve_is_monotonic": self.rmse_curve_is_monotonic,
            "max_modes_checked": self.max_modes_checked,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PODValidationSummary:
    """Extended POD validation bundle for notebook/report usage."""

    consistency: PODConsistencyReport
    theoretical_rmse: float
    empirical_rmse: float
    rmse_abs_difference: float
    snapshot_singular_value_relative_error: float
    snapshot_modes_compared: int
    passed: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "consistency": self.consistency.to_dict(),
            "theoretical_rmse": self.theoretical_rmse,
            "empirical_rmse": self.empirical_rmse,
            "rmse_abs_difference": self.rmse_abs_difference,
            "snapshot_singular_value_relative_error": self.snapshot_singular_value_relative_error,
            "snapshot_modes_compared": self.snapshot_modes_compared,
            "passed": self.passed,
        }


def validate_pod_consistency(
    pod: POD,
    X: ArrayLike,
    max_modes: int | None = None,
    atol: float = 1e-8,
) -> PODConsistencyReport:
    """Validate core identities for a fitted POD model."""
    _require_fitted_pod(pod)
    matrix = _validate_input_matrix(pod, X)

    centered = matrix - pod.mean_
    modes = pod.modes_
    temporal = pod.temporal_coeffs_
    sigma = pod.singular_values_
    n_samples, n_features = matrix.shape
    n_modes_total = int(sigma.size)

    # 1) Orthonormal spatial modes: Phi^T Phi = I
    gram = modes.T @ modes
    orthonormality_fro_error = float(
        np.linalg.norm(gram - np.eye(n_modes_total, dtype=np.float64), ord="fro")
    )

    # 2) Temporal coefficients identity: A = X_centered @ Phi
    projected = centered @ modes
    projection_relative_error = float(
        np.linalg.norm(projected - temporal)
        / max(np.linalg.norm(temporal), np.finfo(np.float64).eps)
    )

    # 3) Parseval identity for SVD: ||X_centered||_F^2 = sum(sigma^2)
    centered_energy = float(np.linalg.norm(centered, ord="fro") ** 2)
    svd_energy = float(np.sum(sigma**2))
    parseval_relative_error = abs(centered_energy - svd_energy) / max(
        centered_energy, np.finfo(np.float64).eps
    )

    # 4) Energy vector consistency.
    computed_energy = (sigma**2) / svd_energy
    energy_sum_error = abs(float(np.sum(pod.energy_per_mode_)) - 1.0)
    energy_vector_relative_error = float(
        np.linalg.norm(computed_energy - pod.energy_per_mode_)
        / max(np.linalg.norm(computed_energy), np.finfo(np.float64).eps)
    )

    # 5) Cumulative energy monotonicity.
    cumulative_diffs = np.diff(pod.cumulative_energy_)
    cumulative_energy_is_monotonic = bool(np.all(cumulative_diffs >= -atol))

    # 6) Truncated-SVD RMSE monotonicity.
    max_k = n_modes_total if max_modes is None else min(max_modes, n_modes_total)
    tail_energy = np.cumsum((sigma[::-1] ** 2))[::-1]
    rmse_values = []
    for k in range(1, max_k + 1):
        residual_sq = float(tail_energy[k]) if k < n_modes_total else 0.0
        rmse_values.append(np.sqrt(residual_sq / (n_samples * n_features)))
    rmse_diffs = np.diff(np.asarray(rmse_values, dtype=np.float64))
    rmse_curve_is_monotonic = bool(np.all(rmse_diffs <= atol))

    passed = bool(
        orthonormality_fro_error <= atol
        and projection_relative_error <= atol
        and parseval_relative_error <= atol
        and energy_sum_error <= atol
        and energy_vector_relative_error <= atol
        and cumulative_energy_is_monotonic
        and rmse_curve_is_monotonic
    )

    return PODConsistencyReport(
        orthonormality_fro_error=orthonormality_fro_error,
        projection_relative_error=projection_relative_error,
        parseval_relative_error=parseval_relative_error,
        energy_sum_error=energy_sum_error,
        energy_vector_relative_error=energy_vector_relative_error,
        cumulative_energy_is_monotonic=cumulative_energy_is_monotonic,
        rmse_curve_is_monotonic=rmse_curve_is_monotonic,
        max_modes_checked=max_k,
        passed=passed,
    )


def theoretical_truncation_rmse(pod: POD, n_modes: int) -> float:
    """Compute theoretical RMSE from truncated singular-value tail."""
    _require_fitted_pod(pod)
    if n_modes < 1 or n_modes > int(pod.n_components_):
        raise ValueError(f"n_modes must be in [1, {pod.n_components_}].")

    n_samples = int(pod.n_samples_)
    n_features = int(pod.n_features_)
    sigma = pod.singular_values_
    residual_sq = float(np.sum(sigma[n_modes:] ** 2))
    return float(np.sqrt(residual_sq / (n_samples * n_features)))


def snapshot_method_singular_value_error(
    pod: POD,
    X: ArrayLike,
    n_compare_modes: int = 12,
) -> tuple[float, int]:
    """Compare direct POD singular values with method-of-snapshots singular values."""
    _require_fitted_pod(pod)
    if n_compare_modes < 1:
        raise ValueError("n_compare_modes must be >= 1.")

    matrix = _validate_input_matrix(pod, X)
    centered = matrix - pod.mean_

    correlation = centered @ centered.T
    eigvals = np.linalg.eigvalsh(correlation)[::-1]
    sigma_snapshots = np.sqrt(np.clip(eigvals, 0.0, None))

    sigma = pod.singular_values_
    k = min(n_compare_modes, sigma.size, sigma_snapshots.size)
    rel_error = float(
        np.linalg.norm(sigma[:k] - sigma_snapshots[:k])
        / max(np.linalg.norm(sigma[:k]), np.finfo(np.float64).eps)
    )
    return rel_error, int(k)


def validate_pod_model(
    pod: POD,
    X: ArrayLike,
    selected_modes: int,
    max_modes: int | None = 30,
    atol: float = 1e-8,
    snapshot_rtol: float = 1e-8,
) -> PODValidationSummary:
    """Run full POD validation bundle used in notebook and reports."""
    consistency = validate_pod_consistency(pod=pod, X=X, max_modes=max_modes, atol=atol)
    theoretical_rmse = theoretical_truncation_rmse(pod=pod, n_modes=selected_modes)
    empirical_rmse = pod.reconstruction_error(X, n_modes=selected_modes, metric="rmse")
    rmse_abs_difference = abs(theoretical_rmse - empirical_rmse)
    sv_error, compared = snapshot_method_singular_value_error(pod=pod, X=X, n_compare_modes=12)

    passed = bool(
        consistency.passed
        and rmse_abs_difference <= atol
        and sv_error <= snapshot_rtol
    )
    return PODValidationSummary(
        consistency=consistency,
        theoretical_rmse=theoretical_rmse,
        empirical_rmse=empirical_rmse,
        rmse_abs_difference=rmse_abs_difference,
        snapshot_singular_value_relative_error=sv_error,
        snapshot_modes_compared=compared,
        passed=passed,
    )


def _require_fitted_pod(pod: POD) -> None:
    if not hasattr(pod, "_is_fitted") or not pod._is_fitted:
        raise NotFittedError("Call fit() before validation.")


def _validate_input_matrix(pod: POD, X: ArrayLike) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if matrix.shape[1] != pod.n_features_:
        raise ValueError(
            f"X has {matrix.shape[1]} features but POD was fitted with {pod.n_features_}."
        )
    return matrix

"""Advanced mode selection methods for POD.

Provides principled approaches to selecting the optimal number of
POD modes beyond simple energy thresholds:

- Cross-validation for predictive accuracy
- Information criteria (AIC, BIC, MDL)
- Parallel analysis (Horn's method)
- Optimal hard thresholding (Gavish-Donoho)
- Task-specific selection

References:
    - Gavish & Donoho, "The Optimal Hard Threshold for Singular Values is 4/√3" (2014)
    - Horn, "A rationale and test for the number of factors in factor analysis" (1965)
    - Wold, "Cross-validatory estimation of the number of components" (1978)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import POD, _as_valid_matrix


@dataclass(frozen=True)
class ModeSelectionResult:
    """Result of automatic mode selection."""
    selected_modes: int
    method: str
    criterion_values: NDArray[np.float64]
    optimal_value: float
    confidence_interval: Optional[Tuple[int, int]] = None
    details: Optional[dict] = None


def select_modes_by_energy(
    cumulative_energy: ArrayLike,
    target: float = 0.95,
) -> ModeSelectionResult:
    """Select modes by cumulative energy threshold.

    Args:
        cumulative_energy: Cumulative energy ratios.
        target: Target energy fraction (0.0 to 1.0).

    Returns:
        ModeSelectionResult with selected mode count.
    """
    energy = np.asarray(cumulative_energy, dtype=np.float64)

    if target <= 0 or target > 1:
        raise ValueError("target must be in (0, 1]")

    n_modes = int(np.searchsorted(energy, target, side="left") + 1)
    n_modes = min(n_modes, len(energy))

    return ModeSelectionResult(
        selected_modes=n_modes,
        method="energy_threshold",
        criterion_values=energy,
        optimal_value=target,
        details={"target": target, "achieved": float(energy[n_modes - 1])},
    )


def select_modes_by_elbow(
    singular_values: ArrayLike,
    min_modes: int = 1,
) -> ModeSelectionResult:
    """Select modes using the elbow/knee method.

    Finds the point of maximum curvature in the singular value spectrum,
    indicating transition from signal to noise.

    Args:
        singular_values: Array of singular values.
        min_modes: Minimum number of modes to select.

    Returns:
        ModeSelectionResult with selected mode count.
    """
    sigma = np.asarray(singular_values, dtype=np.float64)
    n = len(sigma)

    if n < 3:
        return ModeSelectionResult(
            selected_modes=n,
            method="elbow",
            criterion_values=sigma,
            optimal_value=float(sigma[-1]) if n > 0 else 0.0,
        )

    # Normalize to [0, 1] range for curvature calculation
    x = np.arange(n) / (n - 1)
    y = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-10)

    # Compute curvature using second derivative
    # For discrete data: curvature ≈ |y''| / (1 + y'^2)^(3/2)
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    curvature = np.abs(d2y) / (1 + dy**2)**1.5

    # Find maximum curvature point
    # Skip first and last points (boundary effects)
    elbow_idx = np.argmax(curvature[1:-1]) + 1
    elbow_idx = max(elbow_idx, min_modes - 1)

    return ModeSelectionResult(
        selected_modes=elbow_idx + 1,
        method="elbow",
        criterion_values=curvature,
        optimal_value=float(curvature[elbow_idx]),
        details={"curvature": curvature.tolist()},
    )


def select_modes_by_parallel_analysis(
    X: ArrayLike,
    n_permutations: int = 100,
    percentile: float = 95.0,
    random_state: Optional[int] = None,
) -> ModeSelectionResult:
    """Select modes using parallel analysis (Horn's method).

    Compares observed eigenvalues against eigenvalues from random
    matrices of the same size. Retains modes with eigenvalues above
    the random threshold.

    Args:
        X: Original data matrix.
        n_permutations: Number of random permutations.
        percentile: Percentile of random eigenvalues for threshold.
        random_state: Random seed.

    Returns:
        ModeSelectionResult with selected mode count.
    """
    matrix = _as_valid_matrix(X)
    n_samples, n_features = matrix.shape

    rng = np.random.default_rng(random_state)

    # Compute observed eigenvalues
    centered = matrix - np.mean(matrix, axis=0)
    _, sigma, _ = np.linalg.svd(centered, full_matrices=False)
    observed_eigvals = sigma**2 / n_samples

    # Generate random eigenvalues
    n_components = min(n_samples, n_features)
    random_eigvals = np.zeros((n_permutations, n_components))

    for i in range(n_permutations):
        # Permute each column independently
        random_matrix = np.empty_like(matrix)
        for j in range(n_features):
            random_matrix[:, j] = rng.permutation(matrix[:, j])

        random_centered = random_matrix - np.mean(random_matrix, axis=0)
        _, s, _ = np.linalg.svd(random_centered, full_matrices=False)
        random_eigvals[i] = s**2 / n_samples

    # Compute threshold (percentile of random eigenvalues)
    threshold = np.percentile(random_eigvals, percentile, axis=0)

    # Count modes above threshold
    n_modes = int(np.sum(observed_eigvals > threshold))
    n_modes = max(1, n_modes)

    return ModeSelectionResult(
        selected_modes=n_modes,
        method="parallel_analysis",
        criterion_values=observed_eigvals,
        optimal_value=float(threshold[n_modes - 1]) if n_modes <= len(threshold) else 0.0,
        details={
            "threshold": threshold.tolist(),
            "observed": observed_eigvals.tolist(),
            "percentile": percentile,
        },
    )


def select_modes_by_optimal_threshold(
    singular_values: ArrayLike,
    n_samples: int,
    n_features: int,
    threshold_type: str = "hard",
) -> ModeSelectionResult:
    """Select modes using Gavish-Donoho optimal threshold.

    For data corrupted by IID Gaussian noise, this provides the
    optimal hard/soft threshold for singular values.

    Args:
        singular_values: Array of singular values.
        n_samples: Number of samples (rows).
        n_features: Number of features (columns).
        threshold_type: "hard" or "soft".

    Returns:
        ModeSelectionResult with selected mode count.
    """
    sigma = np.asarray(singular_values, dtype=np.float64)

    # Aspect ratio
    beta = min(n_samples, n_features) / max(n_samples, n_features)

    # Optimal threshold coefficient (Gavish & Donoho 2014)
    if threshold_type == "hard":
        # lambda* = omega(beta) * sigma_noise
        # omega(beta) ≈ 0.56 * beta^3 - 0.95 * beta^2 + 1.82 * beta + 1.43
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    else:
        # Soft threshold coefficient
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.43 * beta + 1.08

    # Estimate noise level from median singular value
    # Median of Marcenko-Pastur distribution
    mp_median = np.sqrt(2 * beta) * (1 + np.sqrt(beta))**2 / (1 + beta + np.sqrt(1 + beta**2))

    median_sigma = np.median(sigma)
    noise_estimate = median_sigma / mp_median

    # Threshold
    threshold = omega * noise_estimate * np.sqrt(max(n_samples, n_features))

    # Count modes above threshold
    n_modes = int(np.sum(sigma > threshold))
    n_modes = max(1, n_modes)

    return ModeSelectionResult(
        selected_modes=n_modes,
        method=f"optimal_{threshold_type}_threshold",
        criterion_values=sigma,
        optimal_value=float(threshold),
        details={
            "noise_estimate": float(noise_estimate),
            "threshold": float(threshold),
            "omega": float(omega),
            "beta": float(beta),
        },
    )


def select_modes_by_aic(
    X: ArrayLike,
    max_modes: Optional[int] = None,
) -> ModeSelectionResult:
    """Select modes using Akaike Information Criterion (AIC).

    Balances model fit against complexity, suitable when the goal
    is predictive accuracy.

    Args:
        X: Data matrix.
        max_modes: Maximum modes to consider.

    Returns:
        ModeSelectionResult with selected mode count.
    """
    matrix = _as_valid_matrix(X)
    n, p = matrix.shape

    centered = matrix - np.mean(matrix, axis=0)
    _, sigma, _ = np.linalg.svd(centered, full_matrices=False)

    max_k = max_modes if max_modes is not None else min(n, p)
    max_k = min(max_k, len(sigma))

    aic_values = np.zeros(max_k)

    for k in range(1, max_k + 1):
        # Reconstruction error (log-likelihood proxy)
        residual_var = np.sum(sigma[k:]**2) / (n * p)

        # Number of parameters: k modes * (n + p) + k singular values + p mean
        n_params = k * (n + p - k) + p

        # AIC = -2 * log(L) + 2 * k_params
        # Using Gaussian likelihood approximation
        if residual_var > 0:
            log_likelihood = -0.5 * n * p * (np.log(2 * np.pi * residual_var) + 1)
        else:
            log_likelihood = 0

        aic_values[k - 1] = -2 * log_likelihood + 2 * n_params

    optimal_k = int(np.argmin(aic_values)) + 1

    return ModeSelectionResult(
        selected_modes=optimal_k,
        method="aic",
        criterion_values=aic_values,
        optimal_value=float(aic_values[optimal_k - 1]),
    )


def select_modes_by_bic(
    X: ArrayLike,
    max_modes: Optional[int] = None,
) -> ModeSelectionResult:
    """Select modes using Bayesian Information Criterion (BIC).

    Penalizes complexity more heavily than AIC, tends to select
    fewer modes. Better for model selection/identification.

    Args:
        X: Data matrix.
        max_modes: Maximum modes to consider.

    Returns:
        ModeSelectionResult with selected mode count.
    """
    matrix = _as_valid_matrix(X)
    n, p = matrix.shape

    centered = matrix - np.mean(matrix, axis=0)
    _, sigma, _ = np.linalg.svd(centered, full_matrices=False)

    max_k = max_modes if max_modes is not None else min(n, p)
    max_k = min(max_k, len(sigma))

    bic_values = np.zeros(max_k)

    for k in range(1, max_k + 1):
        residual_var = np.sum(sigma[k:]**2) / (n * p)
        n_params = k * (n + p - k) + p

        if residual_var > 0:
            log_likelihood = -0.5 * n * p * (np.log(2 * np.pi * residual_var) + 1)
        else:
            log_likelihood = 0

        # BIC = -2 * log(L) + log(n) * k_params
        bic_values[k - 1] = -2 * log_likelihood + np.log(n) * n_params

    optimal_k = int(np.argmin(bic_values)) + 1

    return ModeSelectionResult(
        selected_modes=optimal_k,
        method="bic",
        criterion_values=bic_values,
        optimal_value=float(bic_values[optimal_k - 1]),
    )


def select_modes_by_cross_validation(
    X: ArrayLike,
    max_modes: Optional[int] = None,
    n_folds: int = 5,
    random_state: Optional[int] = None,
    metric: str = "rmse",
) -> ModeSelectionResult:
    """Select modes using k-fold cross-validation.

    Evaluates predictive performance on held-out data to find
    the optimal number of modes.

    Args:
        X: Data matrix.
        max_modes: Maximum modes to evaluate.
        n_folds: Number of CV folds.
        random_state: Random seed for fold assignment.
        metric: Error metric ("rmse", "mae", "relative_l2").

    Returns:
        ModeSelectionResult with selected mode count and CV scores.
    """
    matrix = _as_valid_matrix(X)
    n_samples, n_features = matrix.shape

    if n_samples < n_folds:
        raise ValueError(f"Need at least {n_folds} samples for {n_folds}-fold CV.")

    max_k = max_modes if max_modes is not None else min(n_samples, n_features) // 2
    max_k = min(max_k, min(n_samples - n_samples // n_folds, n_features))
    max_k = max(1, max_k)

    rng = np.random.default_rng(random_state)
    fold_indices = rng.permutation(n_samples) % n_folds

    cv_scores = np.zeros((n_folds, max_k))

    for fold in range(n_folds):
        train_mask = fold_indices != fold
        test_mask = fold_indices == fold

        X_train = matrix[train_mask]
        X_test = matrix[test_mask]

        # Fit POD on training data
        pod = POD(center=True).fit(X_train)

        for k in range(1, max_k + 1):
            if k > pod.n_components_:
                cv_scores[fold, k - 1] = float('inf')
                continue

            # Evaluate on test data
            error = pod.reconstruction_error(X_test, n_modes=k, metric=metric)
            cv_scores[fold, k - 1] = error

    # Average across folds
    mean_scores = np.mean(cv_scores, axis=0)
    std_scores = np.std(cv_scores, axis=0)

    # Select optimal k (minimum CV error)
    optimal_k = int(np.argmin(mean_scores)) + 1

    # One-standard-error rule: smallest k within 1 SE of minimum
    min_score = mean_scores[optimal_k - 1]
    se = std_scores[optimal_k - 1] / np.sqrt(n_folds)

    for k in range(1, optimal_k):
        if mean_scores[k - 1] <= min_score + se:
            optimal_k_1se = k
            break
    else:
        optimal_k_1se = optimal_k

    return ModeSelectionResult(
        selected_modes=optimal_k,
        method="cross_validation",
        criterion_values=mean_scores,
        optimal_value=float(mean_scores[optimal_k - 1]),
        confidence_interval=(optimal_k_1se, optimal_k),
        details={
            "std_scores": std_scores.tolist(),
            "n_folds": n_folds,
            "optimal_k_1se": optimal_k_1se,
        },
    )


def select_modes_by_reconstruction_gradient(
    pod: POD,
    X: ArrayLike,
    threshold: float = 0.01,
    max_modes: Optional[int] = None,
) -> ModeSelectionResult:
    """Select modes where reconstruction improvement falls below threshold.

    Stops adding modes when relative improvement in RMSE drops below
    the specified threshold.

    Args:
        pod: Fitted POD model.
        X: Data for error evaluation.
        threshold: Relative improvement threshold.
        max_modes: Maximum modes to consider.

    Returns:
        ModeSelectionResult with selected mode count.
    """
    matrix = _as_valid_matrix(X)

    max_k = max_modes if max_modes is not None else pod.n_components_
    max_k = min(max_k, pod.n_components_)

    errors = np.zeros(max_k)
    for k in range(1, max_k + 1):
        errors[k - 1] = pod.reconstruction_error(matrix, n_modes=k, metric="rmse")

    # Compute relative improvements
    improvements = np.zeros(max_k - 1)
    for k in range(1, max_k):
        if errors[k - 1] > 0:
            improvements[k - 1] = (errors[k - 1] - errors[k]) / errors[k - 1]
        else:
            improvements[k - 1] = 0

    # Find first k where improvement drops below threshold
    selected = max_k
    for k in range(len(improvements)):
        if improvements[k] < threshold:
            selected = k + 1
            break

    return ModeSelectionResult(
        selected_modes=selected,
        method="reconstruction_gradient",
        criterion_values=errors,
        optimal_value=threshold,
        details={"improvements": improvements.tolist()},
    )


class AutoModeSelector:
    """Automatic mode selection using ensemble of methods.

    Combines multiple selection criteria to provide robust mode
    count recommendation with uncertainty estimates.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize auto selector.

        Args:
            methods: List of methods to use. Default: all available.
            weights: Weights for each method in consensus.
        """
        available = [
            "energy_95", "energy_99", "elbow", "aic", "bic",
            "optimal_threshold", "parallel_analysis", "cross_validation",
        ]

        self.methods = methods if methods is not None else available[:6]  # Skip expensive ones
        self.weights = weights if weights is not None else [1.0] * len(self.methods)

        if len(self.weights) != len(self.methods):
            raise ValueError("weights must have same length as methods")

    def select(
        self,
        X: ArrayLike,
        pod: Optional[POD] = None,
        random_state: Optional[int] = None,
    ) -> dict:
        """Run all selection methods and return consensus.

        Args:
            X: Data matrix.
            pod: Optional pre-fitted POD model.
            random_state: Random seed for stochastic methods.

        Returns:
            Dictionary with results from each method and consensus.
        """
        matrix = _as_valid_matrix(X)

        if pod is None:
            pod = POD(center=True).fit(matrix)

        results = {}
        selections = []
        weights = []

        for method, weight in zip(self.methods, self.weights):
            try:
                if method == "energy_95":
                    result = select_modes_by_energy(pod.cumulative_energy_, 0.95)
                elif method == "energy_99":
                    result = select_modes_by_energy(pod.cumulative_energy_, 0.99)
                elif method == "elbow":
                    result = select_modes_by_elbow(pod.singular_values_)
                elif method == "aic":
                    result = select_modes_by_aic(matrix)
                elif method == "bic":
                    result = select_modes_by_bic(matrix)
                elif method == "optimal_threshold":
                    result = select_modes_by_optimal_threshold(
                        pod.singular_values_, pod.n_samples_, pod.n_features_
                    )
                elif method == "parallel_analysis":
                    result = select_modes_by_parallel_analysis(
                        matrix, random_state=random_state
                    )
                elif method == "cross_validation":
                    result = select_modes_by_cross_validation(
                        matrix, random_state=random_state
                    )
                else:
                    continue

                results[method] = result
                selections.append(result.selected_modes)
                weights.append(weight)

            except Exception as e:
                results[method] = {"error": str(e)}

        if selections:
            selections = np.array(selections)
            weights = np.array(weights) / np.sum(weights)

            # Weighted average
            consensus_mean = np.average(selections, weights=weights)
            consensus_std = np.sqrt(np.average((selections - consensus_mean)**2, weights=weights))

            # Robust consensus (median)
            consensus_median = int(np.median(selections))

            results["consensus"] = {
                "weighted_mean": float(consensus_mean),
                "recommended": int(np.round(consensus_mean)),
                "median": consensus_median,
                "std": float(consensus_std),
                "min": int(np.min(selections)),
                "max": int(np.max(selections)),
                "all_selections": selections.tolist(),
            }

        return results


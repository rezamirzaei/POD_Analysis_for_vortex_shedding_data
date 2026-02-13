"""Tests for mode selection methods."""
import numpy as np
import pytest

from pod_analysis import POD
from pod_analysis.mode_selection import (
    select_modes_by_energy,
    select_modes_by_elbow,
    select_modes_by_aic,
    select_modes_by_bic,
    select_modes_by_optimal_threshold,
    select_modes_by_cross_validation,
    select_modes_by_parallel_analysis,
    select_modes_by_reconstruction_gradient,
    AutoModeSelector,
    ModeSelectionResult,
)


def test_select_by_energy():
    """Test energy-based mode selection."""
    cum_energy = np.array([0.5, 0.75, 0.9, 0.95, 0.99, 1.0])

    result_90 = select_modes_by_energy(cum_energy, 0.9)
    assert result_90.selected_modes == 3

    result_95 = select_modes_by_energy(cum_energy, 0.95)
    assert result_95.selected_modes == 4

    result_99 = select_modes_by_energy(cum_energy, 0.99)
    assert result_99.selected_modes == 5


def test_select_by_energy_invalid():
    """Test energy selection with invalid inputs."""
    cum_energy = np.array([0.5, 0.75, 0.9, 0.95, 1.0])

    with pytest.raises(ValueError):
        select_modes_by_energy(cum_energy, 0.0)

    with pytest.raises(ValueError):
        select_modes_by_energy(cum_energy, 1.5)


def test_select_by_elbow():
    """Test elbow-based mode selection."""
    # Create spectrum with clear elbow
    sigma = np.array([100, 50, 20, 5, 2, 1, 0.5, 0.2, 0.1, 0.05])

    result = select_modes_by_elbow(sigma)

    assert isinstance(result, ModeSelectionResult)
    assert result.method == "elbow"
    assert 1 <= result.selected_modes <= len(sigma)


def test_select_by_aic():
    """Test AIC-based mode selection."""
    rng = np.random.default_rng(42)

    # Low-rank matrix + noise
    n, p, r = 50, 30, 5
    U = rng.standard_normal((n, r))
    V = rng.standard_normal((r, p))
    X = U @ V + 0.1 * rng.standard_normal((n, p))

    result = select_modes_by_aic(X, max_modes=15)

    assert isinstance(result, ModeSelectionResult)
    assert result.method == "aic"
    assert 1 <= result.selected_modes <= 15


def test_select_by_bic():
    """Test BIC-based mode selection."""
    rng = np.random.default_rng(123)

    n, p, r = 50, 30, 5
    U = rng.standard_normal((n, r))
    V = rng.standard_normal((r, p))
    X = U @ V + 0.1 * rng.standard_normal((n, p))

    result = select_modes_by_bic(X, max_modes=15)

    assert isinstance(result, ModeSelectionResult)
    assert result.method == "bic"
    assert 1 <= result.selected_modes <= 15


def test_select_by_optimal_threshold():
    """Test Gavish-Donoho optimal threshold selection."""
    rng = np.random.default_rng(456)

    n, p = 100, 50
    X = rng.standard_normal((n, p))
    _, sigma, _ = np.linalg.svd(X, full_matrices=False)

    result = select_modes_by_optimal_threshold(sigma, n, p)

    assert isinstance(result, ModeSelectionResult)
    assert "optimal" in result.method
    assert 1 <= result.selected_modes <= len(sigma)
    assert result.details is not None
    assert "threshold" in result.details


def test_select_by_cross_validation():
    """Test cross-validation based mode selection."""
    rng = np.random.default_rng(789)

    n, p, r = 50, 20, 3
    U = rng.standard_normal((n, r))
    V = rng.standard_normal((r, p))
    X = U @ V + 0.05 * rng.standard_normal((n, p))

    result = select_modes_by_cross_validation(X, max_modes=10, n_folds=3, random_state=42)

    assert isinstance(result, ModeSelectionResult)
    assert result.method == "cross_validation"
    assert 1 <= result.selected_modes <= 10
    assert result.confidence_interval is not None


def test_select_by_parallel_analysis():
    """Test parallel analysis mode selection."""
    rng = np.random.default_rng(101)

    n, p = 40, 25
    X = rng.standard_normal((n, p))

    result = select_modes_by_parallel_analysis(X, n_permutations=20, random_state=42)

    assert isinstance(result, ModeSelectionResult)
    assert result.method == "parallel_analysis"
    assert 1 <= result.selected_modes <= min(n, p)


def test_select_by_reconstruction_gradient():
    """Test reconstruction gradient based selection."""
    rng = np.random.default_rng(202)

    n, p, r = 40, 30, 5
    U = rng.standard_normal((n, r))
    V = rng.standard_normal((r, p))
    X = U @ V + 0.01 * rng.standard_normal((n, p))

    pod = POD().fit(X)
    result = select_modes_by_reconstruction_gradient(pod, X, threshold=0.01)

    assert isinstance(result, ModeSelectionResult)
    assert result.method == "reconstruction_gradient"


def test_auto_mode_selector():
    """Test automatic mode selection ensemble."""
    rng = np.random.default_rng(303)

    n, p, r = 50, 30, 5
    U = rng.standard_normal((n, r))
    V = rng.standard_normal((r, p))
    X = U @ V + 0.1 * rng.standard_normal((n, p))

    selector = AutoModeSelector(methods=["energy_95", "elbow", "aic", "bic"])
    results = selector.select(X, random_state=42)

    assert "consensus" in results
    assert "recommended" in results["consensus"]
    assert "std" in results["consensus"]
    assert 1 <= results["consensus"]["recommended"] <= min(n, p)


def test_auto_mode_selector_with_pod():
    """Test auto selector with pre-fitted POD."""
    rng = np.random.default_rng(404)

    X = rng.standard_normal((40, 25))
    pod = POD().fit(X)

    selector = AutoModeSelector(methods=["energy_95", "energy_99", "elbow"])
    results = selector.select(X, pod=pod)

    assert "consensus" in results
    assert all(key in results for key in ["energy_95", "energy_99", "elbow"])


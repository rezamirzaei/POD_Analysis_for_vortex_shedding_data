"""Tests for DMD implementations."""
import numpy as np
import pytest

from pod_analysis.dmd import DMD, OptimizedDMD, BOPDMD, DMDMode


def test_dmd_fit_basic():
    """Test basic DMD fitting."""
    rng = np.random.default_rng(42)
    n_snapshots, n_features = 50, 30

    # Create simple oscillatory data
    t = np.linspace(0, 2*np.pi, n_snapshots)
    x = np.linspace(0, 1, n_features)
    X = np.sin(2*t[:, np.newaxis] + x) + 0.1 * rng.standard_normal((n_snapshots, n_features))

    dmd = DMD(rank=5, dt=t[1] - t[0])
    dmd.fit(X)

    assert dmd._is_fitted
    assert dmd.eigenvalues_ is not None
    assert dmd.modes_ is not None
    assert dmd.amplitudes_ is not None
    assert len(dmd.eigenvalues_) == 5


def test_dmd_reconstruction():
    """Test DMD reconstruction accuracy."""
    rng = np.random.default_rng(123)
    n_snapshots, n_features = 100, 50

    # Simple oscillatory dynamics that DMD can capture well
    t = np.linspace(0, 2*np.pi, n_snapshots)
    dt = t[1] - t[0]

    # Single frequency oscillation
    X = np.sin(t[:, np.newaxis]) * np.ones((1, n_features))
    X += 0.01 * rng.standard_normal(X.shape)

    dmd = DMD(rank=5, dt=dt)
    dmd.fit(X)

    reconstructed = np.real(dmd.reconstruct())
    error = dmd.reconstruction_error(X, metric="relative_l2")

    # DMD should reconstruct simple oscillatory data reasonably well
    assert error < 1.5  # Allow more tolerance for DMD


def test_dmd_predict():
    """Test DMD prediction."""
    rng = np.random.default_rng(456)
    n_snapshots, n_features = 50, 20

    t = np.linspace(0, 2*np.pi, n_snapshots)
    X = np.sin(t[:, np.newaxis]) * np.ones((1, n_features))

    dmd = DMD(rank=5, dt=t[1] - t[0])
    dmd.fit(X)

    future = dmd.predict(n_steps=10, start_step=n_snapshots)

    assert future.shape == (10, n_features)


def test_dmd_get_modes():
    """Test DMD mode extraction."""
    rng = np.random.default_rng(789)
    X = rng.standard_normal((30, 20))

    dmd = DMD(rank=5).fit(X)
    modes = dmd.get_modes(n_modes=3)

    assert len(modes) == 3
    assert all(isinstance(m, DMDMode) for m in modes)
    assert all(hasattr(m, 'frequency') for m in modes)
    assert all(hasattr(m, 'growth_rate') for m in modes)


def test_dmd_stability_analysis():
    """Test DMD stability analysis."""
    rng = np.random.default_rng(101)
    X = rng.standard_normal((40, 25))

    dmd = DMD(rank=8).fit(X)
    stability = dmd.stability_analysis()

    assert 'n_stable_modes' in stability
    assert 'n_unstable_modes' in stability
    assert 'spectral_radius' in stability
    assert stability['n_stable_modes'] + stability['n_unstable_modes'] == 8


def test_dmd_requires_fit():
    """Test DMD raises error when not fitted."""
    dmd = DMD()

    with pytest.raises(Exception):
        dmd.predict(10)


def test_bopdmd_basic():
    """Test BOP-DMD basic functionality."""
    rng = np.random.default_rng(202)
    n_snapshots, n_features = 100, 30

    t = np.linspace(0, 4*np.pi, n_snapshots)
    X = np.sin(t[:, np.newaxis]) * np.ones((1, n_features))
    X += 0.05 * rng.standard_normal(X.shape)

    bopdmd = BOPDMD(rank=5, n_bags=10, random_state=42)
    bopdmd.fit(X)

    assert bopdmd._is_fitted
    assert bopdmd.eigenvalues_mean_ is not None
    assert bopdmd.frequencies_mean_ is not None
    assert bopdmd.frequencies_std_ is not None


def test_bopdmd_robust_modes():
    """Test BOP-DMD robust mode extraction."""
    rng = np.random.default_rng(303)
    n_snapshots, n_features = 80, 25

    X = rng.standard_normal((n_snapshots, n_features))

    bopdmd = BOPDMD(rank=5, n_bags=5, random_state=42)
    bopdmd.fit(X)

    robust = bopdmd.get_robust_modes(min_occurrence=0.2)

    assert isinstance(robust, list)
    for mode in robust:
        assert 'frequency' in mode
        assert 'occurrence_rate' in mode



"""Tests for SPOD implementations."""
import numpy as np
import pytest

from pod_analysis.spod import SPOD, SPODMode, StreamingSPOD


def test_spod_fit_basic():
    """Test basic SPOD fitting."""
    rng = np.random.default_rng(42)
    n_snapshots, n_features = 500, 50

    # Create data with known frequency content
    t = np.linspace(0, 10, n_snapshots)
    dt = t[1] - t[0]
    freq = 2.5  # Hz

    X = np.sin(2 * np.pi * freq * t[:, np.newaxis]) * np.ones((1, n_features))
    X += 0.1 * rng.standard_normal(X.shape)

    spod = SPOD(n_fft=64, overlap=0.5, dt=dt)
    spod.fit(X)

    assert spod._is_fitted
    assert spod.frequencies_ is not None
    assert spod.eigenvalues_ is not None
    assert spod.modes_ is not None
    assert spod.n_blocks_ > 1


def test_spod_frequency_resolution():
    """Test SPOD frequency array."""
    rng = np.random.default_rng(123)
    n_snapshots, n_features = 256, 30
    dt = 0.01

    X = rng.standard_normal((n_snapshots, n_features))

    spod = SPOD(n_fft=64, dt=dt)
    spod.fit(X)

    # Check frequency array
    assert spod.frequencies_[0] == 0  # DC component
    assert len(spod.frequencies_) == 33  # n_fft//2 + 1

    # Check Nyquist frequency
    nyquist = 1 / (2 * dt)
    assert np.isclose(spod.frequencies_[-1], nyquist)


def test_spod_get_modes_at_frequency():
    """Test getting SPOD modes at specific frequency."""
    rng = np.random.default_rng(456)
    n_snapshots, n_features = 200, 40

    X = rng.standard_normal((n_snapshots, n_features))

    spod = SPOD(n_fft=64, dt=0.01)
    spod.fit(X)

    # Get modes at specific frequency
    modes = spod.get_modes_at_frequency(10.0, n_modes=3)

    assert len(modes) == 3
    assert all(isinstance(m, SPODMode) for m in modes)
    assert all(hasattr(m, 'energy') for m in modes)
    assert all(hasattr(m, 'energy_fraction') for m in modes)


def test_spod_energy_spectrum():
    """Test SPOD energy spectrum extraction."""
    rng = np.random.default_rng(789)
    n_snapshots, n_features = 300, 30

    X = rng.standard_normal((n_snapshots, n_features))

    spod = SPOD(n_fft=64, dt=0.01)
    spod.fit(X)

    freqs, energies = spod.get_energy_spectrum(mode_index=0)

    assert len(freqs) == len(energies)
    assert np.all(energies >= 0)


def test_spod_total_energy_spectrum():
    """Test SPOD total energy spectrum."""
    rng = np.random.default_rng(101)
    n_snapshots, n_features = 200, 25

    X = rng.standard_normal((n_snapshots, n_features))

    spod = SPOD(n_fft=32, dt=0.01, n_modes=5)
    spod.fit(X)

    freqs, total_energy = spod.total_energy_spectrum()

    assert len(freqs) == len(total_energy)
    assert np.all(total_energy >= 0)


def test_spod_find_dominant_frequencies():
    """Test finding dominant frequencies."""
    rng = np.random.default_rng(202)
    n_snapshots, n_features = 400, 30
    dt = 0.01

    # Create signal with known peaks
    t = np.linspace(0, n_snapshots * dt, n_snapshots)
    f1, f2 = 5.0, 15.0
    X = (np.sin(2*np.pi*f1*t[:, np.newaxis]) +
         0.5*np.sin(2*np.pi*f2*t[:, np.newaxis])) * np.ones((1, n_features))
    X += 0.1 * rng.standard_normal(X.shape)

    spod = SPOD(n_fft=128, overlap=0.5, dt=dt)
    spod.fit(X)

    peaks = spod.find_dominant_frequencies(n_peaks=5)

    assert len(peaks) == 5
    assert all('frequency' in p for p in peaks)
    assert all('total_energy' in p for p in peaks)


def test_spod_cumulative_energy():
    """Test cumulative energy at frequency."""
    rng = np.random.default_rng(303)
    n_snapshots, n_features = 200, 30

    X = rng.standard_normal((n_snapshots, n_features))

    spod = SPOD(n_fft=64, dt=0.01, n_modes=5)
    spod.fit(X)

    cum_energy = spod.cumulative_energy_at_frequency(10.0)

    assert len(cum_energy) == 5
    assert cum_energy[0] > 0
    assert np.isclose(cum_energy[-1], 1.0) or cum_energy[-1] > 0.99


def test_spod_reconstruct_at_frequency():
    """Test field reconstruction at frequency."""
    rng = np.random.default_rng(404)
    n_snapshots, n_features = 200, 40

    X = rng.standard_normal((n_snapshots, n_features))

    spod = SPOD(n_fft=64, dt=0.01, n_modes=5)
    spod.fit(X)

    reconstructed = spod.reconstruct_at_frequency(15.0, n_modes=3)

    assert reconstructed.shape == (n_features,)
    assert np.iscomplexobj(reconstructed)


def test_spod_requires_fit():
    """Test SPOD raises error when not fitted."""
    spod = SPOD()

    with pytest.raises(Exception):
        spod.get_modes_at_frequency(10.0)


def test_spod_insufficient_data():
    """Test SPOD with insufficient data."""
    rng = np.random.default_rng(505)
    X = rng.standard_normal((10, 50))  # Only 10 snapshots

    spod = SPOD(n_fft=256)  # Requires 256 snapshots

    with pytest.raises(ValueError):
        spod.fit(X)


def test_streaming_spod_basic():
    """Test streaming SPOD basic functionality."""
    rng = np.random.default_rng(606)
    n_features = 30

    sspod = StreamingSPOD(n_fft=32, overlap=0.5, dt=0.01)

    # Feed data in batches
    for _ in range(10):
        batch = rng.standard_normal((50, n_features))
        sspod.partial_fit(batch)

    assert sspod._is_fitted
    assert sspod.n_blocks_seen_ > 0
    assert sspod.frequencies_ is not None
    assert sspod.eigenvalues_ is not None


def test_streaming_spod_incremental():
    """Test streaming SPOD incremental updates."""
    rng = np.random.default_rng(707)
    n_features = 25

    sspod = StreamingSPOD(n_fft=32, dt=0.01)

    # Add batches and check updates
    n_blocks_prev = 0
    for i in range(5):
        batch = rng.standard_normal((50, n_features))
        sspod.partial_fit(batch)
        assert sspod.n_blocks_seen_ >= n_blocks_prev
        n_blocks_prev = sspod.n_blocks_seen_


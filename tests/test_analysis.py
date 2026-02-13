import numpy as np
import pytest

from pod_analysis.analysis import compression_ratio, dominant_frequencies, dominant_frequency


def test_dominant_frequency_finds_known_sine_wave() -> None:
    dt = 0.01
    t = np.arange(0, 10, dt)
    expected_freq = 1.5
    signal = np.sin(2 * np.pi * expected_freq * t)

    frequency, amplitude = dominant_frequency(signal, dt=dt)

    assert frequency == pytest.approx(expected_freq, abs=0.05)
    assert amplitude > 0


def test_dominant_frequencies_for_multiple_modes() -> None:
    dt = 0.02
    t = np.arange(0, 20, dt)
    coeffs = np.column_stack(
        [
            np.sin(2 * np.pi * 0.8 * t),
            np.sin(2 * np.pi * 1.6 * t),
            np.sin(2 * np.pi * 2.4 * t),
        ]
    )
    peaks = dominant_frequencies(coeffs, dt=dt, max_modes=2)

    assert len(peaks) == 2
    assert peaks[0].frequency == pytest.approx(0.8, abs=0.05)
    assert peaks[1].frequency == pytest.approx(1.6, abs=0.05)


def test_compression_ratio_improves_with_fewer_modes() -> None:
    low_rank_ratio = compression_ratio(n_snapshots=151, n_spatial_points=89351, n_modes=8)
    high_rank_ratio = compression_ratio(n_snapshots=151, n_spatial_points=89351, n_modes=80)
    assert low_rank_ratio > high_rank_ratio
    assert low_rank_ratio > 1

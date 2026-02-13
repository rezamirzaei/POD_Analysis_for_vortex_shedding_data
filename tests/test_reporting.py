from pod_analysis.reporting import engineering_metrics, spectral_peak_lines


def test_engineering_metrics_extracts_expected_fields() -> None:
    summary = {
        "pod": {
            "selected_modes": 5,
            "selected_energy_percent": 95.12345,
            "reconstruction_rmse": 0.1234567,
            "reconstruction_relative_l2": 0.2345678,
            "mode_counts_for_energy": {"90_percent": 4, "95_percent": 5, "99_percent": 7},
            "compression_ratio_selected": 25.9876,
            "top_energies_percent": [40.1, 39.8, 5.2, 5.0, 3.0, 2.0, 1.0],
        }
    }

    metrics = engineering_metrics(summary, top_modes=4)

    assert metrics["selected_modes"] == 5
    assert metrics["top_energies_percent"] == [40.1, 39.8, 5.2, 5.0]
    assert metrics["compression_ratio_selected"] == 25.988


def test_spectral_peak_lines_formats_cleanly() -> None:
    peaks = [
        {"mode": 1, "frequency": 0.031234, "amplitude": 1234.56},
        {"mode": 2, "frequency": 0.061111, "amplitude": 987.65},
    ]

    lines = spectral_peak_lines(peaks)
    assert lines[0] == "Mode 1: f=0.031234, amp=1234.56"
    assert lines[1] == "Mode 2: f=0.061111, amp=987.65"

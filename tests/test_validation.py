import numpy as np
import pytest

from pod_analysis import (
    POD,
    snapshot_method_singular_value_error,
    theoretical_truncation_rmse,
    validate_pod_consistency,
    validate_pod_model,
)
from pod_analysis.core import NotFittedError


def test_validate_pod_consistency_passes_on_fitted_matrix() -> None:
    rng = np.random.default_rng(31)
    matrix = rng.standard_normal((24, 16))
    pod = POD().fit(matrix)

    report = validate_pod_consistency(pod, matrix, max_modes=10)
    assert report.passed
    assert report.cumulative_energy_is_monotonic
    assert report.rmse_curve_is_monotonic


def test_validate_pod_consistency_requires_fit() -> None:
    pod = POD()
    matrix = np.eye(4)
    with pytest.raises(NotFittedError):
        validate_pod_consistency(pod, matrix)


def test_validate_pod_consistency_checks_input_shape() -> None:
    pod = POD().fit(np.eye(5))
    bad_matrix = np.eye(6)
    with pytest.raises(ValueError):
        validate_pod_consistency(pod, bad_matrix)


def test_theoretical_truncation_rmse_matches_empirical_rmse() -> None:
    rng = np.random.default_rng(32)
    matrix = rng.standard_normal((30, 20))
    pod = POD().fit(matrix)
    n_modes = 7

    theory = theoretical_truncation_rmse(pod, n_modes=n_modes)
    empirical = pod.reconstruction_error(matrix, n_modes=n_modes, metric="rmse")
    assert theory == pytest.approx(empirical, abs=1e-12)


def test_snapshot_method_matches_direct_svd_singular_values() -> None:
    rng = np.random.default_rng(33)
    matrix = rng.standard_normal((25, 18))
    pod = POD().fit(matrix)

    rel_error, compared = snapshot_method_singular_value_error(pod, matrix, n_compare_modes=10)
    assert compared == 10
    assert rel_error < 1e-10


def test_validate_pod_model_bundle_passes() -> None:
    rng = np.random.default_rng(34)
    matrix = rng.standard_normal((28, 15))
    pod = POD().fit(matrix)
    selected_modes = 6

    summary = validate_pod_model(
        pod=pod,
        X=matrix,
        selected_modes=selected_modes,
        max_modes=12,
        atol=1e-8,
        snapshot_rtol=1e-8,
    )

    assert summary.passed
    assert summary.theoretical_rmse == pytest.approx(summary.empirical_rmse, abs=1e-12)

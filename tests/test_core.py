import numpy as np
import pytest

from pod_analysis.core import NotFittedError, POD


def test_low_rank_reconstruction_is_exact_with_true_rank() -> None:
    rng = np.random.default_rng(7)
    n_snapshots, n_features, true_rank = 40, 60, 3
    left = rng.standard_normal((n_snapshots, true_rank))
    right = rng.standard_normal((true_rank, n_features))
    matrix = left @ right + 2.5

    pod = POD().fit(matrix)
    reconstructed = pod.reconstruct(n_modes=true_rank)

    assert np.allclose(matrix, reconstructed, atol=1e-10)
    assert pod.modes_for_energy(0.99) <= true_rank


def test_reconstruction_error_decreases_as_modes_increase() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((30, 20))
    pod = POD().fit(matrix)

    rmse_2 = pod.reconstruction_error(matrix, n_modes=2, metric="rmse")
    rmse_8 = pod.reconstruction_error(matrix, n_modes=8, metric="rmse")

    assert rmse_8 <= rmse_2


def test_transform_and_inverse_transform_match_reconstruct() -> None:
    rng = np.random.default_rng(13)
    matrix = rng.standard_normal((18, 12))
    pod = POD().fit(matrix)

    coeffs = pod.transform(matrix, n_modes=5)
    recovered = pod.inverse_transform(coeffs, n_modes=5)
    direct = pod.reconstruct(n_modes=5)

    assert np.allclose(recovered, direct, atol=1e-10)


def test_validation_errors() -> None:
    pod = POD()
    with pytest.raises(NotFittedError):
        pod.reconstruct()

    with pytest.raises(ValueError):
        pod.fit(np.array([1.0, 2.0, 3.0]))

    pod.fit(np.eye(5))
    with pytest.raises(ValueError):
        pod.modes_for_energy(1.1)

    with pytest.raises(ValueError):
        pod.reconstruct(n_modes=0)

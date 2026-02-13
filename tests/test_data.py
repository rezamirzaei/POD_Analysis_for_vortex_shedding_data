from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pod_analysis.data import load_vortex_data


def test_load_vortex_data_with_explicit_grid_shape(tmp_path: Path) -> None:
    matrix = np.arange(24, dtype=float).reshape(6, 4)
    file_path = tmp_path / "toy.mat"
    savemat(file_path, {"VORTALL": matrix})

    dataset = load_vortex_data(file_path, grid_shape=(2, 3))

    assert dataset.flattened_fields.shape == (6, 4)
    assert dataset.snapshots.shape == (4, 6)
    assert dataset.grid_shape == (2, 3)
    assert np.allclose(dataset.snapshots[0], matrix[:, 0])


def test_load_vortex_data_infers_square_grid(tmp_path: Path) -> None:
    matrix = np.arange(45, dtype=float).reshape(9, 5)
    file_path = tmp_path / "square.mat"
    savemat(file_path, {"VORTALL": matrix})

    dataset = load_vortex_data(file_path)
    assert dataset.grid_shape == (3, 3)


def test_invalid_grid_shape_raises(tmp_path: Path) -> None:
    matrix = np.arange(24, dtype=float).reshape(6, 4)
    file_path = tmp_path / "bad_grid.mat"
    savemat(file_path, {"VORTALL": matrix})

    with pytest.raises(ValueError):
        load_vortex_data(file_path, grid_shape=(4, 4))

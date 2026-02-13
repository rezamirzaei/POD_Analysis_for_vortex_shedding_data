from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

VORTEX_DATA_URL = "https://github.com/dynamicslab/databook_python/raw/master/DATA/VORTALL.mat"


@dataclass(frozen=True)
class VortexDataset:
    """Container for the Brunton & Kutz cylinder wake dataset."""

    flattened_fields: NDArray[np.float64]
    snapshots: NDArray[np.float64]
    grid_shape: Tuple[int, int]
    variable_name: str

    @property
    def n_spatial_points(self) -> int:
        return int(self.flattened_fields.shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(self.flattened_fields.shape[1])


def ensure_vortex_data(path: str | Path, url: str = VORTEX_DATA_URL) -> Path:
    destination = Path(path)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)
    return destination


def load_vortex_data(
    path: str | Path,
    variable_name: str = "VORTALL",
    grid_shape: Optional[Tuple[int, int]] = None,
) -> VortexDataset:
    mat_path = Path(path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    raw: Dict[str, NDArray[np.float64]] = loadmat(mat_path)
    if variable_name in raw:
        matrix = raw[variable_name]
        selected_name = variable_name
    else:
        selected_name, matrix = _first_numeric_2d_variable(raw)

    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D data matrix in '{selected_name}', found shape {data.shape}."
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("Loaded dataset contains NaN or infinite values.")

    n_spatial = int(data.shape[0])
    shape = grid_shape or _infer_grid_shape(n_spatial)
    if shape[0] * shape[1] != n_spatial:
        raise ValueError(
            f"grid_shape={shape} does not match n_spatial_points={n_spatial}."
        )

    snapshots = data.T
    return VortexDataset(
        flattened_fields=data,
        snapshots=snapshots,
        grid_shape=shape,
        variable_name=selected_name,
    )


def reshape_field(flattened_field: NDArray[np.float64], grid_shape: Tuple[int, int]) -> NDArray[np.float64]:
    field = np.asarray(flattened_field, dtype=np.float64)
    if field.ndim != 1:
        raise ValueError("flattened_field must be a 1D vector.")
    if field.size != grid_shape[0] * grid_shape[1]:
        raise ValueError(
            f"Field length {field.size} does not match grid_shape product {grid_shape[0] * grid_shape[1]}."
        )
    return field.reshape(grid_shape, order="C")


def _first_numeric_2d_variable(raw_mat: Dict[str, NDArray[np.float64]]) -> tuple[str, NDArray[np.float64]]:
    for key, value in raw_mat.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
            return key, value
    raise ValueError("No numeric 2D variable found in .mat file.")


def _infer_grid_shape(n_spatial_points: int) -> Tuple[int, int]:
    known_shapes = {89351: (199, 449)}
    if n_spatial_points in known_shapes:
        return known_shapes[n_spatial_points]

    root = int(np.sqrt(n_spatial_points))
    if root * root == n_spatial_points:
        return root, root

    raise ValueError(
        "Unable to infer grid shape automatically. Pass --grid-shape NY NX explicitly."
    )

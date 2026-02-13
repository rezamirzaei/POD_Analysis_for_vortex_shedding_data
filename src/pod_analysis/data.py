from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any, List
from urllib.request import urlretrieve
import json

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, savemat

VORTEX_DATA_URL = "https://github.com/dynamicslab/databook_python/raw/master/DATA/VORTALL.mat"


@dataclass(frozen=True)
class VortexDataset:
    """Container for the Brunton & Kutz cylinder wake dataset."""

    flattened_fields: NDArray[np.float64]
    snapshots: NDArray[np.float64]
    grid_shape: Tuple[int, int]
    variable_name: str
    metadata: Optional[Dict[str, Any]] = None

    @property
    def n_spatial_points(self) -> int:
        return int(self.flattened_fields.shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(self.flattened_fields.shape[1])


@dataclass(frozen=True)
class SnapshotDataset:
    """Generic container for snapshot data from any source."""

    snapshots: NDArray[np.float64]  # (n_snapshots, n_spatial)
    grid_shape: Optional[Tuple[int, ...]] = None
    coordinates: Optional[NDArray[np.float64]] = None  # (n_spatial, n_dims)
    time_values: Optional[NDArray[np.float64]] = None  # (n_snapshots,)
    variable_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None

    @property
    def n_snapshots(self) -> int:
        return self.snapshots.shape[0]

    @property
    def n_spatial_points(self) -> int:
        return self.snapshots.shape[1]

    @property
    def dt(self) -> Optional[float]:
        """Infer time step from time_values if available."""
        if self.time_values is not None and len(self.time_values) > 1:
            return float(np.mean(np.diff(self.time_values)))
        return None


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


# ============================================================================
# Multi-format data loading
# ============================================================================

def load_snapshot_data(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> SnapshotDataset:
    """Load snapshot data from various formats.

    Supported formats:
    - .mat: MATLAB files
    - .npy, .npz: NumPy arrays
    - .h5, .hdf5: HDF5 files
    - .csv: CSV files
    - .vtk, .vtu, .vtp: VTK files (requires vtk package)

    Args:
        path: Path to data file.
        format: Explicit format override (inferred from extension if None).
        **kwargs: Format-specific options.

    Returns:
        SnapshotDataset with loaded data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Infer format from extension
    if format is None:
        ext = path.suffix.lower()
        format_map = {
            '.mat': 'matlab',
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.csv': 'csv',
            '.vtk': 'vtk',
            '.vtu': 'vtk',
            '.vtp': 'vtk',
        }
        format = format_map.get(ext)
        if format is None:
            raise ValueError(f"Unknown file format: {ext}")

    loaders = {
        'matlab': _load_matlab,
        'numpy': _load_numpy,
        'hdf5': _load_hdf5,
        'csv': _load_csv,
        'vtk': _load_vtk,
    }

    loader = loaders.get(format)
    if loader is None:
        raise ValueError(f"Unsupported format: {format}")

    return loader(path, **kwargs)


def _load_matlab(
    path: Path,
    variable_name: Optional[str] = None,
    transpose: bool = True,
    **kwargs,
) -> SnapshotDataset:
    """Load from MATLAB .mat file."""
    raw = loadmat(path)

    if variable_name is not None:
        if variable_name not in raw:
            raise KeyError(f"Variable '{variable_name}' not found in {path}")
        data = raw[variable_name]
    else:
        _, data = _first_numeric_2d_variable(raw)

    data = np.asarray(data, dtype=np.float64)

    # Standard convention: snapshots as rows
    if transpose and data.shape[0] > data.shape[1]:
        data = data.T

    return SnapshotDataset(
        snapshots=data,
        source_path=str(path),
        metadata={"format": "matlab"},
    )


def _load_numpy(
    path: Path,
    key: Optional[str] = None,
    **kwargs,
) -> SnapshotDataset:
    """Load from NumPy .npy or .npz file."""
    if path.suffix == '.npz':
        with np.load(path) as f:
            if key is not None:
                data = f[key]
            elif 'snapshots' in f:
                data = f['snapshots']
            elif 'data' in f:
                data = f['data']
            else:
                # Use first array
                data = f[list(f.keys())[0]]

            # Load optional metadata
            time_values = f.get('time', None)
            coordinates = f.get('coordinates', None)
    else:
        data = np.load(path)
        time_values = None
        coordinates = None

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return SnapshotDataset(
        snapshots=data,
        time_values=time_values,
        coordinates=coordinates,
        source_path=str(path),
        metadata={"format": "numpy"},
    )


def _load_hdf5(
    path: Path,
    dataset_name: Optional[str] = None,
    group: str = "/",
    **kwargs,
) -> SnapshotDataset:
    """Load from HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 files: pip install h5py")

    with h5py.File(path, 'r') as f:
        grp = f[group]

        if dataset_name is not None:
            data = grp[dataset_name][:]
        elif 'snapshots' in grp:
            data = grp['snapshots'][:]
        elif 'data' in grp:
            data = grp['data'][:]
        else:
            # Use first dataset
            for key in grp.keys():
                if isinstance(grp[key], h5py.Dataset):
                    data = grp[key][:]
                    break
            else:
                raise ValueError("No dataset found in HDF5 file")

        # Load optional arrays
        time_values = grp['time'][:] if 'time' in grp else None
        coordinates = grp['coordinates'][:] if 'coordinates' in grp else None

        # Load attributes as metadata
        metadata = dict(grp.attrs)
        metadata['format'] = 'hdf5'

    return SnapshotDataset(
        snapshots=np.asarray(data, dtype=np.float64),
        time_values=time_values,
        coordinates=coordinates,
        source_path=str(path),
        metadata=metadata,
    )


def _load_csv(
    path: Path,
    delimiter: str = ',',
    skip_header: int = 0,
    transpose: bool = False,
    **kwargs,
) -> SnapshotDataset:
    """Load from CSV file."""
    data = np.loadtxt(
        path,
        delimiter=delimiter,
        skiprows=skip_header,
        dtype=np.float64,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if transpose:
        data = data.T

    return SnapshotDataset(
        snapshots=data,
        source_path=str(path),
        metadata={"format": "csv"},
    )


def _load_vtk(
    path: Path,
    array_name: Optional[str] = None,
    **kwargs,
) -> SnapshotDataset:
    """Load from VTK file."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("vtk required for VTK files: pip install vtk")

    ext = path.suffix.lower()

    if ext == '.vtk':
        reader = vtk.vtkUnstructuredGridReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported VTK format: {ext}")

    reader.SetFileName(str(path))
    reader.Update()

    output = reader.GetOutput()

    # Get point data
    point_data = output.GetPointData()

    if array_name is not None:
        arr = point_data.GetArray(array_name)
        if arr is None:
            raise KeyError(f"Array '{array_name}' not found")
        data = vtk_to_numpy(arr)
    else:
        # Use first array
        if point_data.GetNumberOfArrays() == 0:
            raise ValueError("No point data arrays in VTK file")
        arr = point_data.GetArray(0)
        data = vtk_to_numpy(arr)

    # Get coordinates
    points = output.GetPoints()
    if points is not None:
        coordinates = vtk_to_numpy(points.GetData())
    else:
        coordinates = None

    # Reshape if needed (single snapshot)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        # Flatten spatial dimensions
        data = data.reshape(data.shape[0], -1)

    return SnapshotDataset(
        snapshots=data,
        coordinates=coordinates,
        source_path=str(path),
        metadata={"format": "vtk"},
    )


# ============================================================================
# Data export functions
# ============================================================================

def save_pod_results(
    path: Union[str, Path],
    pod,  # POD model
    format: str = "hdf5",
    include_reconstructions: bool = False,
    n_modes: Optional[int] = None,
    **kwargs,
) -> Path:
    """Save POD results to file.

    Args:
        path: Output path.
        pod: Fitted POD model.
        format: Output format ("hdf5", "matlab", "numpy", "json").
        include_reconstructions: If True, save reconstructed snapshots.
        n_modes: Number of modes to save (default: all).

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    k = n_modes if n_modes is not None else pod.n_components_
    k = min(k, pod.n_components_)

    if format == "hdf5":
        return _save_hdf5(path, pod, k, include_reconstructions, **kwargs)
    elif format == "matlab":
        return _save_matlab(path, pod, k, include_reconstructions, **kwargs)
    elif format == "numpy":
        return _save_numpy(path, pod, k, include_reconstructions, **kwargs)
    elif format == "json":
        return _save_json(path, pod, k, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_hdf5(
    path: Path,
    pod,
    n_modes: int,
    include_reconstructions: bool,
    compression: str = "gzip",
    **kwargs,
) -> Path:
    """Save POD results to HDF5."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    with h5py.File(path, 'w') as f:
        # Modes and coefficients
        f.create_dataset('modes', data=pod.modes_[:, :n_modes], compression=compression)
        f.create_dataset('temporal_coefficients', data=pod.temporal_coeffs_[:, :n_modes], compression=compression)
        f.create_dataset('singular_values', data=pod.singular_values_[:n_modes])
        f.create_dataset('mean', data=pod.mean_, compression=compression)
        f.create_dataset('energy_per_mode', data=pod.energy_per_mode_[:n_modes])
        f.create_dataset('cumulative_energy', data=pod.cumulative_energy_[:n_modes])

        if include_reconstructions:
            reconstructed = pod.reconstruct(n_modes=n_modes)
            f.create_dataset('reconstructed', data=reconstructed, compression=compression)

        # Metadata
        f.attrs['n_samples'] = pod.n_samples_
        f.attrs['n_features'] = pod.n_features_
        f.attrs['n_modes_saved'] = n_modes
        f.attrs['n_components_total'] = pod.n_components_

    return path


def _save_matlab(
    path: Path,
    pod,
    n_modes: int,
    include_reconstructions: bool,
    **kwargs,
) -> Path:
    """Save POD results to MATLAB .mat file."""
    data = {
        'modes': pod.modes_[:, :n_modes],
        'temporal_coefficients': pod.temporal_coeffs_[:, :n_modes],
        'singular_values': pod.singular_values_[:n_modes],
        'mean': pod.mean_,
        'energy_per_mode': pod.energy_per_mode_[:n_modes],
        'cumulative_energy': pod.cumulative_energy_[:n_modes],
        'n_samples': pod.n_samples_,
        'n_features': pod.n_features_,
    }

    if include_reconstructions:
        data['reconstructed'] = pod.reconstruct(n_modes=n_modes)

    savemat(path, data)
    return path


def _save_numpy(
    path: Path,
    pod,
    n_modes: int,
    include_reconstructions: bool,
    **kwargs,
) -> Path:
    """Save POD results to NumPy .npz file."""
    data = {
        'modes': pod.modes_[:, :n_modes],
        'temporal_coefficients': pod.temporal_coeffs_[:, :n_modes],
        'singular_values': pod.singular_values_[:n_modes],
        'mean': pod.mean_,
        'energy_per_mode': pod.energy_per_mode_[:n_modes],
        'cumulative_energy': pod.cumulative_energy_[:n_modes],
    }

    if include_reconstructions:
        data['reconstructed'] = pod.reconstruct(n_modes=n_modes)

    np.savez_compressed(path, **data)
    return path


def _save_json(
    path: Path,
    pod,
    n_modes: int,
    **kwargs,
) -> Path:
    """Save POD summary to JSON (no large arrays)."""
    data = {
        'n_samples': int(pod.n_samples_),
        'n_features': int(pod.n_features_),
        'n_components': int(pod.n_components_),
        'n_modes_saved': int(n_modes),
        'singular_values': pod.singular_values_[:n_modes].tolist(),
        'energy_per_mode': pod.energy_per_mode_[:n_modes].tolist(),
        'cumulative_energy': pod.cumulative_energy_[:n_modes].tolist(),
        'total_energy_captured': float(pod.cumulative_energy_[n_modes - 1]),
    }

    path.write_text(json.dumps(data, indent=2))
    return path


# ============================================================================
# Utility functions
# ============================================================================

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


def create_synthetic_dataset(
    n_snapshots: int = 100,
    n_spatial: int = 1000,
    true_rank: int = 5,
    noise_level: float = 0.01,
    random_state: Optional[int] = None,
) -> SnapshotDataset:
    """Create synthetic low-rank dataset for testing.

    Args:
        n_snapshots: Number of time snapshots.
        n_spatial: Number of spatial points.
        true_rank: True rank of the data.
        noise_level: Noise standard deviation relative to signal.
        random_state: Random seed.

    Returns:
        SnapshotDataset with synthetic data.
    """
    rng = np.random.default_rng(random_state)

    # Create low-rank signal
    U = rng.standard_normal((n_snapshots, true_rank))
    V = rng.standard_normal((true_rank, n_spatial))

    # Decay singular values
    decay = np.exp(-np.arange(true_rank) * 0.5)
    signal = U @ (decay[:, np.newaxis] * V)

    # Add noise
    noise = noise_level * np.std(signal) * rng.standard_normal(signal.shape)
    data = signal + noise

    return SnapshotDataset(
        snapshots=data,
        time_values=np.arange(n_snapshots, dtype=np.float64),
        metadata={
            "synthetic": True,
            "true_rank": true_rank,
            "noise_level": noise_level,
        },
    )

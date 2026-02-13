# POD Analysis Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A comprehensive, production-ready toolkit for Proper Orthogonal Decomposition (POD), Dynamic Mode Decomposition (DMD), and Spectral POD (SPOD) analysis.**

## 🚀 Features

### Core Decomposition Methods
- **POD (Proper Orthogonal Decomposition)**
  - Standard SVD-based POD with full/randomized/truncated solvers
  - Method of Snapshots (efficient for high spatial dimensions)
  - Incremental/Streaming POD for large or online datasets
  - Weighted POD for non-Euclidean inner products
  
- **DMD (Dynamic Mode Decomposition)**
  - Exact DMD with optimal amplitude computation
  - Optimized DMD with variable projection
  - BOP-DMD (Bagging Optimized DMD) for robust mode extraction
  - Stability analysis and prediction capabilities
  
- **SPOD (Spectral POD)**
  - Frequency-resolved coherent structure extraction
  - Streaming SPOD for real-time applications
  - Dominant frequency identification

### Advanced Mode Selection
- Energy-based thresholding (90%, 95%, 99%)
- Elbow/knee detection
- Parallel analysis (Horn's method)
- Gavish-Donoho optimal hard thresholding
- Information criteria (AIC, BIC)
- K-fold cross-validation
- Reconstruction gradient stopping
- Ensemble mode selection with uncertainty

### Data I/O
- **Input formats:** MATLAB (.mat), NumPy (.npy/.npz), HDF5 (.h5), CSV, VTK
- **Export formats:** HDF5, MATLAB, NumPy, JSON
- Automatic format detection
- Metadata preservation

### Validation & Diagnostics
- POD identity verification (orthonormality, projection, Parseval)
- Theoretical vs empirical RMSE comparison
- Method of snapshots consistency check
- Comprehensive validation reports

## 📦 Installation

```bash
# Basic installation
pip install pod-analysis

# With HDF5 support
pip install pod-analysis[hdf5]

# Full installation (all optional dependencies)
pip install pod-analysis[all]

# Development installation
git clone https://github.com/pod-analysis/pod-analysis.git
cd pod-analysis
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Simple POD Analysis

```python
from pod_analysis import POD

# Fit POD to snapshot data (n_snapshots × n_spatial)
pod = POD(svd_solver="auto").fit(snapshots)

# Get reduced coordinates
reduced = pod.transform(snapshots, n_modes=10)

# Reconstruct with specified modes
reconstructed = pod.reconstruct(n_modes=10)

# Find modes for 95% energy
n_modes_95 = pod.modes_for_energy(0.95)

# Compute reconstruction error
rmse = pod.reconstruction_error(snapshots, n_modes=10)
```

### Full Analysis Workflow

```python
from pod_analysis import AnalysisConfig, run_analysis

config = AnalysisConfig(
    data_path="vortex_data.mat",
    output_dir="results",
    energy_target=0.95,
    generate_plots=True,
)

result = run_analysis(config)

print(f"Selected modes: {result.selected_modes}")
print(f"Captured energy: {result.summary['pod']['selected_energy_percent']:.1f}%")
```

### Dynamic Mode Decomposition

```python
from pod_analysis import DMD

dmd = DMD(rank=20, dt=0.01).fit(snapshots)

# Predict future states
future = dmd.predict(n_steps=100)

# Get dominant modes
modes = dmd.get_modes(n_modes=5)
for mode in modes:
    print(f"Frequency: {mode.frequency:.2f} Hz, Growth: {mode.growth_rate:.4f}")

# Stability analysis
stability = dmd.stability_analysis()
```

### Spectral POD

```python
from pod_analysis import SPOD

spod = SPOD(n_fft=256, overlap=0.5, dt=0.001).fit(snapshots)

# Get modes at specific frequency
modes_50hz = spod.get_modes_at_frequency(50.0)

# Find dominant frequencies
peaks = spod.find_dominant_frequencies(n_peaks=5)
```

### Advanced Mode Selection

```python
from pod_analysis import (
    AutoModeSelector,
    select_modes_by_cross_validation,
    select_modes_by_optimal_threshold,
)

# Automatic selection using ensemble of methods
selector = AutoModeSelector()
results = selector.select(snapshots)
print(f"Recommended modes: {results['consensus']['recommended']}")

# Cross-validation based selection
cv_result = select_modes_by_cross_validation(snapshots, n_folds=5)
print(f"CV optimal: {cv_result.selected_modes}")

# Gavish-Donoho optimal threshold
od_result = select_modes_by_optimal_threshold(
    pod.singular_values_, pod.n_samples_, pod.n_features_
)
```

### Incremental/Streaming POD

```python
from pod_analysis import IncrementalPOD

ipod = IncrementalPOD(n_components=20)

# Process data in batches
for batch in data_loader:
    ipod.partial_fit(batch)

# Use like regular POD
reduced = ipod.transform(new_data)
```

### Multi-Format Data Loading

```python
from pod_analysis import load_snapshot_data, save_pod_results

# Auto-detect format from extension
data = load_snapshot_data("simulation.h5")
data = load_snapshot_data("results.mat")
data = load_snapshot_data("snapshots.csv")

# Export results
save_pod_results("pod_results.h5", pod, format="hdf5", n_modes=20)
save_pod_results("pod_results.mat", pod, format="matlab")
```

## 🖥️ Command Line Interface

```bash
# Basic analysis
pod-analyze --data vortex_data.mat --output-dir results

# With specific settings
pod-analyze --data simulation.h5 \
            --modes 20 \
            --energy-target 0.99 \
            --dt 0.001 \
            --output-dir results

# Download sample data if missing
pod-analyze --download-if-missing --output-dir results
```

## 📊 Output Files

The toolkit generates:

- `summary.json` - Comprehensive analysis results
- `energy_spectrum.png` - Singular value and cumulative energy plots
- `spatial_modes.png` - Leading POD mode visualizations
- `reconstructions.png` - Original vs reconstructed snapshots
- `temporal_coefficients.png` - Time evolution of mode coefficients

## 📓 Notebooks

- `pod_analysis.ipynb` - Main POD workflow notebook with rich theory + strict validation checks.
- `advanced_modal_analysis.ipynb` - Advanced module notebook for mode selection, DMD, and SPOD using source APIs.

## 🧪 Validation

```python
from pod_analysis import validate_pod_model

validation = validate_pod_model(
    pod=pod,
    X=snapshots,
    selected_modes=10,
    atol=1e-8,
)

assert validation.passed, "POD validation failed"
print(validation.to_dict())
```

## 📚 Method Comparison

| Method | Best For | Complexity | Memory |
|--------|----------|------------|--------|
| POD (SVD) | General purpose | O(mn·min(m,n)) | O(mn) |
| POD (Randomized) | Large n_features | O(mn·k) | O(mk + nk) |
| Method of Snapshots | n_spatial >> n_snapshots | O(n_t³ + n_t²·n_x) | O(n_t²) |
| Incremental POD | Streaming/large data | O(mk·n_batch) | O(mk + nk) |
| DMD | Temporal dynamics | O(mn·r) | O(mn) |
| SPOD | Frequency analysis | O(n_blocks·n_fft·n_x) | O(n_freq·n_x²) |

## 🔬 Theory

### POD Formulation

Given snapshot matrix X ∈ ℝⁿᵗˣⁿˣ:

1. Mean-center: X̃ = X - X̄
2. SVD: X̃ = UΣVᵀ
3. Modes: Φ = V, Coefficients: A = UΣ
4. Energy: Eᵢ = σᵢ² / Σⱼσⱼ²

### DMD Formulation

Find best-fit linear operator A: X' ≈ AX

1. SVD of X: X = UΣVᵀ
2. Projected operator: Ã = Uᵀ X' V Σ⁻¹
3. Eigendecomposition: ÃW = WΛ
4. DMD modes: Φ = X' V Σ⁻¹ W

## 📖 References

- Brunton & Kutz, *Data-Driven Science and Engineering* (2019)
- Holmes et al., *Turbulence, Coherent Structures, Dynamical Systems and Symmetry* (2012)
- Schmid, *Dynamic mode decomposition of numerical and experimental data* (2010)
- Towne, Schmidt & Colonius, *Spectral proper orthogonal decomposition* (2018)
- Gavish & Donoho, *The Optimal Hard Threshold for Singular Values is 4/√3* (2014)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

"""POD analysis toolkit for vortex street datasets.

A comprehensive, production-ready toolkit for Proper Orthogonal Decomposition
and related modal analysis techniques.

Features:
    - Multiple POD implementations: Standard SVD, Method of Snapshots, Incremental
    - Dynamic Mode Decomposition (DMD) with variants: Exact, Optimized, BOP-DMD
    - Spectral POD (SPOD) for frequency-resolved analysis
    - Advanced mode selection: Cross-validation, information criteria, optimal thresholds
    - Multi-format data I/O: MATLAB, HDF5, NumPy, CSV, VTK
    - Comprehensive validation and diagnostics
    - Publication-quality plotting

Quick Start:
    >>> from pod_analysis import POD, run_analysis, AnalysisConfig
    >>>
    >>> # Simple API
    >>> pod = POD().fit(snapshots)
    >>> reduced = pod.transform(new_data, n_modes=10)
    >>>
    >>> # Full workflow
    >>> config = AnalysisConfig(data_path="data.mat", output_dir="results")
    >>> result = run_analysis(config)

References:
    - Brunton & Kutz, "Data-Driven Science and Engineering"
    - Holmes et al., "Turbulence, Coherent Structures, Dynamical Systems"
    - Schmid, "Dynamic mode decomposition of numerical and experimental data"
    - Towne, Schmidt & Colonius, "Spectral proper orthogonal decomposition"
"""

__version__ = "1.0.0"

# Core POD implementations
from .core import (
    POD,
    IncrementalPOD,
    MethodOfSnapshots,
    SVDSolver,
    NotFittedError,
)

# DMD implementations
from .dmd import (
    DMD,
    DMDMode,
    OptimizedDMD,
    BOPDMD,
)

# SPOD implementations
from .spod import (
    SPOD,
    SPODMode,
    StreamingSPOD,
)

# Mode selection methods
from .mode_selection import (
    ModeSelectionResult,
    AutoModeSelector,
    select_modes_by_energy,
    select_modes_by_elbow,
    select_modes_by_parallel_analysis,
    select_modes_by_optimal_threshold,
    select_modes_by_aic,
    select_modes_by_bic,
    select_modes_by_cross_validation,
    select_modes_by_reconstruction_gradient,
)

# Analysis utilities
from .analysis import (
    SpectrumPeak,
    compression_ratio,
    dominant_frequencies,
    dominant_frequency,
    mode_count_for_energy,
)

# Data I/O
from .data import (
    VORTEX_DATA_URL,
    VortexDataset,
    SnapshotDataset,
    ensure_vortex_data,
    load_vortex_data,
    load_snapshot_data,
    save_pod_results,
    reshape_field,
    create_synthetic_dataset,
)
from .demo_data import (
    OscillatoryDataset,
    create_oscillatory_dataset,
    create_multitone_dataset,
)

# Validation
from .validation import (
    PODConsistencyReport,
    PODValidationSummary,
    snapshot_method_singular_value_error,
    theoretical_truncation_rmse,
    validate_pod_consistency,
    validate_pod_model,
)

# Reporting
from .reporting import (
    engineering_metrics,
    spectral_peak_lines,
)

# Workflow
from .workflow import (
    AnalysisConfig,
    AnalysisResult,
    PODAnalysisWorkflow,
    run_analysis,
)
from .advanced import (
    AdvancedModalDemo,
    ModeSelectionDemoConfig,
    ModeSelectionDemoResult,
    ModeSelectionOutcome,
    DMDDemoConfig,
    DMDDemoResult,
    DMDModeSummary,
    SPODDemoConfig,
    SPODDemoResult,
    SPODPeakSummary,
    format_mode_selection_result,
    format_dmd_result,
    format_spod_result,
    plot_spod_spectrum,
)

__all__ = [
    # Version
    "__version__",
    # Core POD
    "POD",
    "IncrementalPOD",
    "MethodOfSnapshots",
    "SVDSolver",
    "NotFittedError",
    # DMD
    "DMD",
    "DMDMode",
    "OptimizedDMD",
    "BOPDMD",
    # SPOD
    "SPOD",
    "SPODMode",
    "StreamingSPOD",
    # Mode selection
    "ModeSelectionResult",
    "AutoModeSelector",
    "select_modes_by_energy",
    "select_modes_by_elbow",
    "select_modes_by_parallel_analysis",
    "select_modes_by_optimal_threshold",
    "select_modes_by_aic",
    "select_modes_by_bic",
    "select_modes_by_cross_validation",
    "select_modes_by_reconstruction_gradient",
    # Analysis
    "SpectrumPeak",
    "compression_ratio",
    "dominant_frequencies",
    "dominant_frequency",
    "mode_count_for_energy",
    # Data I/O
    "VORTEX_DATA_URL",
    "VortexDataset",
    "SnapshotDataset",
    "ensure_vortex_data",
    "load_vortex_data",
    "load_snapshot_data",
    "save_pod_results",
    "reshape_field",
    "create_synthetic_dataset",
    "OscillatoryDataset",
    "create_oscillatory_dataset",
    "create_multitone_dataset",
    # Validation
    "PODConsistencyReport",
    "PODValidationSummary",
    "snapshot_method_singular_value_error",
    "theoretical_truncation_rmse",
    "validate_pod_consistency",
    "validate_pod_model",
    # Reporting
    "engineering_metrics",
    "spectral_peak_lines",
    # Workflow
    "AnalysisConfig",
    "AnalysisResult",
    "PODAnalysisWorkflow",
    "run_analysis",
    # Advanced demo orchestration
    "AdvancedModalDemo",
    "ModeSelectionDemoConfig",
    "ModeSelectionDemoResult",
    "ModeSelectionOutcome",
    "DMDDemoConfig",
    "DMDDemoResult",
    "DMDModeSummary",
    "SPODDemoConfig",
    "SPODDemoResult",
    "SPODPeakSummary",
    "format_mode_selection_result",
    "format_dmd_result",
    "format_spod_result",
    "plot_spod_spectrum",
]

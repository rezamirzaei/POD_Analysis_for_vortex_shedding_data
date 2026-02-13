"""POD analysis toolkit for vortex street datasets."""

from .analysis import SpectrumPeak, compression_ratio, dominant_frequencies
from .core import POD
from .data import VORTEX_DATA_URL, VortexDataset, ensure_vortex_data, load_vortex_data
from .reporting import engineering_metrics, spectral_peak_lines
from .validation import (
    PODConsistencyReport,
    PODValidationSummary,
    snapshot_method_singular_value_error,
    theoretical_truncation_rmse,
    validate_pod_consistency,
    validate_pod_model,
)
from .workflow import AnalysisConfig, AnalysisResult, PODAnalysisWorkflow, run_analysis

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "POD",
    "PODAnalysisWorkflow",
    "SpectrumPeak",
    "VORTEX_DATA_URL",
    "VortexDataset",
    "PODConsistencyReport",
    "PODValidationSummary",
    "compression_ratio",
    "dominant_frequencies",
    "engineering_metrics",
    "ensure_vortex_data",
    "load_vortex_data",
    "run_analysis",
    "snapshot_method_singular_value_error",
    "spectral_peak_lines",
    "theoretical_truncation_rmse",
    "validate_pod_consistency",
    "validate_pod_model",
]

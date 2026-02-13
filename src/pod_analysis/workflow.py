from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .analysis import SpectrumPeak, compression_ratio, dominant_frequencies
from .core import POD
from .data import VortexDataset, ensure_vortex_data, load_vortex_data


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for a complete POD analysis run."""

    data_path: Path = Path("vortex_data.mat")
    output_dir: Path = Path("outputs")
    modes: Optional[int] = None
    energy_target: float = 0.95
    grid_shape: Optional[Tuple[int, int]] = None
    snapshot_index: Optional[int] = None
    dt: float = 1.0
    generate_plots: bool = True
    auto_download: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_path", Path(self.data_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass
class AnalysisResult:
    """Artifacts produced by the POD workflow."""

    config: AnalysisConfig
    dataset: VortexDataset
    pod: POD
    selected_modes: int
    snapshot_index: int
    summary: Dict[str, object]
    summary_path: Path
    generated_files: Dict[str, Path] = field(default_factory=dict)
    dominant_peaks: Tuple[SpectrumPeak, ...] = field(default_factory=tuple)


class PODAnalysisWorkflow:
    """Object-oriented orchestrator for POD analysis, reporting, and plotting."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def run(self) -> AnalysisResult:
        data_path = self._prepare_data_path()
        dataset = load_vortex_data(data_path, grid_shape=self.config.grid_shape)
        pod = POD(center=True).fit(dataset.snapshots)

        selected_modes = self._resolve_selected_modes(pod)
        snapshot_index = self._resolve_snapshot_index(dataset.n_snapshots)
        peaks = tuple(dominant_frequencies(pod.temporal_coeffs_, dt=self.config.dt, max_modes=4))
        summary = self._build_summary(dataset, pod, selected_modes, peaks)
        summary_path = self._write_summary(summary)

        generated_files: Dict[str, Path] = {"summary": summary_path}
        if self.config.generate_plots:
            generated_files.update(
                self._generate_plots(dataset, pod, selected_modes, snapshot_index)
            )

        return AnalysisResult(
            config=self.config,
            dataset=dataset,
            pod=pod,
            selected_modes=selected_modes,
            snapshot_index=snapshot_index,
            summary=summary,
            summary_path=summary_path,
            generated_files=generated_files,
            dominant_peaks=peaks,
        )

    def _prepare_data_path(self) -> Path:
        if self.config.auto_download:
            return ensure_vortex_data(self.config.data_path)
        return self.config.data_path

    def _resolve_selected_modes(self, pod: POD) -> int:
        if self.config.modes is not None:
            if self.config.modes < 1 or self.config.modes > pod.n_components_:
                raise ValueError(f"modes must be in [1, {pod.n_components_}].")
            return self.config.modes
        return pod.modes_for_energy(self.config.energy_target)

    def _resolve_snapshot_index(self, n_snapshots: int) -> int:
        if self.config.snapshot_index is None:
            return n_snapshots // 2
        idx = int(self.config.snapshot_index)
        if idx < 0 or idx >= n_snapshots:
            raise ValueError(f"snapshot_index must be in [0, {n_snapshots - 1}].")
        return idx

    def _build_summary(
        self,
        dataset: VortexDataset,
        pod: POD,
        selected_modes: int,
        peaks: Tuple[SpectrumPeak, ...],
    ) -> Dict[str, object]:
        rmse = pod.reconstruction_error(dataset.snapshots, n_modes=selected_modes, metric="rmse")
        rel_l2 = pod.reconstruction_error(
            dataset.snapshots, n_modes=selected_modes, metric="relative_l2"
        )

        summary: Dict[str, object] = {
            "dataset": {
                "path": str(self.config.data_path),
                "variable": dataset.variable_name,
                "grid_shape": [dataset.grid_shape[0], dataset.grid_shape[1]],
                "n_spatial_points": dataset.n_spatial_points,
                "n_snapshots": dataset.n_snapshots,
            },
            "pod": {
                "n_total_modes": pod.n_components_,
                "selected_modes": selected_modes,
                "selected_energy_percent": float(pod.cumulative_energy_[selected_modes - 1] * 100),
                "reconstruction_rmse": rmse,
                "reconstruction_relative_l2": rel_l2,
                "mode_counts_for_energy": {
                    "90_percent": pod.modes_for_energy(0.90),
                    "95_percent": pod.modes_for_energy(0.95),
                    "99_percent": pod.modes_for_energy(0.99),
                },
                "compression_ratio_selected": compression_ratio(
                    dataset.n_snapshots, dataset.n_spatial_points, selected_modes
                ),
                "top_energies_percent": [
                    float(value * 100)
                    for value in pod.energy_per_mode_[: min(10, pod.energy_per_mode_.size)]
                ],
            },
            "spectral_peaks": [
                {
                    "mode": peak.mode_index,
                    "frequency": peak.frequency,
                    "amplitude": peak.amplitude,
                }
                for peak in peaks
            ],
        }
        return summary

    def _write_summary(self, summary: Dict[str, object]) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.config.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary_path

    def _generate_plots(
        self,
        dataset: VortexDataset,
        pod: POD,
        selected_modes: int,
        snapshot_index: int,
    ) -> Dict[str, Path]:
        import matplotlib

        matplotlib.use("Agg")
        from .plotting import (
            plot_energy_spectrum,
            plot_modes_grid,
            plot_snapshot_reconstructions,
            plot_temporal_coefficients,
        )

        energy_path = self.config.output_dir / "energy_spectrum.png"
        modes_path = self.config.output_dir / "spatial_modes.png"
        recon_path = self.config.output_dir / "reconstructions.png"
        temporal_path = self.config.output_dir / "temporal_coefficients.png"

        plot_energy_spectrum(
            singular_values=pod.singular_values_,
            cumulative_energy=pod.cumulative_energy_,
            output_path=energy_path,
            max_modes=80,
        )
        plot_modes_grid(
            spatial_modes=pod.modes_,
            energy_per_mode=pod.energy_per_mode_,
            grid_shape=dataset.grid_shape,
            output_path=modes_path,
            max_modes=12,
        )
        mode_choices = sorted(
            {
                1,
                2,
                4,
                min(8, pod.n_components_),
                selected_modes,
                pod.n_components_,
            }
        )
        reconstructions = {k: pod.reconstruct_snapshot(snapshot_index, n_modes=k) for k in mode_choices}
        plot_snapshot_reconstructions(
            original=dataset.snapshots[snapshot_index],
            reconstructions=reconstructions,
            grid_shape=dataset.grid_shape,
            output_path=recon_path,
        )
        plot_temporal_coefficients(
            temporal_coeffs=pod.temporal_coeffs_,
            energy_per_mode=pod.energy_per_mode_,
            output_path=temporal_path,
            max_modes=4,
        )
        return {
            "energy_spectrum": energy_path,
            "spatial_modes": modes_path,
            "reconstructions": recon_path,
            "temporal_coefficients": temporal_path,
        }


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Functional API for one-line execution."""
    return PODAnalysisWorkflow(config).run()

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .data import create_synthetic_dataset
from .demo_data import create_multitone_dataset, create_oscillatory_dataset
from .dmd import BOPDMD, DMD
from .mode_selection import AutoModeSelector, ModeSelectionResult
from .spod import SPOD


@dataclass(frozen=True)
class ModeSelectionDemoConfig:
    n_snapshots: int = 220
    n_spatial: int = 600
    true_rank: int = 8
    noise_level: float = 0.04
    methods: Tuple[str, ...] = (
        "energy_95",
        "energy_99",
        "elbow",
        "optimal_threshold",
        "parallel_analysis",
    )
    random_state: int = 42


@dataclass(frozen=True)
class ModeSelectionOutcome:
    method: str
    selected_modes: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ModeSelectionDemoResult:
    config: ModeSelectionDemoConfig
    consensus: Dict[str, object]
    outcomes: Tuple[ModeSelectionOutcome, ...]


@dataclass(frozen=True)
class DMDDemoConfig:
    n_snapshots: int = 320
    n_spatial: int = 96
    dt: float = 0.01
    frequencies: Tuple[float, ...] = (2.0, 5.0)
    amplitudes: Tuple[float, ...] = (1.0, 0.6)
    noise_level: float = 0.01
    rank: int = 10
    n_top_modes: int = 6
    run_bopdmd: bool = True
    n_bags: int = 20
    bag_fraction: float = 0.75
    robust_min_occurrence: float = 0.25
    random_state: int = 7


@dataclass(frozen=True)
class DMDModeSummary:
    index: int
    frequency: float
    growth_rate: float
    amplitude: float


@dataclass(frozen=True)
class DMDDemoResult:
    config: DMDDemoConfig
    true_frequencies: Tuple[float, ...]
    top_modes: Tuple[DMDModeSummary, ...]
    stability: Dict[str, float | int]
    reconstruction_rmse: float
    robust_mode_count: Optional[int] = None


@dataclass(frozen=True)
class SPODDemoConfig:
    n_snapshots: int = 1536
    n_spatial: int = 80
    dt: float = 0.001
    frequencies: Tuple[float, ...] = (20.0, 50.0, 100.0)
    amplitudes: Tuple[float, ...] = (1.0, 0.7, 0.4)
    noise_level: float = 0.15
    n_fft: int = 256
    overlap: float = 0.5
    n_modes: int = 5
    n_peaks: int = 5
    min_separation: float = 8.0
    random_state: int = 11


@dataclass(frozen=True)
class SPODPeakSummary:
    frequency: float
    total_energy: float
    leading_mode_fraction: float


@dataclass(frozen=True)
class SPODDemoResult:
    config: SPODDemoConfig
    true_frequencies: Tuple[float, ...]
    peaks: Tuple[SPODPeakSummary, ...]
    frequencies: NDArray[np.float64]
    total_energy: NDArray[np.float64]


class AdvancedModalDemo:
    """Reusable orchestrator for advanced modal-analysis notebook demos."""

    def run_mode_selection(
        self,
        config: ModeSelectionDemoConfig = ModeSelectionDemoConfig(),
    ) -> ModeSelectionDemoResult:
        dataset = create_synthetic_dataset(
            n_snapshots=config.n_snapshots,
            n_spatial=config.n_spatial,
            true_rank=config.true_rank,
            noise_level=config.noise_level,
            random_state=config.random_state,
        )

        from .core import POD

        pod = POD(center=True).fit(dataset.snapshots)
        selector = AutoModeSelector(methods=list(config.methods))
        raw = selector.select(dataset.snapshots, pod=pod, random_state=config.random_state)

        outcomes: List[ModeSelectionOutcome] = []
        for method in config.methods:
            value = raw.get(method)
            if isinstance(value, ModeSelectionResult):
                outcomes.append(ModeSelectionOutcome(method=method, selected_modes=int(value.selected_modes)))
            elif isinstance(value, dict) and "error" in value:
                outcomes.append(ModeSelectionOutcome(method=method, error=str(value["error"])))
            else:
                outcomes.append(ModeSelectionOutcome(method=method, error="No result returned."))

        consensus = raw.get("consensus", {})
        return ModeSelectionDemoResult(config=config, consensus=consensus, outcomes=tuple(outcomes))

    def run_dmd(self, config: DMDDemoConfig = DMDDemoConfig()) -> DMDDemoResult:
        oscillatory = create_oscillatory_dataset(
            n_snapshots=config.n_snapshots,
            n_spatial=config.n_spatial,
            dt=config.dt,
            frequencies=config.frequencies,
            amplitudes=config.amplitudes,
            noise_level=config.noise_level,
            random_state=config.random_state,
        )

        dmd = DMD(rank=config.rank, dt=config.dt, exact=True).fit(oscillatory.snapshots)
        modes = dmd.get_modes(n_modes=config.n_top_modes)
        mode_summaries = tuple(
            DMDModeSummary(
                index=int(mode.index),
                frequency=float(mode.frequency),
                growth_rate=float(mode.growth_rate),
                amplitude=float(mode.amplitude),
            )
            for mode in modes
        )
        raw_stability = dmd.stability_analysis()
        stability: Dict[str, float | int] = {
            "n_stable_modes": int(raw_stability["n_stable_modes"]),
            "n_unstable_modes": int(raw_stability["n_unstable_modes"]),
            "max_eigenvalue_magnitude": float(raw_stability["max_eigenvalue_magnitude"]),
            "spectral_radius": float(raw_stability["spectral_radius"]),
            "dominant_frequency": float(raw_stability["dominant_frequency"]),
            "dominant_growth_rate": float(raw_stability["dominant_growth_rate"]),
        }
        rmse = float(dmd.reconstruction_error(oscillatory.snapshots, metric="rmse"))

        robust_mode_count: Optional[int] = None
        if config.run_bopdmd:
            bop = BOPDMD(
                rank=config.rank,
                dt=config.dt,
                n_bags=config.n_bags,
                bag_fraction=config.bag_fraction,
                random_state=config.random_state,
            ).fit(oscillatory.snapshots)
            robust = bop.get_robust_modes(min_occurrence=config.robust_min_occurrence)
            robust_mode_count = len(robust)

        return DMDDemoResult(
            config=config,
            true_frequencies=tuple(float(f) for f in oscillatory.frequencies),
            top_modes=mode_summaries,
            stability=stability,
            reconstruction_rmse=rmse,
            robust_mode_count=robust_mode_count,
        )

    def run_spod(self, config: SPODDemoConfig = SPODDemoConfig()) -> SPODDemoResult:
        multitone = create_multitone_dataset(
            n_snapshots=config.n_snapshots,
            n_spatial=config.n_spatial,
            dt=config.dt,
            frequencies=config.frequencies,
            amplitudes=config.amplitudes,
            noise_level=config.noise_level,
            random_state=config.random_state,
        )

        spod = SPOD(
            n_fft=config.n_fft,
            overlap=config.overlap,
            dt=config.dt,
            n_modes=config.n_modes,
        ).fit(multitone.snapshots)
        peak_dicts = spod.find_dominant_frequencies(
            n_peaks=config.n_peaks,
            min_separation=config.min_separation,
        )
        peaks = tuple(
            SPODPeakSummary(
                frequency=float(item["frequency"]),
                total_energy=float(item["total_energy"]),
                leading_mode_fraction=float(item["leading_mode_fraction"]),
            )
            for item in peak_dicts
        )
        freqs, total_energy = spod.total_energy_spectrum()
        return SPODDemoResult(
            config=config,
            true_frequencies=tuple(float(f) for f in multitone.frequencies),
            peaks=peaks,
            frequencies=freqs,
            total_energy=total_energy,
        )


def format_mode_selection_result(result: ModeSelectionDemoResult) -> List[str]:
    lines: List[str] = ["Consensus:", str(result.consensus), "Per-method selected modes:"]
    for item in result.outcomes:
        if item.error is not None:
            lines.append(f"  {item.method}: ERROR -> {item.error}")
        else:
            lines.append(f"  {item.method}: {item.selected_modes}")
    return lines


def format_dmd_result(result: DMDDemoResult) -> List[str]:
    lines: List[str] = [
        f"True frequencies (Hz): {result.true_frequencies}",
        "Top DMD modes by amplitude:",
    ]
    for mode in result.top_modes:
        lines.append(
            "  Mode "
            f"{mode.index}: f={mode.frequency:.3f} Hz, "
            f"growth={mode.growth_rate:.4f}, amp={mode.amplitude:.3f}"
        )
    lines.append("Stability summary:")
    lines.append(str(result.stability))
    lines.append(f"DMD reconstruction RMSE: {result.reconstruction_rmse:.6f}")
    if result.robust_mode_count is not None:
        lines.append(f"BOP-DMD robust modes found: {result.robust_mode_count}")
    return lines


def format_spod_result(result: SPODDemoResult) -> List[str]:
    lines: List[str] = [
        f"True frequencies (Hz): {result.true_frequencies}",
        "Top SPOD peaks:",
    ]
    for peak in result.peaks:
        lines.append(
            "  "
            f"f={peak.frequency:.2f} Hz, total_energy={peak.total_energy:.4f}, "
            f"lead_mode={peak.leading_mode_fraction * 100:.1f}%"
        )
    return lines


def plot_spod_spectrum(
    result: SPODDemoResult,
    max_frequency: float = 150.0,
    figsize: Tuple[float, float] = (8.0, 3.5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.frequencies, result.total_energy, lw=1.5)
    ax.set_xlim(0, max_frequency)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Total SPOD Energy")
    ax.set_title("SPOD Total Energy Spectrum")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

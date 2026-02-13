from __future__ import annotations

from typing import Dict, Iterable, List, Mapping


def engineering_metrics(summary: Mapping[str, object], top_modes: int = 6) -> Dict[str, object]:
    """Extract compact engineering metrics from workflow summary."""
    pod = summary["pod"]
    energies = pod["top_energies_percent"][:top_modes]
    return {
        "selected_modes": pod["selected_modes"],
        "selected_energy_percent": round(float(pod["selected_energy_percent"]), 3),
        "reconstruction_rmse": round(float(pod["reconstruction_rmse"]), 6),
        "reconstruction_relative_l2": round(float(pod["reconstruction_relative_l2"]), 6),
        "mode_counts_for_energy": pod["mode_counts_for_energy"],
        "compression_ratio_selected": round(float(pod["compression_ratio_selected"]), 3),
        "top_energies_percent": [round(float(x), 3) for x in energies],
    }


def spectral_peak_lines(peaks: Iterable[Mapping[str, object]]) -> List[str]:
    """Format spectral peak records into readable one-line summaries."""
    lines: List[str] = []
    for peak in peaks:
        lines.append(
            f"Mode {int(peak['mode'])}: "
            f"f={float(peak['frequency']):.6f}, amp={float(peak['amplitude']):.2f}"
        )
    return lines

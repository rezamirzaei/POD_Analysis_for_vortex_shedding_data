from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from .workflow import AnalysisConfig, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pod-analyze",
        description="Run POD analysis on vortex street snapshot data.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("vortex_data.mat"),
        help="Path to .mat data file containing VORTALL matrix.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download canonical VORTALL dataset when --data does not exist.",
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs=2,
        metavar=("NY", "NX"),
        help="Grid shape for reshaping spatial fields.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=None,
        help="Number of POD modes for reduced-order reconstruction.",
    )
    parser.add_argument(
        "--energy-target",
        type=float,
        default=0.95,
        help="Energy capture target used when --modes is not provided.",
    )
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=None,
        help="Snapshot index to use for reconstruction plots. Default: middle snapshot.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Temporal spacing between snapshots (arbitrary units).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for summary JSON and generated figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation and only write summary JSON.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    grid_shape: Optional[Tuple[int, int]] = tuple(args.grid_shape) if args.grid_shape else None
    config = AnalysisConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        modes=args.modes,
        energy_target=args.energy_target,
        grid_shape=grid_shape,
        snapshot_index=args.snapshot_index,
        dt=args.dt,
        generate_plots=not args.no_plots,
        auto_download=args.download_if_missing,
    )

    try:
        result = run_analysis(config)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    pod = result.pod
    selected_modes = result.selected_modes
    rmse = result.summary["pod"]["reconstruction_rmse"]
    rel_l2 = result.summary["pod"]["reconstruction_relative_l2"]

    print("POD analysis complete.")
    print(
        f"Snapshots: {result.dataset.n_snapshots}, "
        f"Spatial points: {result.dataset.n_spatial_points}"
    )
    print(
        f"Selected modes: {selected_modes} "
        f"({pod.cumulative_energy_[selected_modes - 1] * 100:.2f}% cumulative energy)"
    )
    print(f"Reconstruction RMSE: {rmse:.6f}, Relative L2: {rel_l2:.6f}")
    print("Generated outputs:")
    for label, path in result.generated_files.items():
        print(f"  - {label}: {path}")
    return 0

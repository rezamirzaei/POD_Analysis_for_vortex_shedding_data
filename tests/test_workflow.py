from pathlib import Path

import numpy as np
from scipy.io import savemat

from pod_analysis.workflow import AnalysisConfig, run_analysis


def test_run_analysis_with_fixed_modes_writes_summary(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    matrix = rng.standard_normal((6, 5))
    data_path = tmp_path / "toy.mat"
    savemat(data_path, {"VORTALL": matrix})

    config = AnalysisConfig(
        data_path=str(data_path),
        output_dir=str(tmp_path / "out"),
        modes=2,
        grid_shape=(2, 3),
        generate_plots=False,
    )
    result = run_analysis(config)

    assert result.selected_modes == 2
    assert result.summary_path.exists()
    assert set(result.generated_files) == {"summary"}
    assert result.dataset.n_spatial_points == 6
    assert result.dataset.n_snapshots == 5


def test_run_analysis_with_energy_target_selects_modes(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    left = rng.standard_normal((15, 2))
    right = rng.standard_normal((2, 9))
    matrix = (left @ right).T
    data_path = tmp_path / "low_rank.mat"
    savemat(data_path, {"VORTALL": matrix})

    config = AnalysisConfig(
        data_path=data_path,
        output_dir=tmp_path / "out_energy",
        energy_target=0.95,
        grid_shape=(3, 3),
        generate_plots=False,
    )
    result = run_analysis(config)

    assert result.selected_modes <= 2
    assert result.summary["pod"]["selected_energy_percent"] >= 95

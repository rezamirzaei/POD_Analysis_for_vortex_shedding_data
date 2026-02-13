from pod_analysis import (
    AdvancedModalDemo,
    format_dmd_result,
    format_mode_selection_result,
    format_spod_result,
    plot_spod_spectrum,
)


def test_advanced_demo_mode_selection_runs() -> None:
    demo = AdvancedModalDemo()
    result = demo.run_mode_selection()
    lines = format_mode_selection_result(result)

    assert result.consensus
    assert len(result.outcomes) > 0
    assert any("Per-method selected modes" in line for line in lines)


def test_advanced_demo_dmd_runs() -> None:
    demo = AdvancedModalDemo()
    result = demo.run_dmd()
    lines = format_dmd_result(result)

    assert result.reconstruction_rmse >= 0
    assert len(result.top_modes) > 0
    assert any("Top DMD modes" in line for line in lines)


def test_advanced_demo_spod_runs_and_plots() -> None:
    demo = AdvancedModalDemo()
    result = demo.run_spod()
    lines = format_spod_result(result)
    fig = plot_spod_spectrum(result)

    assert len(result.peaks) > 0
    assert result.frequencies.shape == result.total_energy.shape
    assert any("Top SPOD peaks" in line for line in lines)
    fig.clf()

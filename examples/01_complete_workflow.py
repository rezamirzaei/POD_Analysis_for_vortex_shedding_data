"""Example: Complete POD analysis workflow.

This example demonstrates a complete POD analysis workflow including:
- Data loading
- POD computation
- Mode selection using multiple methods
- Validation
- Visualization
"""

import numpy as np
from pathlib import Path

# Import from pod_analysis
from pod_analysis import (
    POD,
    AnalysisConfig,
    run_analysis,
    validate_pod_model,
    AutoModeSelector,
    create_synthetic_dataset,
    save_pod_results,
)


def main():
    print("=" * 60)
    print("POD Analysis Toolkit - Complete Workflow Example")
    print("=" * 60)

    # Create synthetic dataset for demonstration
    print("\n1. Creating synthetic dataset...")
    dataset = create_synthetic_dataset(
        n_snapshots=200,
        n_spatial=1000,
        true_rank=8,
        noise_level=0.05,
        random_state=42,
    )
    print(f"   Snapshots: {dataset.n_snapshots}")
    print(f"   Spatial points: {dataset.n_spatial_points}")

    # Fit POD model
    print("\n2. Fitting POD model...")
    pod = POD(svd_solver="auto", center=True)
    pod.fit(dataset.snapshots)
    print(f"   Total modes: {pod.n_components_}")

    # Energy analysis
    print("\n3. Energy analysis:")
    for target in [0.90, 0.95, 0.99]:
        n_modes = pod.modes_for_energy(target)
        print(f"   Modes for {target*100:.0f}% energy: {n_modes}")

    # Advanced mode selection
    print("\n4. Advanced mode selection:")
    selector = AutoModeSelector(
        methods=["energy_95", "energy_99", "elbow", "aic", "bic"]
    )
    selection_results = selector.select(dataset.snapshots, pod=pod)

    consensus = selection_results["consensus"]
    print(f"   Recommended modes: {consensus['recommended']}")
    print(f"   Standard deviation: {consensus['std']:.2f}")
    print(f"   Range: [{consensus['min']}, {consensus['max']}]")

    # Use recommended mode count
    n_modes = consensus["recommended"]

    # Reconstruction quality
    print(f"\n5. Reconstruction quality (using {n_modes} modes):")
    rmse = pod.reconstruction_error(dataset.snapshots, n_modes=n_modes, metric="rmse")
    rel_l2 = pod.reconstruction_error(dataset.snapshots, n_modes=n_modes, metric="relative_l2")
    r2 = pod.score(dataset.snapshots, n_modes=n_modes)
    print(f"   RMSE: {rmse:.6f}")
    print(f"   Relative L2: {rel_l2:.6f}")
    print(f"   R² score: {r2:.6f}")

    # Validation
    print("\n6. POD validation:")
    validation = validate_pod_model(
        pod=pod,
        X=dataset.snapshots,
        selected_modes=n_modes,
        atol=1e-8,
    )
    print(f"   Validation passed: {validation.passed}")
    print(f"   Orthonormality error: {validation.consistency.orthonormality_fro_error:.2e}")
    print(f"   Projection error: {validation.consistency.projection_relative_error:.2e}")

    # Transform new data
    print("\n7. Transforming new data:")
    new_data = dataset.snapshots[:10]  # Use first 10 snapshots as "new"
    reduced = pod.transform(new_data, n_modes=n_modes)
    reconstructed = pod.inverse_transform(reduced, n_modes=n_modes)
    print(f"   Original shape: {new_data.shape}")
    print(f"   Reduced shape: {reduced.shape}")
    print(f"   Reconstructed shape: {reconstructed.shape}")

    # Save results
    print("\n8. Saving results:")
    output_dir = Path("outputs/example")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pod_results(
        output_dir / "pod_results.npz",
        pod,
        format="numpy",
        n_modes=n_modes,
    )
    print(f"   Saved to: {output_dir / 'pod_results.npz'}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""Example: Dynamic Mode Decomposition (DMD) analysis.

This example demonstrates DMD capabilities including:
- Basic DMD fitting and reconstruction
- Mode extraction with frequencies and growth rates
- Future state prediction
- Stability analysis
- Comparison with POD
"""

import numpy as np
import matplotlib.pyplot as plt

from pod_analysis import POD, DMD, BOPDMD


def create_oscillatory_data(n_snapshots=100, n_spatial=50, dt=0.01):
    """Create data with known oscillatory dynamics."""
    t = np.arange(n_snapshots) * dt
    x = np.linspace(0, 2*np.pi, n_spatial)

    # Two oscillatory modes with different frequencies
    f1, f2 = 5.0, 12.0  # Hz

    mode1 = np.sin(x)[:, np.newaxis]
    mode2 = np.sin(2*x)[:, np.newaxis]

    coeff1 = np.sin(2*np.pi*f1*t)
    coeff2 = 0.5 * np.sin(2*np.pi*f2*t)

    data = (mode1 @ coeff1[np.newaxis, :] + mode2 @ coeff2[np.newaxis, :]).T

    # Add small noise
    rng = np.random.default_rng(42)
    data += 0.05 * rng.standard_normal(data.shape)

    return data, t, (f1, f2)


def main():
    print("=" * 60)
    print("Dynamic Mode Decomposition (DMD) Example")
    print("=" * 60)

    # Create data
    print("\n1. Creating oscillatory data...")
    dt = 0.01
    snapshots, time, true_freqs = create_oscillatory_data(
        n_snapshots=200, n_spatial=100, dt=dt
    )
    print(f"   Snapshots: {snapshots.shape[0]}")
    print(f"   Spatial points: {snapshots.shape[1]}")
    print(f"   True frequencies: {true_freqs[0]:.1f} Hz, {true_freqs[1]:.1f} Hz")

    # Fit DMD
    print("\n2. Fitting DMD...")
    dmd = DMD(rank=10, dt=dt)
    dmd.fit(snapshots)
    print(f"   Number of DMD modes: {len(dmd.eigenvalues_)}")

    # Extract modes
    print("\n3. DMD mode analysis:")
    modes = dmd.get_modes(n_modes=5)
    print("   Top 5 modes by amplitude:")
    for mode in modes:
        stability = "stable" if mode.is_stable else "unstable"
        print(f"   Mode {mode.index}: f={mode.frequency:.2f} Hz, "
              f"growth={mode.growth_rate:.4f}, amp={mode.amplitude:.3f} ({stability})")

    # Stability analysis
    print("\n4. Stability analysis:")
    stability = dmd.stability_analysis()
    print(f"   Stable modes: {stability['n_stable_modes']}")
    print(f"   Unstable modes: {stability['n_unstable_modes']}")
    print(f"   Spectral radius: {stability['spectral_radius']:.4f}")
    print(f"   Dominant frequency: {stability['dominant_frequency']:.2f} Hz")

    # Reconstruction
    print("\n5. Reconstruction quality:")
    rmse = dmd.reconstruction_error(snapshots, metric="rmse")
    rel_l2 = dmd.reconstruction_error(snapshots, metric="relative_l2")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   Relative L2: {rel_l2:.6f}")

    # Prediction
    print("\n6. Future prediction:")
    n_predict = 50
    future = dmd.predict(n_steps=n_predict, start_step=snapshots.shape[0])
    print(f"   Predicted {n_predict} future time steps")
    print(f"   Prediction shape: {future.shape}")

    # Compare with POD
    print("\n7. Comparison with POD:")
    pod = POD().fit(snapshots)
    pod_rmse = pod.reconstruction_error(snapshots, n_modes=10, metric="rmse")
    print(f"   POD RMSE (10 modes): {pod_rmse:.6f}")
    print(f"   DMD RMSE (10 modes): {rmse:.6f}")
    print(f"   Note: DMD captures dynamics, POD captures energy")

    # BOP-DMD for robust analysis
    print("\n8. Robust DMD (BOP-DMD):")
    bopdmd = BOPDMD(rank=10, n_bags=50, dt=dt, random_state=42)
    bopdmd.fit(snapshots)

    robust_modes = bopdmd.get_robust_modes(min_occurrence=0.3)
    print(f"   Found {len(robust_modes)} robust modes:")
    for mode in robust_modes[:3]:
        print(f"   f={mode['frequency']:.2f} Hz ± {mode['frequency_std']:.2f}, "
              f"occurrence={mode['occurrence_rate']*100:.0f}%")

    print("\n" + "=" * 60)
    print("DMD Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


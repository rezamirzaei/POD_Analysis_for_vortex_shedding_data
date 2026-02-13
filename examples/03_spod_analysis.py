"""Example: Spectral POD (SPOD) analysis.

This example demonstrates SPOD capabilities for frequency-resolved
coherent structure extraction:
- Basic SPOD computation
- Frequency-resolved mode analysis
- Finding dominant frequencies
- Comparing with standard POD
"""

import numpy as np

from pod_analysis import POD, SPOD


def create_multitone_signal(n_snapshots=1000, n_spatial=50, dt=0.001):
    """Create signal with multiple frequency components."""
    t = np.arange(n_snapshots) * dt
    x = np.linspace(0, 2*np.pi, n_spatial)

    # Multiple frequency components
    freqs = [20.0, 50.0, 100.0]  # Hz
    amplitudes = [1.0, 0.7, 0.4]

    data = np.zeros((n_snapshots, n_spatial))
    for f, amp in zip(freqs, amplitudes):
        spatial_pattern = np.sin(x * (freqs.index(f) + 1))
        temporal_pattern = amp * np.sin(2*np.pi*f*t)
        data += temporal_pattern[:, np.newaxis] * spatial_pattern

    # Add broadband noise
    rng = np.random.default_rng(42)
    data += 0.2 * rng.standard_normal(data.shape)

    return data, t, freqs


def main():
    print("=" * 60)
    print("Spectral POD (SPOD) Example")
    print("=" * 60)

    # Create data
    print("\n1. Creating multi-frequency data...")
    dt = 0.001  # 1 kHz sampling
    snapshots, time, true_freqs = create_multitone_signal(
        n_snapshots=2000, n_spatial=100, dt=dt
    )
    print(f"   Snapshots: {snapshots.shape[0]}")
    print(f"   Spatial points: {snapshots.shape[1]}")
    print(f"   Duration: {time[-1]:.2f} s")
    print(f"   True frequencies: {true_freqs} Hz")

    # Fit SPOD
    print("\n2. Fitting SPOD...")
    spod = SPOD(n_fft=256, overlap=0.5, dt=dt, n_modes=5)
    spod.fit(snapshots)
    print(f"   Number of frequency bins: {len(spod.frequencies_)}")
    print(f"   Number of blocks: {spod.n_blocks_}")
    print(f"   Frequency resolution: {spod.frequencies_[1]:.2f} Hz")

    # Find dominant frequencies
    print("\n3. Dominant frequencies:")
    peaks = spod.find_dominant_frequencies(n_peaks=5, min_separation=10.0)
    for i, peak in enumerate(peaks):
        print(f"   Peak {i+1}: f={peak['frequency']:.1f} Hz, "
              f"energy={peak['total_energy']:.4f}, "
              f"leading mode fraction={peak['leading_mode_fraction']*100:.0f}%")

    # Analyze modes at specific frequencies
    print("\n4. Mode analysis at key frequencies:")
    for freq in [20.0, 50.0, 100.0]:
        modes = spod.get_modes_at_frequency(freq, n_modes=3)
        print(f"\n   At {freq:.0f} Hz:")
        for mode in modes:
            print(f"     Mode {mode.mode_index}: energy={mode.energy:.4f}, "
                  f"fraction={mode.energy_fraction*100:.1f}%")

    # Cumulative energy at dominant frequency
    print("\n5. Cumulative energy at 50 Hz:")
    cum_energy = spod.cumulative_energy_at_frequency(50.0)
    for i, e in enumerate(cum_energy):
        print(f"   First {i+1} mode(s): {e*100:.1f}%")

    # Energy spectrum of leading mode
    print("\n6. Energy spectrum (leading mode):")
    freqs, energies = spod.get_energy_spectrum(mode_index=0)
    peak_idx = np.argmax(energies)
    print(f"   Peak frequency: {freqs[peak_idx]:.1f} Hz")
    print(f"   Peak energy: {energies[peak_idx]:.4f}")

    # Compare with standard POD
    print("\n7. Comparison with standard POD:")
    pod = POD().fit(snapshots)

    print(f"\n   Standard POD energies (first 5 modes):")
    for i in range(5):
        print(f"     Mode {i+1}: {pod.energy_per_mode_[i]*100:.2f}%")

    print(f"\n   SPOD at 50 Hz energies (first 5 modes):")
    for mode in spod.get_modes_at_frequency(50.0, n_modes=5):
        print(f"     Mode {mode.mode_index}: {mode.energy_fraction*100:.2f}%")

    print("\n   Key difference:")
    print("   - POD ranks by total energy (all frequencies mixed)")
    print("   - SPOD ranks by energy at each frequency separately")
    print("   - SPOD reveals frequency-coherent structures")

    # Total energy spectrum
    print("\n8. Total SPOD energy spectrum:")
    freqs, total_energy = spod.total_energy_spectrum()
    sorted_idx = np.argsort(total_energy)[::-1]
    print("   Top 5 frequencies by total energy:")
    for i in range(5):
        idx = sorted_idx[i]
        print(f"     {freqs[idx]:.1f} Hz: {total_energy[idx]:.4f}")

    print("\n" + "=" * 60)
    print("SPOD Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


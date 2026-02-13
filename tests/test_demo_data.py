import numpy as np

from pod_analysis import create_multitone_dataset, create_oscillatory_dataset


def test_create_oscillatory_dataset_shapes() -> None:
    dataset = create_oscillatory_dataset(
        n_snapshots=120,
        n_spatial=64,
        dt=0.02,
        frequencies=(3.0, 7.0),
        amplitudes=(1.0, 0.5),
        noise_level=0.0,
        random_state=1,
    )

    assert dataset.snapshots.shape == (120, 64)
    assert dataset.time.shape == (120,)
    assert dataset.frequencies == (3.0, 7.0)
    assert dataset.amplitudes == (1.0, 0.5)


def test_multitone_dataset_can_convert_to_snapshot_dataset() -> None:
    dataset = create_multitone_dataset(n_snapshots=256, n_spatial=32, random_state=2)
    snapshot_dataset = dataset.to_snapshot_dataset()

    assert snapshot_dataset.snapshots.shape == (256, 32)
    assert snapshot_dataset.time_values is not None
    assert snapshot_dataset.metadata is not None
    assert snapshot_dataset.metadata["synthetic"] is True

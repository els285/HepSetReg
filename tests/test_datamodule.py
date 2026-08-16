import numpy as np
import torch

from hepsetreg.data.datamodule import RegressionDataModule
from hepsetreg.data.dataset import PaddedObjectDataset
from hepsetreg.data.preprocessing import pad_and_mask, write_padded_hdf5


def _write_fixture(path, n_events=20, seed=0):
    rng = np.random.default_rng(seed)
    events = [rng.normal(size=(int(rng.integers(1, 5)), 4)).astype(np.float32) for _ in range(n_events)]
    jets_features, jets_mask = pad_and_mask(events, max_count=6, feature_dim=4)
    leptons = rng.normal(size=(n_events, 1, 3)).astype(np.float32)
    leptons_mask = np.ones((n_events, 1), dtype=bool)
    target = rng.normal(size=(n_events, 2)).astype(np.float32)
    truth = rng.normal(size=(n_events, 1)).astype(np.float32)

    write_padded_hdf5(
        path,
        {"jets": (jets_features, jets_mask), "leptons": (leptons, leptons_mask)},
        target,
        extras={"truth_mass": truth},
    )


def test_padded_object_dataset_item_shapes(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path, n_events=20)

    dataset = PaddedObjectDataset(path, group_names=["jets", "leptons"], extra_keys=["truth_mass"])
    assert len(dataset) == 20

    item = dataset[0]
    assert item["objects"]["jets"].shape == (6, 4)
    assert item["mask"]["jets"].shape == (6,)
    assert item["mask"]["jets"].dtype == torch.bool
    assert item["objects"]["leptons"].shape == (1, 3)
    assert item["target"].shape == (2,)
    assert item["truth_mass"].shape == (1,)


def test_datamodule_produces_correctly_shaped_batches(tmp_path):
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    _write_fixture(train_path, n_events=16, seed=1)
    _write_fixture(val_path, n_events=8, seed=2)

    dm = RegressionDataModule(
        group_names=["jets", "leptons"],
        train_file=train_path,
        val_file=val_path,
        extra_keys=["truth_mass"],
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    assert batch["objects"]["jets"].shape == (4, 6, 4)
    assert batch["mask"]["jets"].shape == (4, 6)
    assert batch["objects"]["leptons"].shape == (4, 1, 3)
    assert batch["target"].shape == (4, 2)
    assert batch["truth_mass"].shape == (4, 1)

    val_batch = next(iter(dm.val_dataloader()))
    assert val_batch["target"].shape[0] <= 4

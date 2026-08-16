"""Lightning DataModule wrapping :class:`~hepsetreg.data.dataset.PaddedObjectDataset`."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import lightning as L
from torch.utils.data import DataLoader

from hepsetreg.data.dataset import PaddedObjectDataset


class RegressionDataModule(L.LightningDataModule):
    def __init__(
        self,
        group_names: List[str],
        train_file: Union[str, Path],
        val_file: Union[str, Path],
        test_file: Optional[Union[str, Path]] = None,
        target_key: str = "target",
        extra_keys: Optional[List[str]] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
    ):
        super().__init__()
        self.group_names = group_names
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.target_key = target_key
        self.extra_keys = extra_keys
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if persistent_workers is not None else (num_workers > 0)

        self.train_dataset: Optional[PaddedObjectDataset] = None
        self.val_dataset: Optional[PaddedObjectDataset] = None
        self.test_dataset: Optional[PaddedObjectDataset] = None

    def _build(self, path):
        return PaddedObjectDataset(
            path, group_names=self.group_names, target_key=self.target_key, extra_keys=self.extra_keys
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._build(self.train_file)
            self.val_dataset = self._build(self.val_file)
        if stage in (None, "test") and self.test_file is not None:
            self.test_dataset = self._build(self.test_file)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return self._loader(self.test_dataset, shuffle=False)

import argparse
import pathlib
from pathlib import Path
import joblib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .data_loading import MRIDataset


class MRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: argparse.ArgumentParser,
    ):
        super().__init__()
        self.config = config
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = MRIDataset(data_path=pathlib.Path(self.config.train_path), dataset_type='ACDC')
        self.val_dataset = MRIDataset(data_path=pathlib.Path(self.config.val_path), dataset_type='ACDC')

    def train_dataloader(self):
        sampling_dict = dict(shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True,
            **sampling_dict,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.val_shuffle,
        )

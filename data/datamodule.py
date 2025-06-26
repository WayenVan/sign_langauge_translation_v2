from lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import List


class DataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.batch_size_val_test = cfg.data.batch_size_val_test

        self.pipline_train = instantiate(getattr(cfg.data.train, "pipline", None))
        self.pipline_val = instantiate(getattr(cfg.data.val, "pipline", None))
        self.pipline_test = instantiate(
            getattr(cfg.data.test, "pipline", cfg.data.val.pipline),
        )

    def setup(self, stage=None):
        # Set up the dataset for training, validation, and testing
        if stage == "fit" or stage is None:
            self.train_dataset = instantiate(
                self.cfg.data.train.dataset, pipline=self.pipline_train
            )
            self.val_dataset = instantiate(
                self.cfg.data.val.dataset, pipline=self.pipline_val
            )

        if stage == "test" or stage is None:
            self.test_dataset = instantiate(
                self.cfg.data.test.dataset, pipline=self.pipline_test
            )

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(
            self.train_dataset,
            collate_fn=instantiate(self.cfg.data.train.collator),
            **self.cfg.data.train.loader_kwargs,
        )

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(
            self.val_dataset,
            collate_fn=instantiate(self.cfg.data.train.collator),
            **self.cfg.data.val.loader_kwargs,
        )

    def test_dataloader(self):
        kwargs = getattr(
            self.cfg.data, "test_dataloader_kwargs", self.cfg.data.val_dataloader_kwargs
        )
        collator = getattr(self.cfg.data.test, "collator", self.cfg.data.val_collator)
        return DataLoader(
            self.test_dataset,
            collate_fn=instantiate(collator),
            **kwargs,
        )

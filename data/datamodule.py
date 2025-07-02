from lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import List


class DataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
    ):
        super().__init__()
        self.cfg = cfg

        self.pipline_train = instantiate(getattr(cfg.train, "pipline", None))
        self.pipline_val = instantiate(getattr(cfg.val, "pipline", None))
        self.pipline_test = instantiate(
            getattr(cfg.test, "pipline", cfg.val.pipline),
        )
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        # Set up the dataset for training, validation, and testing
        if stage == "fit" or stage is None:
            self.train_dataset = instantiate(
                self.cfg.train.dataset, pipline=self.pipline_train
            )
            self.val_dataset = instantiate(
                self.cfg.val.dataset, pipline=self.pipline_val
            )

        if stage == "test" or stage is None:
            if self.cfg.test is not None:
                self.test_dataset = instantiate(
                    self.cfg.test.dataset, pipline=self.pipline_test
                )

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(
            self.train_dataset,
            collate_fn=instantiate(
                self.cfg.train.collator,
                tokenizer=self.tokenizer,
            ),
            **self.cfg.train.loader_kwargs,
        )

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(
            self.val_dataset,
            collate_fn=instantiate(self.cfg.val.collator, tokenizer=self.tokenizer),
            **self.cfg.val.loader_kwargs,
        )

    def test_dataloader(self):
        kwargs = getattr(
            self.cfg, "test_dataloader_kwargs", self.cfg.val_dataloader_kwargs
        )
        collator = getattr(self.cfg.test, "collator", self.cfg.val_collator)
        return DataLoader(
            self.test_dataset,
            collate_fn=instantiate(collator, tokenizer=self.tokenizer),
            **kwargs,
        )

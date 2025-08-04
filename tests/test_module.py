from pydoc import visiblename
import sys

sys.path.append(".")
from hydra import compose, initialize
from data.datamodule import DataModule
from model.mbart_slt.mbart_slt import MBartSLTModel
import torch
from hydra.utils import instantiate


def test_pe():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("gfslt-vlp_pretrain_mec")

    model = instantiate(cfg.model.backbone)
    print(model.__class__.__name__)


if __name__ == "__main__":
    test_pe()
    # test_slt_model_inspect()
    # test_slt_model_generation()

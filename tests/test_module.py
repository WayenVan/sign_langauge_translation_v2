from pydoc import visiblename
import sys

sys.path.append(".")
from hydra import compose, initialize
from data.datamodule import DataModule
from model.mbart_slt.mbart_slt import MBartSLTModel
import torch
from hydra.utils import instantiate
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def test_pe():
    import polars as pl

    device = "cpu"
    initialize(config_path="../configs")
    cfg = compose("gfslt-vlp_pretrain_mec")

    model = instantiate(cfg.model.type, cfg)

    input = torch.randn(8 + 16, 3, 224, 224).to(device)
    length = torch.tensor([8, 16]).to(device)  # Example length tensor
    model.visual_backbone.to(device)
    model.visual_adapter.to(device)
    model.visual_position_embedding.to(device)
    model.gemma.get_input_embeddings().to(device)
    with torch.no_grad():
        out = model.get_visual_feats(input, length)
    print(out.visual_feats.shape)  # Should print the shape of the output tensor
    print(out.t_length)  # Should print the length tensor


if __name__ == "__main__":
    test_pe()
    # test_slt_model_inspect()
    # test_slt_model_generation()

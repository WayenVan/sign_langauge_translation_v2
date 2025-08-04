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

    model = instantiate(cfg.model.type, cfg)

    input = torch.randn(8 + 16, 3, 224, 224).cuda()  # Example input tensor
    length = torch.tensor([8, 16]).cuda()  # Example length tensor
    model.visual_backbone.cuda()
    model.visual_adapter.cuda()
    model.visual_position_embedding.cuda()
    model.gemma.get_input_embeddings().cuda()
    with torch.no_grad():
        out = model.get_visual_feats(input, length)
    print(out.visual_feats.shape)  # Should print the shape of the output tensor
    print(out.t_length)  # Should print the length tensor


if __name__ == "__main__":
    test_pe()
    # test_slt_model_inspect()
    # test_slt_model_generation()

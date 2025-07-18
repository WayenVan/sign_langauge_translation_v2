from pydoc import visiblename
import sys

sys.path.append(".")
from hydra import compose, initialize
from data.datamodule import DataModule
from model.mbart_slt.mbart_slt import MBartSLTModel
import torch
from hydra.utils import instantiate


def test_slt_model_inspect():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("initial_train")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    model = SLTModel(
        cfg=cfg,
    )
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)


def test_slt_model():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("gfslt-vlp_pretrain_8a100")
    cfg.data.train.loader_kwargs.batch_size = 2
    cfg.data.train.loader_kwargs.num_workers = 1
    cfg.data.val.loader_kwargs.batch_size = 2
    cfg.data.val.loader_kwargs.num_workers = 1

    model = instantiate(cfg.model.type, cfg).to("cuda:4")
    for name, param in model.named_parameters():
        print(name)

    data_module = DataModule(cfg.data, model.tokenizer)
    data_module.setup()

    loader = data_module.train_dataloader()
    # loader = data_module.val_dataloader()
    model.train()
    for i, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.training_step(batch, 0)
            # model.validation_step(batch, 0)
            print("ok")


def test_slt_model_generation():
    initialize(config_path="../configs")
    cfg = compose("initial_train")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    # model = SLTModel.load_from_checkpoint(
    #     "outputs/train/2025-06-08_15-07-26/05hhj2d9-epoch=61-val_generate_bleu=0.2160.ckpt",
    #     strict=False,
    #     cfg=cfg,
    # )
    model = SLTModel(
        cfg=cfg,
    ).to("cuda:4")
    loader = data_module.train_dataloader()
    for i, batch in enumerate(loader):
        if i < 8:
            continue
        ids = batch["id"]
        video = batch["video"].to(model.device)
        video_length = batch["video_length"].to(model.device)
        text = batch["text"]
        text_ids = model.tokenizer(text)
        # print(text_ids)
        print(text)

        # Generate
        generated_ids = model.generate(video, video_length, max_length=20)

        for item in generated_ids.cpu().tolist():
            print(model.tokenizer.decode(item, skip_special_tokens=True))

        break


if __name__ == "__main__":
    test_slt_model()
    # test_slt_model_inspect()
    # test_slt_model_generation()

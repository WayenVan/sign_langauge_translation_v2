import sys

sys.path.append(".")
from trl import AutoModelForSeq2SeqLMWithValueHead
from model.mbart_slt.mbart_slt_ppo import MBartSLTModel
from hydra import initialize, compose
from hydra.utils import instantiate
from data.datamodule import DataModule
import torch

if __name__ == "__main__":
    device = "cuda:2"
    initialize(config_path="../configs")
    cfg = compose("gfslt-vlp_pretrain_8a100")

    model = AutoModelForSeq2SeqLMWithValueHead(MBartSLTModel(cfg))
    model.is_peft_model = False
    model = model.to(device)

    for name, param in model.named_parameters():
        print(name)

    cfg.data.train.loader_kwargs.batch_size = 2
    cfg.data.train.loader_kwargs.num_workers = 1

    data_module = DataModule(cfg.data, model.pretrained_model.tokenizer)
    data_module.setup()
    for i, batch in enumerate(data_module.train_dataloader()):
        video_input, text_src, _ = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(
                input_ids=video_input["video"].to(device),
                attention_mask=video_input["video_length"].to(device),
                labels=text_src["input_ids"].to(device),
            )
            model.generate(
                input_ids=video_input["video"].to(device),
                attention_mask=video_input["video_length"].to(device),
                max_new_tokens=20,
            )
            print("ok")
            # model.training_step(batch, 0)

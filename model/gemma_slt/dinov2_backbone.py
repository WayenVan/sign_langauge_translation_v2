from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)
from torch import nn
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig
import re

from einops import rearrange

from typing import NamedTuple, List

import logging

logger = logging.getLogger(__name__)


class DinoV2Backbone(nn.Module):
    def __init__(self, id, output_layer=-1, enable_lora=False, **lora_kwargs):
        super().__init__()
        self.id = id
        if not enable_lora:
            self.visual_encoder = Dinov2WithRegistersModel.from_pretrained(id)
            self.is_lora = False
        else:
            self._init_lora_model(lora_kwargs)
            self.is_lora = True

        self.output_layer = output_layer

    def _init_lora_model(self, lora_kwargs):
        visual_encoder = Dinov2WithRegistersModel.from_pretrained(self.id)
        # for name, p in visual_encoder.named_parameters():
        #     print(name)
        lora_config = LoraConfig(
            # task_type=TaskType.IMAGE_CLASSIFICATION,
            target_modules=[
                "query",
                "key",
                "value",
            ],
            # lora_alpha=self.lora_alpha,
            # lora_dropout=self.lora_dropout,
            # r=self.lora_rank,
            **lora_kwargs,
        )

        self.visual_encoder = get_peft_model(
            visual_encoder,
            lora_config,
        )
        trainable, all = self.visual_encoder.get_nb_trainable_parameters()

        logger.info(
            f"Created Lora DinoV2 for {self.id} Trainable parameters: {trainable}, All parameters: {all}, Ratio: {trainable / all:.2%}"
        )

    def forward(self, x):
        """
        videoo: [B, C, H, W]
        """
        B, C, H, W = x.shape
        feats = self.visual_encoder(x, output_hidden_states=True).hidden_states[
            self.output_layer
        ]

        return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = DinoV2Backbone(
        "facebook/dinov2-with-registers-base",
        start_lora_layer=6,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    # print(model)
    video = torch.randn(2, 8, 3, 224, 224)  # Example video tensor
    feats = model(video)
    print(feats.shape)

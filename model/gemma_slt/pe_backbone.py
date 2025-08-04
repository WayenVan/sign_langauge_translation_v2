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

from core.vision_encoder import pe


class PerceptionEncoderBackbone(nn.Module):
    def __init__(self, id):
        super().__init__()
        self.pe = pe.VisionTransformer.from_config(id, pretrained=True)

    def forward(self, x):
        """
        videoo: [B, C, H, W]
        """
        B, C, H, W = x.shape
        return self.pe(x)  # [B, L, D]

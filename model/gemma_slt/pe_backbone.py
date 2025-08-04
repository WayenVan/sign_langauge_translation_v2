from torch import nn
from .vision_encoder import pe

import logging

logger = logging.getLogger(__name__)


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

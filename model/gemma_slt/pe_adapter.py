from torch import nn

from collections import OrderedDict
from typing import Callable
import torch
from einops import rearrange


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % num_heads == 0, (
            "embed_dim must be divisible by num_heads"
        )

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor, t_length=None):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        if t_length is not None:
            t_length = t_length * self.probe.shape[1]

        return rearrange(x, "bt p d -> (bt p) d"), t_length


class SimpleMLP(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, scale_factor):
        super().__init__()
        input_size = input_hidden_size * (scale_factor**2)
        output_size = output_hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class TemporalShuffleConnector(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.modality_projection = nn.Linear(
            input_hidden_size * scale_factor, output_hidden_size, bias=False
        )

    def temporal_shuffle(self, x, t_length, scale_factor=2):
        # x [BT, D]
        #
        assert t_length.fmod(scale_factor).eq(0).all(), (
            "temporal length of all frames must be divisible by scale_factor"
        )
        BT, D = x.size()
        x = rearrange(x, "(b s) d -> b  (s d)", s=scale_factor, d=D)
        return x

    def forward(self, video_hidden_states, t_length=None):
        video_hidden_states = self.temporal_shuffle(
            video_hidden_states, t_length, self.scale_factor
        )
        video_hidden_states = self.modality_projection(video_hidden_states)

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return video_hidden_states, t_length


class SpatialShuffleConnector(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.modality_projection = SimpleMLP(
            input_hidden_size, output_hidden_size, scale_factor
        )

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class PEAdapter(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        spatial_scale_factor: int,
        temporal_scale_factor: int,
        pooling_num_heads: int,
        pooling_num_probe: int = 1,
        pooling_mlp_ratio: int = 4,
    ):
        super().__init__()
        self.spatial_shuffle_connector = SpatialShuffleConnector(
            input_hidden_size, output_hidden_size, spatial_scale_factor
        )
        self.pooling = AttentionPooling(
            output_hidden_size,
            pooling_num_heads,
            num_probe=pooling_num_probe,
            mlp_ratio=pooling_mlp_ratio,
            act_layer=nn.SiLU,
        )
        self.temporal_shuffle_connector = TemporalShuffleConnector(
            output_hidden_size, output_hidden_size, temporal_scale_factor
        )

    def forward(self, video_hidden_states, t_length=None):
        """
        video_hidden_states: [B, CLS+L, D]
        t_length: [B]
        """
        cls = video_hidden_states[:, 0, :].unsqueeze(1)  # [B, 1, D]
        video_hidden_states = video_hidden_states[:, 1:, :]  # [B, T

        # spatial shuffle
        video_hidden_states = self.spatial_shuffle_connector(video_hidden_states)

        # attention pooling
        video_hidden_states, t_length = self.pooling(video_hidden_states, t_length)

        # temporal shuffle
        video_hidden_states, t_length = self.temporal_shuffle_connector(
            video_hidden_states, t_length
        )
        return video_hidden_states, t_length


if __name__ == "__main__":
    # Example usage

    input_hidden_size = 768
    output_hidden_size = 512
    scale_factor = 4
    pooling_num_heads = 8
    pooling_num_probe = 1
    pooling_mlp_ratio = 4
    pe_adapter = PEAdapter(
        input_hidden_size,
        output_hidden_size,
        scale_factor,
        pooling_num_heads,
        pooling_num_probe,
        pooling_mlp_ratio,
    ).cuda()
    video_hidden_states = torch.randn(
        128 + 256, 256, input_hidden_size
    ).cuda()  # [BT, L, D]
    t_length = torch.tensor([128, 256]).cuda()  # [B]
    output, t_length = pe_adapter(video_hidden_states, t_length)
    print("Output shape:", output.shape)  # Should be [B, T', D
    print("Temporal length:", t_length)  # Should be [B]

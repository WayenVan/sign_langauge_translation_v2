import torch
from einops import rearrange
from torch import nn


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class SimpleMLP(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, scale_factor):
        super().__init__()
        input_size = input_hidden_size * (scale_factor**2)
        output_size = output_hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


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


class TemporalShuffleConnector(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.modality_projection = nn.Linear(
            input_hidden_size * scale_factor, output_hidden_size, bias=False
        )

    def temporal_shuffle(self, x, t_length, scale_factor=2):
        # x [BT, L, D]
        #
        assert t_length.fmod(scale_factor).eq(0).all(), (
            "temporal length of all frames must be divisible by scale_factor"
        )
        BT, L, D = x.size()
        x = rearrange(x, "(b s) l d -> b l (s d)", s=scale_factor, d=D)
        return x

    def forward(self, video_hidden_states, t_length=None):
        video_hidden_states = self.temporal_shuffle(
            video_hidden_states, t_length, self.scale_factor
        )
        video_hidden_states = self.modality_projection(video_hidden_states)

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return video_hidden_states, t_length


class PatchMergeAdapter(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        spatial_scale_factor: int,
        temporal_scale_factor: int,
    ):
        super().__init__()
        self.spatial_shuffle_connector = SpatialShuffleConnector(
            input_hidden_size, output_hidden_size, spatial_scale_factor
        )
        self.cls_projection = nn.Linear(
            input_hidden_size, output_hidden_size, bias=False
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

        cls = self.cls_projection(cls)  # [B, 1, D]
        # spatial shuffle
        video_hidden_states = self.spatial_shuffle_connector(video_hidden_states)

        # concatenate projected cls token
        video_hidden_states = torch.cat(
            [cls, video_hidden_states], dim=1
        ).contiguous()  # [B, T', D]

        # temporal shuffle
        video_hidden_states, t_length = self.temporal_shuffle_connector(
            video_hidden_states, t_length
        )

        L = video_hidden_states.size(1)
        video_hidden_states = rearrange(video_hidden_states, "bt l d -> (bt l) d")
        return video_hidden_states, t_length * L


if __name__ == "__main__":
    # Example usage
    input_hidden_size = 768
    output_hidden_size = 512
    s_scale_factor = 8
    t_scale_factor = 4
    pooling_num_heads = 8
    pooling_num_probe = 1
    pooling_mlp_ratio = 4
    HW = 1 + 256
    pe_adapter = PatchMergeAdapter(
        input_hidden_size,
        output_hidden_size,
        s_scale_factor,
        t_scale_factor,
    ).cuda()
    video_hidden_states = torch.randn(
        128 + 256, HW, input_hidden_size
    ).cuda()  # [BT, L, D]
    _t_length = torch.tensor([128, 256]).cuda()  # [B]
    output, t_length = pe_adapter(video_hidden_states, _t_length)
    print("Output shape:", output.shape)  # Should be [B, T', D
    print("Temporal length:", t_length)  # Should be [B]
    factor = 1 / t_scale_factor * (1 + ((HW - 1) / (s_scale_factor**2)))
    print(factor * _t_length)  # Should be [B]
    print(factor)

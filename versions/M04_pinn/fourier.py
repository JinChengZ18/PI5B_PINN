import torch
import torch.nn as nn
import math


class FourierFeatureEncoding(nn.Module):
    """
    Fourier Feature Mapping:
    x -> [sin(2πBx), cos(2πBx)]
    """

    def __init__(self, in_dim, num_frequencies, scale=10.0):
        super().__init__()

        self.in_dim = in_dim
        self.num_frequencies = num_frequencies

        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        # x: [N, in_dim]
        x_proj = 2 * math.pi * x @ self.B  # [N, num_frequencies]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

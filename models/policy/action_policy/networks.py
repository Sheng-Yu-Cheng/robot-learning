"""Small observation-conditioned temporal U-Net shared by the action generators."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2 * channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(condition).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x + self.conv2(F.silu(h))


class ConditionalUnet1D(nn.Module):
    """One down/up stage; supports odd horizons, including horizon one.

    Inputs: actions [B, H, A], time [B] in [0, 1], condition [B, C].
    Output: [B, H, A]. This is a compact architecture, not an upstream replica.
    """

    def __init__(self, action_dim: int, condition_dim: int, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim < 16 or hidden_dim % 8:
            raise ValueError("hidden_dim must be >= 16 and divisible by 8")
        frequencies = torch.exp(-math.log(10000) * torch.arange(16) / 15)
        self.register_buffer("frequencies", frequencies)
        self.time_encoder = nn.Sequential(nn.Linear(32, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        cond_dim = condition_dim + hidden_dim
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)
        self.encoder = ConditionalResidualBlock(hidden_dim, cond_dim)
        self.downsample = nn.Conv1d(hidden_dim, 2 * hidden_dim, 3, stride=2, padding=1)
        self.middle = ConditionalResidualBlock(2 * hidden_dim, cond_dim)
        self.merge = nn.Conv1d(3 * hidden_dim, hidden_dim, 1)
        self.decoder = ConditionalResidualBlock(hidden_dim, cond_dim)
        self.output_proj = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(self, actions, time, condition):
        phase = time[:, None] * self.frequencies[None, :] * 1000
        time_features = self.time_encoder(torch.cat((phase.sin(), phase.cos()), dim=-1))
        context = torch.cat((condition, time_features), dim=-1)
        skip = self.encoder(self.input_proj(actions.transpose(1, 2)), context)
        h = self.middle(self.downsample(skip), context)
        h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
        h = self.merge(torch.cat((skip, h), dim=1))
        return self.output_proj(self.decoder(h, context)).transpose(1, 2)

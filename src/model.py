import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SpectralResBlock(nn.Module):
    """
    A residual block
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm([channels, 1, 1])

    def forward(self, x):
        return x + self.block(x)


class SpectralDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class EnergyHead(nn.Module):
    def __init__(self, image_shape: tuple, blocks_per_stage: int = 2):
        super().__init__()
        c, h, w = image_shape

        self.stem = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c, 32, 3, padding=1)), nn.GELU()
        )
        self.stage1 = nn.Sequential(
            *[SpectralResBlock(32) for _ in range(blocks_per_stage)],
            SpectralDownsample(32, 64)
        )
        self.stage2 = nn.Sequential(
            *[SpectralResBlock(64) for _ in range(blocks_per_stage)],
            SpectralDownsample(64, 128)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.utils.spectral_norm(nn.Linear(128, 1))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.gap(x)

        x = torch.flatten(x, 1)
        return self.fc(x)


class EBM(nn.Module):

    def __init__(
        self, image_shape: tuple, n_heads: int = 4, batch_size: int = 32
    ) -> None:
        super(EBM, self).__init__()
        self.in_dim = image_shape
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList([EnergyHead(image_shape) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o

    def to(self, device):
        for energyhead in self.heads:
            energyhead.to(device)


class EnHead2(nn.Module):

    def __init__(self, hid_dim=4, n_hid=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.LayerNorm(hid_dim),
            *[nn.Linear(n_hid, n_hid) for _ in range(n_hid)],
            nn.LayerNorm(n_hid),
            nn.Linear(n_hid, 1)
        )

    def forward(self, x):
        return self.net(x)


class EBM2(nn.Model):

    def __init__(self, n_head, hid_dim, depth):
        super().__init__()
        self.n_head = n_head
        self.heads = nn.ModuleList([EnHead2(hid_dim, depth) for _ in range(n_head)])

    def forward(self, x: torch.Tensor, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o

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
            nn.GELU()
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
            nn.utils.spectral_norm(nn.Conv2d(c, 32, 3, padding=1)),
            nn.GELU()
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

    def __init__(self, image_shape: tuple, n_heads: int = 4, batch_size: int = 32) -> None:
        super(EBM, self).__init__()
        self.in_dim = image_shape
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [EnergyHead(image_shape) for _ in range(n_heads)]
        )

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

class SmallEnergyHead(nn.Module):
    def __init__(self, image_shape: tuple[int]) -> None:
        super().__init__()
        self.image_shape = image_shape
        b, w, _ = image_shape
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(b, 4, 3, stride=1)), # (B, 1, 12, 12)
            nn.Flatten(),
            nn.GELU(),
            nn.utils.spectral_norm((nn.Linear(4 * (w-2) * (w-2), 32))),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(32, 1))
        )

    def forward(self, x):
        return self.net(x)

class SmallEBM(nn.Module):

    def __init__(self, image_shape: tuple, n_heads: int = 4, batch_size: int = 32) -> None:
        super(SmallEBM, self).__init__()
        self.in_dim = image_shape
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [SmallEnergyHead(image_shape) for _ in range(n_heads)]
        )

    def forward(self, x: torch.Tensor, head_idx: int | None = None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o

class EnergyHeadOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 16, 3, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.GELU(),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(64 * 7 * 7, 1))  # 3136 → 1
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class EBM_Old(nn.Module):

    def __init__(self, in_dim: int, mid_dim: int, n_heads: int = 4, batch_size: int = 32) -> None:
        super(EBM_Old, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.mid_dim = mid_dim
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [EnergyHeadOld() for _ in range(n_heads)]
        )


    def forward(self, x: torch.Tensor, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)
            self.head_outputs = h_o.detach() # Used to estimate the TC

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o

class SmallEnergyHead2d(nn.Module):
    
    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        super().__init__()
        input_dim = image_shape[0] * image_shape[1] * image_shape[2]

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(input_dim, 32)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(32, 1)),
        )

    def forward(self, x):
        return self.net(x)


class SmallEBM2d(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 1, 2),
        n_heads: int = 4,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_dim = image_shape
        self.n_heads = n_heads

        self.heads = nn.ModuleList(
            [SmallEnergyHead2d(image_shape) for _ in range(n_heads)]
        )

    def forward(self, x: torch.Tensor, head_idx: int | None = None):
        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)
        else:
            h_o = torch.stack(
                [head(x).squeeze(-1) for head in self.heads],
                dim=1,
            )

        energy = h_o if h_o.dim() == 1 else h_o.sum(dim=1)
        return energy, h_o

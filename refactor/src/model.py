"""Energy models used by the refactored gaussian / face pipelines."""

import math

import torch
import torch.nn as nn


def make_mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Sequential:
    """ LINEAR -> LAYERNORM -> GELU blocks, spectrally normalized.

    Spectral normalization bounds the Lipschitz constant of each head, which
    keeps energies finite without needing a large L2 penalty on the energies
    themselves (a large penalty collapses them to constants).
    """
    layers = []
    in_features = input_dim

    for _ in range(depth):
        layers.extend(
            [
                nn.utils.spectral_norm(nn.Linear(in_features, width)),
                nn.LayerNorm(width),
                nn.GELU(),
            ]
        )
        in_features = width

    layers.append(nn.utils.spectral_norm(nn.Linear(in_features, output_dim)))
    return nn.Sequential(*layers)


class SpectralResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.GELU(),
        )

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
    """Convolutional energy head for images (needs H, W >= 8)."""

    def __init__(self, image_shape: tuple, blocks_per_stage: int = 2):
        super().__init__()
        c, _, _ = image_shape

        self.stem = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c, 32, 3, padding=1)), nn.GELU()
        )
        self.stage1 = nn.Sequential(
            *[SpectralResBlock(32) for _ in range(blocks_per_stage)],
            SpectralDownsample(32, 64),
        )
        self.stage2 = nn.Sequential(
            *[SpectralResBlock(64) for _ in range(blocks_per_stage)],
            SpectralDownsample(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.utils.spectral_norm(nn.Linear(128, 1))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.gap(x)
        return self.fc(torch.flatten(x, 1))


class FactorizedEBM(nn.Module):
    """Common interface: energy [B] and head outputs [B, n_heads]."""

    n_heads: int
    input_shape: tuple

    def _head_outputs(self, x: torch.Tensor, head_idx=None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, head_idx=None):
        h = self._head_outputs(x, head_idx)
        h = h.reshape(x.shape[0], -1)

        expected = 1 if head_idx is not None else self.n_heads
        if h.shape[1] != expected:
            raise ValueError(f"expected {expected} head output(s), got {h.shape[1]}")
        return h.sum(dim=1), h


class EBM(FactorizedEBM):
    """Convolutional heads; used for LFW / Olivetti / FMNIST."""

    def __init__(self, image_shape: tuple, n_heads: int = 4, batch_size: int = 32):
        super().__init__()
        self.input_shape = tuple(image_shape)
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [EnergyHead(image_shape) for _ in range(n_heads)]
        )

    def _head_outputs(self, x: torch.Tensor, head_idx=None):
        if head_idx is not None:
            return self.heads[head_idx](x)
        return torch.cat([head(x) for head in self.heads], dim=1)


class EnHead2(nn.Module):
    def __init__(self, input_dim: int = 2, hid_dim: int = 4, depth: int = 1):
        super().__init__()
        self.net = make_mlp(input_dim=input_dim, output_dim=1, width=hid_dim, depth=depth)

    def forward(self, x):
        return torch.flatten(self.net(x), 1)


class EBM2(FactorizedEBM):
    """MLP heads for 2-D gaussian data (input shape ``(2,)``)."""

    def __init__(self, n_head, hid_dim, depth, input_shape=(2,)):
        super().__init__()
        self.n_heads = n_head
        self.input_shape = tuple(input_shape)
        input_dim = math.prod(self.input_shape)
        self.heads = nn.ModuleList(
            [EnHead2(input_dim, hid_dim, depth) for _ in range(n_head)]
        )

    def _head_outputs(self, x: torch.Tensor, head_idx=None):
        if head_idx is not None:
            return self.heads[head_idx](x)
        return torch.cat([head(x) for head in self.heads], dim=1)


class EBM2_Dense(FactorizedEBM):
    """Single trunk with one output per head (2-D gaussian data)."""

    def __init__(self, n_head, hid_dim, depth, input_shape=(2,)):
        super().__init__()
        self.n_heads = n_head
        self.input_shape = tuple(input_shape)
        input_dim = math.prod(self.input_shape)
        self.trunk = make_mlp(
            input_dim=input_dim, output_dim=n_head, width=hid_dim, depth=depth
        )

    def _head_outputs(self, x: torch.Tensor, head_idx=None):
        head_outputs = self.trunk(x).reshape(x.shape[0], -1)
        if head_idx is not None:
            return head_outputs[:, head_idx : head_idx + 1]
        return head_outputs

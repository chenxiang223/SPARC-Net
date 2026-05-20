from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn
import torch.nn.functional as F


def reduce_spectral_depth(x: torch.Tensor, target_depth: int = 32) -> torch.Tensor:
    if x.shape[2] == target_depth:
        return x
    return F.adaptive_avg_pool3d(x, (target_depth, x.shape[-2], x.shape[-1]))


class ConvBNAct3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding=0, act: str = "relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(out_channels) if act == "prelu" else nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=1, act: str = "relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels) if act == "prelu" else nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Dense3DBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, layers: int, kernel_size, padding, act: str = "prelu") -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(layers):
            self.layers.append(ConvBNAct3d(channels, growth_rate, kernel_size=kernel_size, padding=padding, act=act))
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size, padding) -> None:
        super().__init__()
        self.conv1 = ConvBNAct3d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct3d(1, width, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            ConvBNAct3d(width, width * 2, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            ConvBNAct3d(width * 2, width * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(width * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reduce_spectral_depth(x.unsqueeze(1))
        return self.classifier(self.features(x))


class FDSSCNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, growth: int = 12) -> None:
        super().__init__()
        self.spectral = Dense3DBlock(1, growth, layers=4, kernel_size=(7, 1, 1), padding=(3, 0, 0), act="prelu")
        self.reduce = ConvBNAct3d(self.spectral.out_channels, 48, kernel_size=(1, 1, 1), padding=0, act="prelu")
        self.spatial = Dense3DBlock(48, growth, layers=4, kernel_size=(1, 3, 3), padding=(0, 1, 1), act="prelu")
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.spatial.out_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reduce_spectral_depth(x.unsqueeze(1))
        x = self.spectral(x)
        x = self.reduce(x)
        x = self.spatial(x)
        return self.head(x)


class HybridSNNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv3d = nn.Sequential(
            ConvBNAct3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            ConvBNAct3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            ConvBNAct3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )
        self.conv2d = nn.Sequential(
            ConvBNAct2d(32 * 8, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(reduce_spectral_depth(x.unsqueeze(1)))
        b, c, d, h, w = x.shape
        x = F.adaptive_avg_pool3d(x, (8, h, w))
        x = x.reshape(b, c * 8, h, w)
        return self.classifier(self.conv2d(x))


class SSRNNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.stem = ConvBNAct3d(1, width, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.spectral = nn.Sequential(
            Residual3DBlock(width, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            Residual3DBlock(width, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
        )
        self.transition = ConvBNAct3d(width, width * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spatial = nn.Sequential(
            Residual3DBlock(width * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            Residual3DBlock(width * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(width * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reduce_spectral_depth(x.unsqueeze(1))
        x = self.stem(x)
        x = self.spectral(x)
        x = self.transition(x)
        x = self.spatial(x)
        return self.head(x)


class SpectralFormerNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dim: int = 64, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.group_size = 8
        self.num_groups = (in_channels + self.group_size - 1) // self.group_size
        self.pad = self.num_groups * self.group_size - in_channels
        self.group_embed = nn.Linear(self.group_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_groups + 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(-1, -2))
        if self.pad:
            x = F.pad(x, (0, self.pad))
        x = x.view(x.size(0), self.num_groups, self.group_size)
        tokens = self.group_embed(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))


class SSFTTNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dim: int = 64, num_tokens: int = 8) -> None:
        super().__init__()
        self.conv3d = nn.Sequential(
            ConvBNAct3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            ConvBNAct3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
        )
        self.project = nn.Linear(32 * 8, dim)
        self.tokenizer = nn.Linear(dim, num_tokens)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(reduce_spectral_depth(x.unsqueeze(1)))
        b, c, d, h, w = x.shape
        x = F.adaptive_avg_pool3d(x, (8, h, w)).permute(0, 3, 4, 1, 2).reshape(b, h * w, c * 8)
        x = self.project(x)
        attn = torch.softmax(self.tokenizer(x), dim=1)
        tokens = torch.einsum("bnd,bnt->btd", x, attn)
        tokens = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1) + self.pos
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))


class SSDGLNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dim: int = 64) -> None:
        super().__init__()
        self.spec_groups = 12
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.spec_proj = nn.Linear(1, dim // 2)
        self.lstm = nn.LSTM(dim // 2, dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.attn = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Dropout(0.3), nn.Linear(dim * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)
        spatial = spatial.mean(dim=(-1, -2))
        spec = x.mean(dim=(-1, -2)).unsqueeze(1)
        spec = F.adaptive_avg_pool1d(spec, self.spec_groups).transpose(1, 2)
        spec = self.spec_proj(spec)
        spec, _ = self.lstm(spec)
        joined = torch.cat([spatial.unsqueeze(1).expand(-1, self.spec_groups, -1), spec], dim=-1)
        weights = torch.softmax(self.attn(joined), dim=1)
        global_feat = (joined * weights).sum(dim=1)
        return self.head(global_feat)


class ViTNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dim: int = 64, patch: int = 3, depth: int = 3) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.max_tokens = 256
        self.pos = nn.Parameter(torch.zeros(1, self.max_tokens + 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if x.size(1) > self.max_tokens:
            x = x[:, : self.max_tokens]
        b = x.size(0)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


METHOD_REGISTRY: Dict[str, Callable[[int, int], nn.Module]] = {
    "3D-CNN": lambda in_channels, num_classes: Simple3DCNN(in_channels, num_classes),
    "FDSSC": lambda in_channels, num_classes: FDSSCNet(in_channels, num_classes),
    "HybridSN": lambda in_channels, num_classes: HybridSNNet(in_channels, num_classes),
    "SSRN": lambda in_channels, num_classes: SSRNNet(in_channels, num_classes),
    "SpectralFormer": lambda in_channels, num_classes: SpectralFormerNet(in_channels, num_classes),
    "SSFTT": lambda in_channels, num_classes: SSFTTNet(in_channels, num_classes),
    "SSDGL": lambda in_channels, num_classes: SSDGLNet(in_channels, num_classes),
    "VIT": lambda in_channels, num_classes: ViTNet(in_channels, num_classes),
}


def build_method_model(method: str, in_channels: int, num_classes: int) -> nn.Module:
    if method not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method {method!r}. Available: {', '.join(sorted(METHOD_REGISTRY))}")
    return METHOD_REGISTRY[method](in_channels, num_classes)

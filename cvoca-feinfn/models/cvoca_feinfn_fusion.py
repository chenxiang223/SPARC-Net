from __future__ import annotations

from itertools import chain
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .heads import BaselineCosineHead, ClassificationHead, LongTailDynamicHead
except ImportError:  # pragma: no cover - fallback for direct script execution.
    from heads import BaselineCosineHead, ClassificationHead, LongTailDynamicHead


def _conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    act: bool = True,
) -> nn.Sequential:
    padding = dilation * (kernel_size // 2)
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _class_count_ratio(
    class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor],
    num_classes: int,
) -> Optional[float]:
    if class_counts is None:
        return None
    if isinstance(class_counts, Mapping):
        counts = torch.zeros(num_classes, dtype=torch.float32)
        keys = [int(key) for key in class_counts]
        key_offset = 1 if keys and 0 not in keys and min(keys) >= 1 and max(keys) <= num_classes else 0
        for key, value in class_counts.items():
            key = int(key) - key_offset
            if 0 <= key < num_classes:
                counts[key] = float(value)
    else:
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.numel() == 0:
        return None
    counts = counts.flatten().clamp(min=1.0)
    return float((counts.max() / counts.min()).item())


def _build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim < 4:
        raise ValueError(f"token_dim must be >= 4, got {dim}.")
    quarter = max(1, dim // 4)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    inv_freq = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=dtype) / quarter))
    pos_x = grid_x.reshape(-1, 1) * inv_freq.reshape(1, -1)
    pos_y = grid_y.reshape(-1, 1) * inv_freq.reshape(1, -1)
    pos = torch.cat([pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()], dim=1)
    if pos.shape[1] < dim:
        pos = F.pad(pos, (0, dim - pos.shape[1]))
    elif pos.shape[1] > dim:
        pos = pos[:, :dim]
    return pos.unsqueeze(0)


def _build_radial_frequency_masks(
    height: int,
    width: int,
    num_bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_bands <= 0:
        raise ValueError(f"num_bands must be > 0, got {num_bands}.")

    if num_bands == 1:
        return torch.ones(1, height, width, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    radius = torch.sqrt(grid_y.square() + grid_x.square())
    radius = radius / radius.max().clamp(min=1e-6)

    edges = torch.linspace(0.0, 1.0, num_bands + 1, device=device, dtype=dtype)
    masks = []
    for band_id in range(num_bands):
        center = 0.5 * (edges[band_id] + edges[band_id + 1])
        half_width = (edges[band_id + 1] - edges[band_id]).clamp(min=1e-3) * 1.5
        mask = (1.0 - (radius - center).abs() / half_width).clamp(min=0.0)
        masks.append(mask)
    masks = torch.stack(masks, dim=0)
    masks = masks / masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
    return masks


def _build_1d_frequency_masks(
    length: int,
    num_bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}.")
    if num_bands <= 0:
        raise ValueError(f"num_bands must be > 0, got {num_bands}.")
    if num_bands == 1 or length == 1:
        return torch.ones(1, length, device=device, dtype=dtype)

    coords = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    edges = torch.linspace(0.0, 1.0, num_bands + 1, device=device, dtype=dtype)
    masks = []
    for band_id in range(num_bands):
        center = 0.5 * (edges[band_id] + edges[band_id + 1])
        half_width = (edges[band_id + 1] - edges[band_id]).clamp(min=1e-3) * 1.5
        mask = (1.0 - (coords - center).abs() / half_width).clamp(min=0.0)
        masks.append(mask)
    masks = torch.stack(masks, dim=0)
    masks = masks / masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
    return masks


class TokenContextPooling(nn.Module):
    """
    Learn a task-adaptive weighted token summary instead of mean pooling.
    """

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        hidden = max(token_dim // 2, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=True),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return torch.sum(tokens * attn.unsqueeze(-1), dim=1)


class TokenSpatialProjector(nn.Module):
    """
    Reproject token sequences to a dense modulation map for spatially aware fusion.
    """

    def __init__(self, token_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels, bias=True),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            _conv_bn_act(channels, channels, kernel_size=3, groups=channels),
            _conv_bn_act(channels, channels, kernel_size=1, act=False),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_hw: Tuple[int, int],
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        bsz, num_tokens, _ = tokens.shape
        token_h, token_w = token_hw
        if token_h * token_w != num_tokens:
            raise ValueError(
                f"TokenSpatialProjector expected {token_h * token_w} tokens from token_hw={token_hw}, "
                f"but got {num_tokens}."
            )
        feat = self.proj(tokens).transpose(1, 2).reshape(bsz, -1, token_h, token_w)
        if (token_h, token_w) != out_hw:
            feat = F.interpolate(feat, size=out_hw, mode="bilinear", align_corners=False)
        feat = self.refine(feat)
        return torch.tanh(feat)


class GaborActivation(nn.Module):
    """
    Trainable Gabor-like activation used in TSFI interaction.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.omega = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            sigma_param = self.log_sigma.squeeze(-1).squeeze(-1)
            omega_param = self.omega.squeeze(-1).squeeze(-1)
        elif x.dim() == 4:
            sigma_param = self.log_sigma
            omega_param = self.omega
        else:
            raise ValueError(f"GaborActivation expects 2D or 4D tensor, got shape {tuple(x.shape)}.")

        if x.shape[1] != sigma_param.shape[1]:
            raise ValueError(
                f"Channel mismatch in GaborActivation: x has {x.shape[1]} channels, "
                f"but parameters have {sigma_param.shape[1]} channels."
            )
        sigma = sigma_param.exp().clamp(min=1e-3)
        gaussian = torch.exp(-0.5 * (x / sigma) ** 2)
        carrier = torch.cos(omega_param * x)
        return gaussian * carrier


def _spectral_hilbert_imag(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Hilbert-transform imaginary component along spectral/channel axis.
    """
    num_bands = x.shape[1]
    h = torch.zeros(num_bands, device=x.device, dtype=x.dtype)
    if num_bands % 2 == 0:
        h[0] = 1.0
        h[num_bands // 2] = 1.0
        h[1 : num_bands // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (num_bands + 1) // 2] = 2.0
    analytic = torch.fft.ifft(torch.fft.fft(x, dim=1, norm="ortho") * h.view(1, -1, 1, 1), dim=1, norm="ortho")
    return analytic.imag


class AmplitudePhaseChannelRecalibration(nn.Module):
    """
    APCR: amplitude-phase channel recalibration over complex features.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(real.square() + imag.square() + 1e-6)
        phase = torch.atan2(imag, real + 1e-6)
        pooled = torch.cat(
            [F.adaptive_avg_pool2d(mag, 1), F.adaptive_avg_pool2d(phase, 1)],
            dim=1,
        )
        gain, phase_shift = self.mlp(pooled).chunk(2, dim=1)
        phase_shift = (phase_shift - 0.5) * math.pi
        cos_shift = torch.cos(phase_shift)
        sin_shift = torch.sin(phase_shift)
        real_out = gain * (real * cos_shift - imag * sin_shift)
        imag_out = gain * (real * sin_shift + imag * cos_shift)
        return real_out, imag_out


class AnalyticComplexSpectralEncoder(nn.Module):
    """
    ACSE: lightweight analytic complex spectral encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        analytic_init: str = "none",
        use_complex_attention: bool = True,
    ) -> None:
        super().__init__()
        valid_modes = {"none", "hilbert", "fft", "hybrid"}
        if analytic_init not in valid_modes:
            raise ValueError(f"analytic_init must be one of {valid_modes}, got {analytic_init}.")
        self.analytic_init = analytic_init
        self.use_complex_attention = use_complex_attention
        if analytic_init == "hybrid":
            # Start from a conservative Hilbert-dominant prior, then let the
            # model adapt if data truly benefits from stronger FFT mixing.
            self.imag_mix_logits = nn.Parameter(torch.tensor([-1.25, 1.75, -1.25], dtype=torch.float32))
            self.real_fft_gate = nn.Parameter(torch.tensor(-3.0))
        else:
            self.imag_mix_logits = None
            self.real_fft_gate = None
        self.real_stem = _conv_bn_act(in_channels, out_channels, kernel_size=3)
        self.imag_stem = _conv_bn_act(in_channels, out_channels, kernel_size=3)
        self.attn = AmplitudePhaseChannelRecalibration(out_channels) if use_complex_attention else None
        self.real_refine = _conv_bn_act(out_channels, out_channels, kernel_size=3)
        self.imag_refine = _conv_bn_act(out_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.analytic_init == "hilbert":
            real_seed = x
            imag_seed = _spectral_hilbert_imag(x)
        elif self.analytic_init == "fft":
            fft_seed = torch.fft.fft(x, dim=1, norm="ortho")
            real_seed = fft_seed.real
            imag_seed = fft_seed.imag
        elif self.analytic_init == "hybrid":
            fft_seed = torch.fft.fft(x, dim=1, norm="ortho")
            hilbert_imag = _spectral_hilbert_imag(x)
            mix = torch.softmax(self.imag_mix_logits, dim=0)
            imag_seed = mix[0] * x + mix[1] * hilbert_imag + mix[2] * fft_seed.imag
            fft_real_gate = torch.sigmoid(self.real_fft_gate)
            real_seed = (1.0 - fft_real_gate) * x + fft_real_gate * fft_seed.real
        else:
            real_seed = x
            imag_seed = x

        real = self.real_stem(real_seed)
        imag = self.imag_stem(imag_seed)
        if self.attn is not None:
            real, imag = self.attn(real, imag)
        real = self.real_refine(real)
        imag = self.imag_refine(imag)
        return real, imag


@dataclass
class AdapterOutput:
    feature_map: torch.Tensor
    z_spe: torch.Tensor
    z_spa: torch.Tensor
    token_hw: Tuple[int, int]


class AmplitudePhaseTokenAdapter(nn.Module):
    """
    APTA: amplitude-phase token adapter:
    1) split to magnitude/phase or real/imag
    2) 1x1 channel projection
    3) unfold to local patches
    4) add positional encoding
    5) generate z_spe and z_spa
    """

    def __init__(
        self,
        complex_channels: int,
        adapter_channels: int,
        patch_size: int = 3,
        patch_stride: int = 1,
        token_dim: int = 96,
        split_mode: str = "mag_phase",
    ) -> None:
        super().__init__()
        if split_mode not in {"mag_phase", "real_imag"}:
            raise ValueError(f"split_mode must be 'mag_phase' or 'real_imag', got {split_mode}.")
        self.split_mode = split_mode
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_pad = patch_size // 2
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_stride,
            padding=self.patch_pad,
        )
        self.input_proj = _conv_bn_act(2 * complex_channels, adapter_channels, kernel_size=1)
        self.spe_encoder = nn.Sequential(
            _conv_bn_act(adapter_channels, adapter_channels, kernel_size=1),
            _conv_bn_act(adapter_channels, adapter_channels, kernel_size=3, groups=adapter_channels),
            _conv_bn_act(adapter_channels, adapter_channels, kernel_size=1),
        )
        self.spa_encoder = nn.Sequential(
            _conv_bn_act(adapter_channels, adapter_channels, kernel_size=3, groups=adapter_channels),
            _conv_bn_act(adapter_channels, adapter_channels, kernel_size=1),
        )
        patch_vec_dim = adapter_channels * patch_size * patch_size
        self.spe_token_proj = nn.Linear(patch_vec_dim, token_dim)
        self.spa_token_proj = nn.Linear(patch_vec_dim, token_dim)

    def _split_complex(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        if self.split_mode == "mag_phase":
            mag = torch.sqrt(real.square() + imag.square() + 1e-6)
            phase = torch.atan2(imag, real + 1e-6)
            return torch.cat([mag, phase], dim=1)
        return torch.cat([real, imag], dim=1)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> AdapterOutput:
        fused = self._split_complex(real, imag)
        feat = self.input_proj(fused)
        spe_map = self.spe_encoder(feat)
        spa_map = self.spa_encoder(feat)

        _, _, h, w = feat.shape
        token_h = (h + 2 * self.patch_pad - self.patch_size) // self.patch_stride + 1
        token_w = (w + 2 * self.patch_pad - self.patch_size) // self.patch_stride + 1

        spe_patches = self.unfold(spe_map).transpose(1, 2)
        spa_patches = self.unfold(spa_map).transpose(1, 2)
        z_spe = self.spe_token_proj(spe_patches)
        z_spa = self.spa_token_proj(spa_patches)

        pos = _build_2d_sincos_pos_embed(
            height=token_h,
            width=token_w,
            dim=z_spe.shape[-1],
            device=z_spe.device,
            dtype=z_spe.dtype,
        )
        z_spe = z_spe + pos
        z_spa = z_spa + pos

        return AdapterOutput(
            feature_map=feat,
            z_spe=z_spe,
            z_spa=z_spa,
            token_hw=(token_h, token_w),
        )


class PlainTokenAdapter(nn.Module):
    """
    Baseline adapter for APTA ablation.
    It keeps the same output contract, but removes magnitude/phase splitting and
    the separate spectral/spatial token encoders.
    """

    def __init__(
        self,
        complex_channels: int,
        adapter_channels: int,
        patch_size: int = 3,
        patch_stride: int = 1,
        token_dim: int = 96,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_pad = patch_size // 2
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_stride,
            padding=self.patch_pad,
        )
        self.input_proj = _conv_bn_act(2 * complex_channels, adapter_channels, kernel_size=1)
        patch_vec_dim = adapter_channels * patch_size * patch_size
        self.token_proj = nn.Linear(patch_vec_dim, token_dim)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> AdapterOutput:
        feat = self.input_proj(torch.cat([real, imag], dim=1))

        _, _, h, w = feat.shape
        token_h = (h + 2 * self.patch_pad - self.patch_size) // self.patch_stride + 1
        token_w = (w + 2 * self.patch_pad - self.patch_size) // self.patch_stride + 1

        patches = self.unfold(feat).transpose(1, 2)
        tokens = self.token_proj(patches)
        pos = _build_2d_sincos_pos_embed(
            height=token_h,
            width=token_w,
            dim=tokens.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + pos

        return AdapterOutput(
            feature_map=feat,
            z_spe=tokens,
            z_spa=tokens,
            token_hw=(token_h, token_w),
        )


class SpatialHighFrequencyBranch(nn.Module):
    """
    Spatial branch that compensates local high-frequency details.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            _conv_bn_act(channels, channels, kernel_size=3, groups=channels),
            _conv_bn_act(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        return x + self.refine(high)


class DualAxisAmplitudePhaseFrequencyBranch(nn.Module):
    """
    DAF Branch:
    spatial FFT + spectral/channel FFT -> amplitude/phase decoupling ->
    latent-conditioned processing -> adaptive fusion.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int,
        num_radial_bands: int = 3,
        num_spectral_bands: int = 4,
    ) -> None:
        super().__init__()
        if num_radial_bands <= 0:
            raise ValueError("num_radial_bands must be > 0.")
        if num_spectral_bands <= 0:
            raise ValueError("num_spectral_bands must be > 0.")
        self.num_radial_bands = num_radial_bands
        self.num_spectral_bands = num_spectral_bands
        self.spectral_bins = channels // 2 + 1
        self.amp_refine = nn.Sequential(
            _conv_bn_act(channels, channels, kernel_size=3, groups=channels),
            _conv_bn_act(channels, channels, kernel_size=1, act=False),
        )
        self.phase_refine = nn.Sequential(
            _conv_bn_act(2 * channels, channels, kernel_size=1),
            _conv_bn_act(channels, channels, kernel_size=3, groups=channels),
            _conv_bn_act(channels, channels, kernel_size=1, act=False),
        )
        self.context_pool = TokenContextPooling(token_dim)
        self.latent_to_global = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * channels, bias=True),
        )
        self.latent_to_band = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * channels * num_radial_bands, bias=True),
        )
        self.spectral_amp_refine = nn.Sequential(
            _conv_bn_act(self.spectral_bins, self.spectral_bins, kernel_size=3, groups=self.spectral_bins),
            _conv_bn_act(self.spectral_bins, self.spectral_bins, kernel_size=1, act=False),
        )
        self.spectral_phase_refine = nn.Sequential(
            _conv_bn_act(2 * self.spectral_bins, self.spectral_bins, kernel_size=1),
            _conv_bn_act(self.spectral_bins, self.spectral_bins, kernel_size=3, groups=self.spectral_bins),
            _conv_bn_act(self.spectral_bins, self.spectral_bins, kernel_size=1, act=False),
        )
        self.latent_to_spectral_global = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * self.spectral_bins, bias=True),
        )
        self.latent_to_spectral_band = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * num_spectral_bands, bias=True),
        )
        self.domain_gate = nn.Sequential(
            _conv_bn_act(4 * channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.domain_refine = nn.Sequential(
            _conv_bn_act(2 * channels, channels, kernel_size=1),
            _conv_bn_act(channels, channels, kernel_size=3),
        )
        self.domain_res_scale = nn.Parameter(torch.tensor(0.5))
        self.spatial_amp_update_scale = nn.Parameter(torch.tensor(-1.0))
        self.spatial_phase_update_scale = nn.Parameter(torch.tensor(-1.0))
        self.spectral_amp_update_scale = nn.Parameter(torch.tensor(-1.0))
        self.spectral_phase_update_scale = nn.Parameter(torch.tensor(-1.0))
        self.anchor_gate = nn.Sequential(
            _conv_bn_act(3 * channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.output_gate = nn.Sequential(
            _conv_bn_act(5 * channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.output_context = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels, bias=True),
        )

    def _forward_spatial_frequency(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft2(x, norm="ortho")
        amp = torch.abs(freq)
        phase = torch.angle(freq)
        log_amp = torch.log1p(amp)

        amp_delta = self.amp_refine(log_amp)
        phase_input = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        phase_delta = self.phase_refine(phase_input)

        cond = self.latent_to_global(ctx)
        amp_bias, phase_bias = cond.chunk(2, dim=1)
        amp_bias = amp_bias.unsqueeze(-1).unsqueeze(-1)
        phase_bias = phase_bias.unsqueeze(-1).unsqueeze(-1)

        radial_basis = _build_radial_frequency_masks(
            height=amp.shape[-2],
            width=amp.shape[-1],
            num_bands=self.num_radial_bands,
            device=amp.device,
            dtype=amp.dtype,
        )
        radial_cond = self.latent_to_band(ctx).view(x.shape[0], 2, x.shape[1], self.num_radial_bands)
        amp_band = torch.einsum("bck,khw->bchw", radial_cond[:, 0], radial_basis)
        phase_band = torch.einsum("bck,khw->bchw", radial_cond[:, 1], radial_basis)

        amp_update = torch.sigmoid(self.spatial_amp_update_scale) * (amp_delta + amp_bias + amp_band)
        phase_update = torch.sigmoid(self.spatial_phase_update_scale) * (phase_delta + phase_bias + phase_band)

        amp_new = F.softplus(log_amp + amp_update) - 1.0
        amp_new = amp_new.clamp(min=1e-6)
        phase_new = phase + (math.pi / 4.0) * torch.tanh(phase_update)
        freq_new = torch.polar(amp_new, phase_new)
        return torch.fft.irfft2(freq_new, s=x.shape[-2:], norm="ortho")

    def _forward_spectral_frequency(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft(x, dim=1, norm="ortho")
        amp = torch.abs(freq)
        phase = torch.angle(freq)
        log_amp = torch.log1p(amp)

        amp_delta = self.spectral_amp_refine(log_amp)
        phase_input = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        phase_delta = self.spectral_phase_refine(phase_input)

        cond = self.latent_to_spectral_global(ctx)
        amp_bias, phase_bias = cond.chunk(2, dim=1)
        amp_bias = amp_bias.unsqueeze(-1).unsqueeze(-1)
        phase_bias = phase_bias.unsqueeze(-1).unsqueeze(-1)

        band_basis = _build_1d_frequency_masks(
            length=self.spectral_bins,
            num_bands=self.num_spectral_bands,
            device=amp.device,
            dtype=amp.dtype,
        )
        band_cond = self.latent_to_spectral_band(ctx).view(x.shape[0], 2, self.num_spectral_bands)
        amp_band = torch.einsum("bk,kn->bn", band_cond[:, 0], band_basis).unsqueeze(-1).unsqueeze(-1)
        phase_band = torch.einsum("bk,kn->bn", band_cond[:, 1], band_basis).unsqueeze(-1).unsqueeze(-1)

        amp_update = torch.sigmoid(self.spectral_amp_update_scale) * (amp_delta + amp_bias + amp_band)
        phase_update = torch.sigmoid(self.spectral_phase_update_scale) * (phase_delta + phase_bias + phase_band)

        amp_new = F.softplus(log_amp + amp_update) - 1.0
        amp_new = amp_new.clamp(min=1e-6)
        phase_new = phase + (math.pi / 4.0) * torch.tanh(phase_update)
        freq_new = torch.polar(amp_new, phase_new)
        return torch.fft.irfft(freq_new, n=x.shape[1], dim=1, norm="ortho")

    def forward(self, x: torch.Tensor, z_spe: torch.Tensor, raw_hint: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = self.context_pool(z_spe)
        spatial_freq_feat = self._forward_spatial_frequency(x, ctx)
        spectral_freq_feat = self._forward_spectral_frequency(x, ctx)
        if raw_hint is not None:
            anchor_gate = self.anchor_gate(torch.cat([spectral_freq_feat, raw_hint, spectral_freq_feat - raw_hint], dim=1))
            spectral_freq_feat = anchor_gate * spectral_freq_feat + (1.0 - anchor_gate) * raw_hint
        gate = self.domain_gate(
            torch.cat(
                [spatial_freq_feat, spectral_freq_feat, spatial_freq_feat - spectral_freq_feat, x],
                dim=1,
            )
        )
        fused = gate * spatial_freq_feat + (1.0 - gate) * spectral_freq_feat
        refine = self.domain_refine(torch.cat([spatial_freq_feat, spectral_freq_feat], dim=1))
        candidate = fused + torch.sigmoid(self.domain_res_scale) * refine
        anchor = raw_hint if raw_hint is not None else x
        agreement = (spatial_freq_feat - spectral_freq_feat).abs()
        update_gate = self.output_gate(
            torch.cat([candidate, x, agreement, candidate - x, spectral_freq_feat - anchor], dim=1)
        )
        ctx_gate = torch.tanh(self.output_context(ctx)).unsqueeze(-1).unsqueeze(-1)
        update_gate = torch.clamp(update_gate + 0.15 * ctx_gate, min=0.0, max=1.0)
        return x + update_gate * (candidate - x)


class TokenGuidedSpatialFrequencyInteraction(nn.Module):
    """
    TSFI: token-guided spatial-frequency interaction with Gabor activation.
    """

    def __init__(self, channels: int, token_dim: int) -> None:
        super().__init__()
        self.spa_pool = TokenContextPooling(token_dim)
        self.spe_pool = TokenContextPooling(token_dim)
        self.spa_from_spe = nn.Linear(token_dim, channels)
        self.spe_from_spa = nn.Linear(token_dim, channels)
        self.spe_to_spa_map = TokenSpatialProjector(token_dim, channels)
        self.spa_to_spe_map = TokenSpatialProjector(token_dim, channels)
        self.gabor = GaborActivation(channels=channels)
        self.spatial_gate_scale = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Sequential(
            _conv_bn_act(2 * channels, channels, kernel_size=1),
            _conv_bn_act(channels, channels, kernel_size=3),
        )

    def forward(
        self,
        spa_feat: torch.Tensor,
        spe_feat: torch.Tensor,
        z_spa: torch.Tensor,
        z_spe: torch.Tensor,
        token_hw: Tuple[int, int],
    ) -> torch.Tensor:
        spa_ctx = self.spa_pool(z_spa)
        spe_ctx = self.spe_pool(z_spe)
        spa_gate = self.gabor(self.spa_from_spe(spe_ctx)).unsqueeze(-1).unsqueeze(-1)
        spe_gate = self.gabor(self.spe_from_spa(spa_ctx)).unsqueeze(-1).unsqueeze(-1)
        spa_map = self.spe_to_spa_map(z_spe, token_hw=token_hw, out_hw=spa_feat.shape[-2:])
        spe_map = self.spa_to_spe_map(z_spa, token_hw=token_hw, out_hw=spe_feat.shape[-2:])
        spatial_scale = torch.sigmoid(self.spatial_gate_scale)

        spa_mod = spa_feat * (1.0 + spa_gate) * (1.0 + spatial_scale * spa_map)
        spe_mod = spe_feat * (1.0 + spe_gate) * (1.0 + spatial_scale * spe_map)
        fused = self.fuse(torch.cat([spa_mod, spe_mod], dim=1))
        return fused + spa_feat + spe_feat


class SpatialFrequencyFusionCore(nn.Module):
    """
    Spatial-frequency fusion core with Spa-Fre IFF + TSFI.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int,
        use_spatial_branch: bool = True,
        use_frequency_branch: bool = True,
        use_tsfi: bool = True,
        use_sfid: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if use_sfid is not None:
            use_tsfi = bool(use_sfid)
        self.use_spatial_branch = use_spatial_branch
        self.use_frequency_branch = use_frequency_branch
        self.use_tsfi = use_tsfi
        self.use_sfid = use_tsfi
        self.spa_branch = SpatialHighFrequencyBranch(channels) if use_spatial_branch else None
        self.fre_branch = (
            DualAxisAmplitudePhaseFrequencyBranch(channels, token_dim=token_dim) if use_frequency_branch else None
        )
        self.sfid = TokenGuidedSpatialFrequencyInteraction(channels, token_dim=token_dim) if use_tsfi else None

    def forward(
        self,
        x: torch.Tensor,
        z_spa: torch.Tensor,
        z_spe: torch.Tensor,
        token_hw: Tuple[int, int],
        raw_hint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_spatial_branch and not self.use_frequency_branch:
            return x

        spa_feat = self.spa_branch(x) if self.spa_branch is not None else x
        fre_feat = self.fre_branch(x, z_spe=z_spe, raw_hint=raw_hint) if self.fre_branch is not None else x

        if self.use_spatial_branch and self.use_frequency_branch:
            if self.use_tsfi and self.sfid is not None:
                return self.sfid(spa_feat, fre_feat, z_spa=z_spa, z_spe=z_spe, token_hw=token_hw)
            return 0.5 * (spa_feat + fre_feat)
        if self.use_spatial_branch:
            return spa_feat
        return fre_feat


class LightweightTransformerRecalibration(nn.Module):
    """
    Optional lightweight transformer attention for global recalibration only.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        depth: int = 1,
        max_tokens: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, h, w = x.shape
        side = max(1, int(math.sqrt(self.max_tokens)))
        pool_h = min(h, side)
        pool_w = min(w, side)

        pooled = F.adaptive_avg_pool2d(x, output_size=(pool_h, pool_w))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.out_norm(tokens)

        attn_map = tokens.transpose(1, 2).reshape(bsz, channels, pool_h, pool_w)
        attn_map = F.interpolate(attn_map, size=(h, w), mode="bilinear", align_corners=False)
        attn_map = torch.sigmoid(attn_map)
        return x * (1.0 + attn_map)


class MultiScaleContext(nn.Module):
    """
    Lightweight multi-scale context with dilated 3x3 branches (d=1/2/3).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch_d1 = _conv_bn_act(channels, channels, kernel_size=3, dilation=1)
        self.branch_d2 = _conv_bn_act(channels, channels, kernel_size=3, dilation=2)
        self.branch_d3 = _conv_bn_act(channels, channels, kernel_size=3, dilation=3)
        self.fuse = _conv_bn_act(3 * channels, channels, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch_d1(x), self.branch_d2(x), self.branch_d3(x)], dim=1)
        return x + torch.sigmoid(self.res_scale) * self.fuse(out)


class SpectralFidelityFusionGate(nn.Module):
    """
    SFF Gate: keep the upgraded fusion path discriminative without letting it
    drift too far from the stable adapter feature space.
    """

    def __init__(self, channels: int, token_dim: int) -> None:
        super().__init__()
        self.context_pool = TokenContextPooling(token_dim)
        self.context_to_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, channels, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(5 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        candidate_feat: torch.Tensor,
        base_feat: torch.Tensor,
        raw_hint: torch.Tensor,
        z_spe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(
            torch.cat(
                [
                    candidate_feat,
                    base_feat,
                    raw_hint,
                    candidate_feat - base_feat,
                    base_feat - raw_hint,
                ],
                dim=1,
            )
        )
        ctx_bias = torch.tanh(self.context_to_gate(self.context_pool(z_spe))).unsqueeze(-1).unsqueeze(-1)
        gate = torch.clamp(gate + 0.15 * ctx_bias, min=0.0, max=1.0)
        out = base_feat + gate * (candidate_feat - base_feat)
        return out, gate


class SpectralAnchorBypass(nn.Module):
    """
    SAB: preserve raw spectral signature through a gated bypass.
    """

    def __init__(self, raw_channels: int, fuse_channels: int, token_dim: int) -> None:
        super().__init__()
        self.raw_proj = _conv_bn_act(raw_channels, fuse_channels, kernel_size=1)
        self.context_pool = TokenContextPooling(token_dim)
        self.context_to_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, fuse_channels, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(4 * fuse_channels, fuse_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.GELU(),
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fused_feat: torch.Tensor,
        raw_spectral: torch.Tensor,
        z_spe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_feat = self.raw_proj(raw_spectral)
        gate = self.gate(torch.cat([fused_feat, raw_feat, fused_feat - raw_feat, fused_feat * raw_feat], dim=1))
        if z_spe is not None:
            ctx_bias = torch.tanh(self.context_to_gate(self.context_pool(z_spe))).unsqueeze(-1).unsqueeze(-1)
            gate = torch.clamp(gate + 0.15 * ctx_bias, min=0.0, max=1.0)
        out = gate * fused_feat + (1.0 - gate) * raw_feat
        return out, gate


class BaselineSpatialSpectralBackbone(nn.Module):
    """
    Plain convolutional backbone used for strict Innovation1 ablation.

    It keeps only a lightweight residual CNN over the raw HSI patch and removes
    the proposed complex-valued, spatial-frequency and spectral-fidelity modules.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem = _conv_bn_act(in_channels, out_channels, kernel_size=3)
        self.block1 = _conv_bn_act(out_channels, out_channels, kernel_size=3)
        self.block2 = _conv_bn_act(out_channels, out_channels, kernel_size=3)
        self.block3 = _conv_bn_act(out_channels, out_channels, kernel_size=3)
        self.out_proj = _conv_bn_act(out_channels, out_channels, kernel_size=1, act=False)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        refined = self.block3(self.block2(self.block1(feat)))
        return feat + torch.sigmoid(self.res_scale) * self.out_proj(refined)


class SPARCNet(nn.Module):
    """
    Optimized fusion architecture:
    Innovation1 on:
    ACSE -> APTA -> Spatial-Frequency Fusion Core (Spa-Fre IFF + TSFI) ->
    optional Transformer -> Multi-scale Context -> SAB.

    Innovation1 off:
    Plain residual CNN backbone.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        adapter_channels: int = 64,
        token_dim: int = 96,
        patch_size: int = 3,
        patch_stride: int = 1,
        split_mode: str = "mag_phase",
        analytic_init: str = "none",
        backbone_mode: str = "innovation1",
        use_complex_attention: bool = True,
        use_apta: bool = True,
        use_spatial_branch: bool = True,
        use_frequency_branch: bool = True,
        use_sfid: bool = True,
        use_tsfi: Optional[bool] = None,
        use_transformer: bool = True,
        use_multi_scale: bool = True,
        use_sff_gate: bool = True,
        use_spectral_bypass: bool = True,
        transformer_heads: int = 4,
        transformer_depth: int = 1,
        transformer_max_tokens: int = 256,
        classifier_type: str = "long_tail",
        head_mode: str = "innovation2",
        class_counts: Optional[Sequence[int] | Mapping[int, int] | torch.Tensor] = None,
        head_embed_dim: Optional[int] = None,
        head_contrast_dim: int = 128,
        use_prototype_branch: bool = True,
        use_dynamic_gate: bool = True,
        use_auxiliary_heads: bool = True,
        prototype_momentum: float = 0.9,
        prototype_blend_min: float = 0.05,
        prototype_blend_max: float = 0.45,
        alignment_scale_limit: float = 0.35,
        alignment_bias_limit: float = 0.20,
        freeze_source_classifier_in_stage2: bool = True,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if classifier_type not in {"linear", "long_tail"}:
            raise ValueError("classifier_type must be 'linear' or 'long_tail'.")
        if backbone_mode not in {"innovation1", "baseline"}:
            raise ValueError("backbone_mode must be 'innovation1' or 'baseline'.")
        if head_mode not in {"innovation2", "baseline"}:
            raise ValueError("head_mode must be 'innovation2' or 'baseline'.")
        if use_tsfi is None:
            use_tsfi = bool(use_sfid)

        self.backbone_mode = backbone_mode
        self.head_mode = head_mode
        self.baseline_backbone = None
        self.cvoca = None
        self.adapter = None
        self.raw_hint_proj = None
        self.feinfn = None
        self.transformer = None
        self.multi_scale = None
        self.fusion_stabilizer = None
        self.spectral_bypass = None
        self.use_transformer = False
        self.use_multi_scale = False
        self.use_spectral_bypass = False

        if backbone_mode == "baseline":
            self.baseline_backbone = BaselineSpatialSpectralBackbone(in_channels, adapter_channels)
        else:
            self.cvoca = AnalyticComplexSpectralEncoder(
                in_channels,
                out_channels=base_channels,
                analytic_init=analytic_init,
                use_complex_attention=use_complex_attention,
            )
            if use_apta:
                self.adapter = AmplitudePhaseTokenAdapter(
                    complex_channels=base_channels,
                    adapter_channels=adapter_channels,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    token_dim=token_dim,
                    split_mode=split_mode,
                )
            else:
                self.adapter = PlainTokenAdapter(
                    complex_channels=base_channels,
                    adapter_channels=adapter_channels,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    token_dim=token_dim,
                )
            self.raw_hint_proj = _conv_bn_act(in_channels, adapter_channels, kernel_size=1)
            self.feinfn = SpatialFrequencyFusionCore(
                channels=adapter_channels,
                token_dim=token_dim,
                use_spatial_branch=use_spatial_branch,
                use_frequency_branch=use_frequency_branch,
                use_tsfi=use_tsfi,
            )
            self.use_transformer = use_transformer
            if use_transformer:
                self.transformer = LightweightTransformerRecalibration(
                    channels=adapter_channels,
                    num_heads=transformer_heads,
                    depth=transformer_depth,
                    max_tokens=transformer_max_tokens,
                )
            else:
                self.transformer = nn.Identity()
            self.use_multi_scale = use_multi_scale
            self.multi_scale = MultiScaleContext(adapter_channels) if use_multi_scale else None
            self.fusion_stabilizer = (
                SpectralFidelityFusionGate(adapter_channels, token_dim=token_dim)
                if use_sff_gate
                and any((use_spatial_branch, use_frequency_branch, use_tsfi, use_transformer, use_multi_scale))
                else None
            )
            self.use_spectral_bypass = use_spectral_bypass
            self.spectral_bypass = (
                SpectralAnchorBypass(
                    raw_channels=in_channels,
                    fuse_channels=adapter_channels,
                    token_dim=token_dim,
                )
                if use_spectral_bypass
                else None
            )
        self.classifier_type = classifier_type
        class_ratio = _class_count_ratio(class_counts, num_classes) if num_classes is not None else None
        use_balanced_fallback_head = (
            head_mode == "innovation2"
            and class_ratio is not None
            and class_ratio <= 30.0
        )
        self.prototype_head_active = head_mode == "innovation2" and not use_balanced_fallback_head
        if num_classes is None:
            self.classifier = None
        elif classifier_type == "linear":
            self.classifier = ClassificationHead(adapter_channels, num_classes)
        elif head_mode == "baseline" or use_balanced_fallback_head:
            self.classifier = BaselineCosineHead(
                adapter_channels,
                num_classes,
                embed_dim=head_embed_dim,
                contrast_dim=head_contrast_dim,
                use_auxiliary_heads=use_auxiliary_heads,
            )
        else:
            self.classifier = LongTailDynamicHead(
                adapter_channels,
                num_classes,
                embed_dim=head_embed_dim,
                contrast_dim=head_contrast_dim,
                class_counts=class_counts,
                use_prototype_branch=use_prototype_branch,
                use_dynamic_gate=use_dynamic_gate,
                prototype_momentum=prototype_momentum,
                prototype_blend_min=prototype_blend_min,
                prototype_blend_max=prototype_blend_max,
                use_auxiliary_heads=use_auxiliary_heads,
                alignment_scale_limit=alignment_scale_limit,
                alignment_bias_limit=alignment_bias_limit,
                freeze_source_classifier_in_stage2=freeze_source_classifier_in_stage2,
            )

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, return_aux: bool = False):
        if self.backbone_mode == "baseline":
            final_feat = self.baseline_backbone(x)
            fused = final_feat
            early_feature = final_feat
            z_spe = None
            z_spa = None
            stability_gate = torch.ones_like(final_feat)
            gate = torch.ones_like(final_feat)
        else:
            real, imag = self.cvoca(x)
            adapter_out = self.adapter(real, imag)
            raw_hint = self.raw_hint_proj(x)

            fused = self.feinfn(
                adapter_out.feature_map,
                z_spa=adapter_out.z_spa,
                z_spe=adapter_out.z_spe,
                token_hw=adapter_out.token_hw,
                raw_hint=raw_hint,
            )
            fused = self.transformer(fused) if self.use_transformer else fused
            if self.multi_scale is not None:
                fused = self.multi_scale(fused)
            if self.fusion_stabilizer is not None:
                fused, stability_gate = self.fusion_stabilizer(
                    candidate_feat=fused,
                    base_feat=adapter_out.feature_map,
                    raw_hint=raw_hint,
                    z_spe=adapter_out.z_spe,
                )
            else:
                stability_gate = torch.ones_like(fused)

            if self.spectral_bypass is not None:
                final_feat, gate = self.spectral_bypass(fused_feat=fused, raw_spectral=x, z_spe=adapter_out.z_spe)
            else:
                final_feat = fused
                gate = torch.ones_like(fused)
            early_feature = adapter_out.feature_map
            z_spe = adapter_out.z_spe
            z_spa = adapter_out.z_spa
        head_output = None
        if self.classifier is None:
            logits = None
        elif self.classifier_type == "linear":
            logits = self.classifier(final_feat)
        else:
            head_output = self.classifier(
                final_feat,
                mid_feat=fused,
                early_feat=early_feature,
                targets=targets,
            )
            logits = head_output.logits

        if not return_aux:
            return logits if logits is not None else final_feat

        aux: Dict[str, torch.Tensor] = {
            "feature": final_feat,
            "mid_feature": fused,
            "early_feature": early_feature,
            "gate": gate,
            "stability_gate": stability_gate,
        }
        if z_spe is not None:
            aux["z_spe"] = z_spe
        if z_spa is not None:
            aux["z_spa"] = z_spa
        if logits is not None:
            aux["logits"] = logits
        if head_output is not None:
            aux["raw_logits"] = head_output.raw_logits
            aux["pooled_feature"] = head_output.pooled_feature
            aux["embedding"] = head_output.embedding
            aux["projection"] = head_output.projection
            aux["main_logits"] = head_output.main_logits
            aux["raw_main_logits"] = head_output.raw_main_logits
            aux["prototype_logits"] = head_output.prototype_logits
            aux["raw_prototype_logits"] = head_output.raw_prototype_logits
            aux["dynamic_gate"] = head_output.dynamic_gate
            aux["logit_scale"] = head_output.logit_scale
            if head_output.fusion_weights is not None:
                aux["fusion_weights"] = head_output.fusion_weights
            if head_output.head_regularizer is not None:
                aux["head_regularizer"] = head_output.head_regularizer
            if head_output.aux_logits_mid is not None:
                aux["aux_logits_mid"] = head_output.aux_logits_mid
            if head_output.aux_logits_early is not None:
                aux["aux_logits_early"] = head_output.aux_logits_early
        return aux

    def reset_classifier(self) -> None:
        if self.classifier is not None and hasattr(self.classifier, "reset_classifier"):
            self.classifier.reset_classifier()

    def apply_classifier_tau_normalization(self, tau: float = 1.0) -> None:
        if self.classifier is not None and hasattr(self.classifier, "apply_tau_normalization"):
            self.classifier.apply_tau_normalization(tau=tau)

    def set_prototype_correction_scale(self, scale: float) -> None:
        if self.classifier is not None and hasattr(self.classifier, "set_prototype_correction_scale"):
            self.classifier.set_prototype_correction_scale(scale)

    def get_prototype_correction_scale(self) -> Optional[float]:
        if self.classifier is not None and hasattr(self.classifier, "get_prototype_correction_scale"):
            return float(self.classifier.get_prototype_correction_scale())
        return None

    def set_head_training_mode(
        self,
        *,
        output_mode: str = "main",
        update_prototypes: bool = True,
        guided_context: bool = False,
    ) -> None:
        if self.classifier is not None and hasattr(self.classifier, "set_head_training_mode"):
            self.classifier.set_head_training_mode(
                output_mode=output_mode,
                update_prototypes=update_prototypes,
                guided_context=guided_context,
            )

    def freeze_backbone(self) -> None:
        """
        Freeze the feature extractor and keep the classifier head trainable.
        """

        backbone_modules = [
            self.baseline_backbone,
            self.cvoca,
            self.adapter,
            self.feinfn,
            self.transformer,
            self.multi_scale,
            self.fusion_stabilizer,
            self.spectral_bypass,
        ]
        for module in backbone_modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = True
            if hasattr(self.classifier, "freeze_source_classifier"):
                self.classifier.freeze_source_classifier()

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        if self.classifier is not None and hasattr(self.classifier, "release_source_classifier"):
            self.classifier.release_source_classifier()

    def classifier_parameters(self):
        if self.classifier is None:
            return iter(())
        return self.classifier.parameters()

    def backbone_parameters(self):
        backbone_parameter_groups = [
            self.baseline_backbone.parameters() if self.baseline_backbone is not None else iter(()),
            self.cvoca.parameters() if self.cvoca is not None else iter(()),
            self.adapter.parameters() if self.adapter is not None else iter(()),
            self.feinfn.parameters() if self.feinfn is not None else iter(()),
            self.transformer.parameters() if self.transformer is not None else iter(()),
        ]
        if self.multi_scale is not None:
            backbone_parameter_groups.append(self.multi_scale.parameters())
        if self.fusion_stabilizer is not None:
            backbone_parameter_groups.append(self.fusion_stabilizer.parameters())
        if self.spectral_bypass is not None:
            backbone_parameter_groups.append(self.spectral_bypass.parameters())
        return chain(*backbone_parameter_groups)

    @property
    def classifier_margin_mode(self) -> str:
        if self.classifier is None:
            return "linear"
        return getattr(self.classifier, "margin_mode", "linear")


# Backward-compatible aliases for old experiment scripts and checkpoints.
ComplexChannelAttention = AmplitudePhaseChannelRecalibration
CVOCAFeatureExtractor = AnalyticComplexSpectralEncoder
CVOCAFeINFNAdapter = AmplitudePhaseTokenAdapter
FrequencyAmplitudePhaseBranch = DualAxisAmplitudePhaseFrequencyBranch
SFIDInteraction = TokenGuidedSpatialFrequencyInteraction
FeINFNCore = SpatialFrequencyFusionCore
RepresentationPreservingFusionGate = SpectralFidelityFusionGate
SpectralResidualBypassGate = SpectralAnchorBypass
CVOCAFeINFNFusion = SPARCNet


if __name__ == "__main__":
    # Minimal sanity check.
    model = SPARCNet(
        in_channels=128,
        base_channels=64,
        adapter_channels=64,
        token_dim=96,
        patch_size=3,
        patch_stride=1,
        split_mode="mag_phase",
        use_transformer=True,
        class_counts={i: 100 - 4 * i for i in range(16)},
        num_classes=16,
    )
    x = torch.randn(2, 128, 15, 15)
    y = torch.tensor([0, 5])
    out = model(x, targets=y, return_aux=True)
    print("feature:", tuple(out["feature"].shape))
    print("logits:", tuple(out["logits"].shape))
    print("main_logits:", tuple(out["main_logits"].shape))
    print("projection:", tuple(out["projection"].shape))
    print("z_spe:", tuple(out["z_spe"].shape))
    print("z_spa:", tuple(out["z_spa"].shape))

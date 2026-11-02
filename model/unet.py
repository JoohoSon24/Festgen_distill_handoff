from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    return math.gcd(channels, 32)


def timestep_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phase = t.float()[:, None] * frequencies[None]
    embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
    return F.pad(embedding, (0, width - embedding.shape[-1]))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, 2 * out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).to(h.dtype).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x).reshape(batch, channels, -1)).chunk(3, dim=1)
        # One head keeps the baseline simple; attention is only enabled at low resolution.
        weights = torch.softmax(torch.bmm(q.transpose(1, 2).float(), k.float()) / math.sqrt(channels), dim=-1)
        out = torch.bmm(v.float(), weights.transpose(1, 2)).to(x.dtype)
        return x + self.proj(out).reshape(batch, channels, height, width)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float, attention: bool):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_ch, dropout)
        self.attn = AttentionBlock(out_ch) if attention else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, emb))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class ClassConditionalUNet(nn.Module):
    """Pixel U-Net. Labels 0..K-1 are classes; K is the learned null class."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        num_classes: int,
        model_channels: int,
        channel_mult: tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: tuple[int, ...],
        dropout: float = 0.0,
        activation_checkpoint_resolutions: tuple[int, ...] = (),
    ):
        super().__init__()
        if image_size % (2 ** (len(channel_mult) - 1)):
            raise ValueError("image_size must be divisible by the downsampling factor")
        self.num_classes = num_classes
        self.activation_checkpoint_resolutions = activation_checkpoint_resolutions
        emb_ch = 4 * model_channels
        self.time_mlp = nn.Sequential(nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))
        self.class_emb = nn.Embedding(num_classes + 1, emb_ch)
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.down = nn.ModuleList()
        skip_channels = [model_channels]
        channels = model_channels
        resolution = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(Block(channels, out_ch, emb_ch, dropout, resolution in attention_resolutions))
                channels = out_ch
                skip_channels.append(channels)
            if level != len(channel_mult) - 1:
                self.down.append(Downsample(channels))
                skip_channels.append(channels)
                resolution //= 2
        self.middle1 = Block(channels, channels, emb_ch, dropout, True)
        self.middle2 = Block(channels, channels, emb_ch, dropout, False)
        self.up = nn.ModuleList()
        for level in reversed(range(len(channel_mult))):
            out_ch = model_channels * channel_mult[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up.append(Block(channels + skip_ch, out_ch, emb_ch, dropout, resolution in attention_resolutions))
                channels = out_ch
            if level != 0:
                self.up.append(Upsample(channels))
                resolution *= 2
        assert not skip_channels
        self.out_norm = nn.GroupNorm(_groups(channels), channels)
        self.out_conv = nn.Conv2d(channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def run_block(self, layer: Block, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled() and x.shape[-1] in self.activation_checkpoint_resolutions:
            return checkpoint(layer,x,emb,use_reentrant=False,preserve_rng_state=True)
        return layer(x,emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or t.shape != (x.shape[0],) or labels.shape != (x.shape[0],):
            raise ValueError("Expected x[B,C,H,W], t[B], labels[B]")
        if torch.any((labels < 0) | (labels > self.num_classes)):
            raise ValueError("Class labels must be in [0, num_classes], including null")
        emb = self.time_mlp(timestep_embedding(t, self.class_emb.embedding_dim // 4)) + self.class_emb(labels)
        h = self.input_conv(x)
        skips = [h]
        for layer in self.down:
            h = self.run_block(layer,h,emb) if isinstance(layer, Block) else layer(h)
            skips.append(h)
        h = self.run_block(self.middle2,self.run_block(self.middle1,h,emb),emb)
        for layer in self.up:
            h = self.run_block(layer,torch.cat((h,skips.pop()),dim=1),emb) if isinstance(layer,Block) else layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))

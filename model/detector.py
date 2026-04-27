"""
detector.py
===========
EfficientNet-V2-S encoder + U-Net decoder → corner heatmap.

Architecture
------------
Encoder  : EfficientNet-V2-S (pretrained ImageNet), feature maps at
           strides [2, 4, 8, 16, 32] with channels [24, 48, 64, 160, 256].
Decoder  : 4 upsampling blocks with skip connections (U-Net style),
           followed by a final 2× upsample to recover full resolution.
Head     : 3×3 conv + 1×1 conv → single-channel logit map.

Input  : (B, 3, H, W) float32 normalised to [0, 1]
Output : (B, 1, H, W) float32 raw logits  (apply sigmoid for probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# EfficientNet-V2-S intermediate channel counts (strides 2→32)
_EFF_V2S_CH = [24, 48, 64, 160, 256]


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _UpBlock(nn.Module):
    """2× transposed-conv upsample + skip-connection + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = _ConvBNReLU(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x  = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh > 0 or dw > 0:                       # pad if spatial sizes differ
            x = F.pad(x, [0, dw, 0, dh])
        return self.conv(torch.cat([x, skip], dim=1))


class CornerDetector(nn.Module):
    """Checkerboard corner heatmap detector."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        ch = _EFF_V2S_CH
        self.dec4   = _UpBlock(ch[4], ch[3], 256)   # /32 → /16
        self.dec3   = _UpBlock(256,   ch[2], 128)   # /16 → /8
        self.dec2   = _UpBlock(128,   ch[1],  64)   # /8  → /4
        self.dec1   = _UpBlock(64,    ch[0],  32)   # /4  → /2
        self.up_out = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.head   = nn.Sequential(
            _ConvBNReLU(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f4, f8, f16, f32 = self.encoder(x)
        x = self.dec4(f32, f16)
        x = self.dec3(x,   f8)
        x = self.dec2(x,   f4)
        x = self.dec1(x,   f2)
        x = self.up_out(x)
        return self.head(x)                         # (B, 1, H, W) logits

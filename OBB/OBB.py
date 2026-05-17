import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PECA_Attention(nn.Module):
    """Progressive Efficient Channel Attention (PECA) (Contribution 3)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        y = F.adaptive_avg_pool2d(x, 1)               # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)            # (B, 1, C)
        y1 = self.silu(self.conv1(y))                  # Large Receptive Field
        y2 = self.conv2(y1)                            # Fine-grained Adjustment
        y = self.sigmoid(y2)                            # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)          # (B, C, 1, 1)
        return x * y                                   # Channel-wise Modulation


class AngleAwareEnhance(nn.Module):
    """Dual-Path Angle-Aware Enhancement (AAE) (Contribution 2)"""
    def __init__(self, c2, hidden=None):
        super().__init__()
        hidden = hidden or max(c2 // 8, 8)
        # Global Channel Descriptor
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        # Local Spatial Rectification
        self.spatial_conv = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Path
        glob = self.global_pool(x)          # (B, C, 1, 1)
        glob = self.fc(glob)                # (B, C, 1, 1)
        # Spatial Path
        spat = self.spatial_conv(x)         # (B, C, H, W)
        spat = self.global_pool(spat)       # Spatial information compressed into the channel dimension
        # Fusion
        w = self.sigmoid(glob + spat)       # Attention weights in (0, 1)
        return (1.0 + w)                    # Enhancement factor mapped to (1, 2)


class GhostConv_OBB(nn.Module):
    """Improved GhostConv for OBB (Integrating Contributions 1–3)"""
    def __init__(self, c1, c2, k=1, s=1, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.g = g if c1 % g == 0 else 1

        # Primary Convolution Branch
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, self.c_, k, s, k // 2, groups=self.g, bias=False),
            nn.BatchNorm2d(self.c_),
            nn.SiLU(inplace=True),
        )

        # Cheap Operation Branch
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(self.c_, c2 - self.c_, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(c2 - self.c_),
            nn.SiLU(inplace=True),
        )

        # Angle-Aware Enhancement (AAE) (replaces the original angle_enhance) 
        self.angle_enhance = AngleAwareEnhance(c2)

        # Progressive Channel Attention (PCA)
        self.channel_att = PECA_Attention(c2)

        # Residual connection (employing a 1×1 projection when input and output channels or dimensions mismatch)
        self.use_residual = (s == 1 and c1 == c2)
        if c1 != c2 or s != 1:
            self.residual_proj = nn.Conv2d(c1, c2, 1, s, 0, bias=False)
        else:
            self.residual_proj = nn.Identity()

        # Learnable Residual Fusion Coefficients
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = self.residual_proj(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        out = torch.cat([x1, x2], dim=1)          # Channel Concatenation

        # Angle Enhancement
        att = self.angle_enhance(out)             # (1, 2) Range
        out = out * att

        # Channel Attention
        out = self.channel_att(out)

        # Numerical Sanity Check
        if not torch.isfinite(out).all():
            print("Warning: GhostConv_OBB has NaN/Inf, falling back to enhanced concat")
            return out * 0.0 + identity * 1.0   # Safe Fallback to Projected Residual

        # Adaptive Residual Fusion (Contribution 1)
        if self.use_residual:
            out = identity * (1.0 - self.residual_scale) + out * self.residual_scale
        return out


class GhostBottleneck_OBB(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv = nn.Sequential(
            GhostConv_OBB(c1, c_, 1, 1, g=g),
            GhostConv_OBB(c_, c2, 3, 1, g=g) if c_ == c2 else
            nn.Sequential(
                nn.Conv2d(c_, c2, 3, 1, 1, groups=g if c_ % g == 0 else 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            ),
        )
        self.add = shortcut and c1 == c2
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        out = self.conv(x)
        if self.add and torch.isfinite(out).all():
            return x + self.scale * out
        return out if torch.isfinite(out).all() else x


class C3k2_GhostConv_OBB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = GhostConv_OBB(c1, 2 * self.c, 1, 1)
        self.cv2 = GhostConv_OBB((2 + n) * self.c, c2, 1, 1)

        from ultralytics.nn.modules.block import Bottleneck
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, k=((k, k), (k, k)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
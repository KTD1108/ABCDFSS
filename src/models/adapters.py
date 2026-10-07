import torch
import torch.nn as nn

class PointwiseAdapter(nn.Module):
    """
    Standard Conv 1x1 adapter projecting multi-channel backbone features
    into low-dimensional task representations.
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super(PointwiseAdapter, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.proj(out)
        return out


class DepthwiseSeparableAdapter(nn.Module):
    """
    Depthwise Separable Conv 3x3 Adapter with Residual Shortcut.
    Captures local spatial contextual relationships and edge contours
    while maintaining a minimal parameter footprint to prevent overfitting
    in few-shot regimes.
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super(DepthwiseSeparableAdapter, self).__init__()
        # Depthwise Conv 3x3: spatial correlation within channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=1, padding_mode='replicate',
            groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Pointwise Conv 1x1: channel mixing and dimensionality reduction
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        
        # Output projection
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Residual shortcut: projects in_channels directly to out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.residual(x)
        
        out = self.depthwise(x)
        out = self.bn_dw(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn_pw(out)
        out = self.relu(out)
        
        out = self.proj(out) + shortcut
        return out


def build_adapter(adapter_type: str, in_channels: int, out_channels: int = 64) -> nn.Module:
    """Factory helper to instantiate adapters by name."""
    adapter_name = adapter_type.lower()
    if adapter_name in ['depthwise_separable_3x3', 'dw_pw_3x3', 'depthwise3x3', 'dw3x3']:
        return DepthwiseSeparableAdapter(in_channels, out_channels)
    elif adapter_name in ['conv1x1', 'pointwise', '1x1']:
        return PointwiseAdapter(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown adapter type: '{adapter_type}'. Available: ['depthwise_separable_3x3', 'conv1x1']")

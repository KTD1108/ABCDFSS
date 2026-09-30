import torch
import torch.nn as nn
from torchvision.models import resnet

class ResNetBackbone(nn.Module):
    """
    Standard ResNet-50 feature extractor extracting multi-layer feature representations
    from bottleneck residual blocks across all 4 stages.
    """
    def __init__(self, weights=resnet.ResNet50_Weights.DEFAULT):
        super(ResNetBackbone, self).__init__()
        self.backbone = resnet.resnet50(weights=weights)
        self.backbone.eval()
        
        # Freeze all backbone weights (Feature extractor remains fixed)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.stages = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ]

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor):
        """
        Extract multi-level feature pyramid maps.
        Args:
            images: Input tensor of shape [B, 3, H, W]
        Returns:
            List of 16 feature maps from bottleneck blocks across stages 1..4
        """
        # Initial stem
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        features = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
                features.append(x.clone())

        return features

    def forward(self, images: torch.Tensor):
        return self.extract_features(images)

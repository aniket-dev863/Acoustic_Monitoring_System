"""
EfficientNet / MobileNet Transfer Learning Architecture for Acoustic Anomaly Detection.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AcousticEfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for single-channel audio Mel spectrograms.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout_rate: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        # Modify first conv layer to accept 1 channel (Mel spectrogram) instead of 3 (RGB)
        orig_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        if pretrained:
            # Average pre-trained RGB weights across the 3 channels for initialization
            with torch.no_grad():
                new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)

        base_model.features[0][0] = new_conv
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Replace classification head
        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deep pooled feature embedding."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        logits = self.classifier(x)
        return logits

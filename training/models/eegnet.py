"""
PyTorch implementation of EEGNet (Lawhern et al., 2018) adapted for
4-channel Muse S wearable EEG workload estimation (256 Hz, 4.0s windows).

Architecture:
  Block 1: Temporal Conv2d (learns bandpass filters) -> Depthwise Spatial Conv2d (mixes TP9/AF7/AF8/TP10)
  Block 2: Depthwise-Separable Conv2d (learns temporal summaries)
  Classifier: Linear projection to workload classes (low / medium / high)
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 4,
        n_samples: int = 1024,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # Block 1: Temporal filtering + Depthwise spatial filtering
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kern_length),
                padding=(0, kern_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable convolution (depthwise temporal + pointwise mixing)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Dynamically compute flattened feature dimension to avoid padding mismatch
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat_dim = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either [B, C, T] or [B, 1, C, T]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feats = self._forward_features(x)
        return self.classifier(feats)

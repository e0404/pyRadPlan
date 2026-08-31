"""Dummy model for pyRadPlan.ai.modelhub local-load tests.

A tiny 3D CNN mapping a multi-channel volume to a scalar. It exists only to
exercise the loading machinery against a real (committed) model directory.
"""

import torch
from torch import nn


class DummyNet(nn.Module):
    """A tiny 3D CNN producing a scalar output."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 4,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(hidden_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)

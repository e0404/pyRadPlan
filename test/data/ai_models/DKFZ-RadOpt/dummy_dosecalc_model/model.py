"""Dummy dose-calc network for AIBeamletEngine tests.

A tiny 3D CNN that maps a stacked (CT, energy-map) cuboid to a two-channel
(physical dose, LET-weighted dose) cuboid of the same spatial shape. It exists
only to exercise the AI dose-engine machinery against a real (committed) model
directory, and is deterministic: the dropout layer is kept for weight
compatibility but configured with ``p=0``.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DummyDoseCalcNet(nn.Module):
    """Predicts a (physical_dose, let_dose) cuboid pair from CT + energy."""

    def __init__(self, hidden_channels: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv3d(hidden_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self.features(x)))

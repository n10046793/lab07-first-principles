"""The network for Lakeshore Bike Share.

Two hidden layers with ReLU, ending in one value with no activation,
because the target is a count of rentals, not a probability.
"""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Two hidden layers. Returns one predicted count per row."""

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)

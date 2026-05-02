"""Module with basic PPO discrete actor network."""

import torch
import torch.nn as nn


class DiscretePPOActor(nn.Module):
    """Basic agent actor neural network for PPO."""
    def __init__(self, n_input: int, n_actions: int, hidden: int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass."""
        return self.net(x.type(torch.float32))

"""Module with basic PPO critic network."""

import torch
import torch.nn as nn


class PPOCritic(nn.Module):
    """Basic agent critic neural network for PPO."""
    def __init__(self, n_input: int, hidden: int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass."""
        return self.net(x.type(torch.float32))

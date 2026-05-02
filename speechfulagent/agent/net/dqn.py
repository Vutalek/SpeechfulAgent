"""Module with basic DQN MLP network for discrete environments."""

import torch
import torch.nn as nn


class MLPDQN(nn.Module):
    """Basic agent neural network for DQN."""
    def __init__(self, n_input: int, n_actions: int, hidden: int=256):
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

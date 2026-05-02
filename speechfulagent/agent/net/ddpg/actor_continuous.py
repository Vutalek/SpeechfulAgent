"""Module with basic DDPG continuous actor network."""

import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    """Basic agent actor neural network for DDPG."""
    def __init__(self, n_input: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass."""
        return self.net(x.type(torch.float32))

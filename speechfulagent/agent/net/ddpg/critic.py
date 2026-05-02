"""Module with basic DDPG continuous critic network."""

import torch
import torch.nn as nn


class DDPGCritic(nn.Module):
    """Basic agent critic neural network for DDPG."""
    def __init__(self, n_input: int, n_actions: int):
        super().__init__()
        self.obs = nn.Sequential(
            nn.Linear(n_input, 400),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(400+n_actions, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass."""
        obs = self.obs(x.type(torch.float32))
        return self.net(torch.cat([obs, a], dim=1))

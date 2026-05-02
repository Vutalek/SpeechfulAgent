"""Module with basic A2C network for continuous environments."""

from typing import Tuple

import torch
import torch.nn as nn


class ContinuousA2C(nn.Module):
    """Basic agent neural network for A2C."""
    def __init__(self, n_input: int, n_actions: int, hidden: int=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softplus()
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Neural network forward pass."""
        backbone = self.backbone(x.type(torch.float32))
        return self.mu(backbone), self.var(backbone), self.critic(backbone)

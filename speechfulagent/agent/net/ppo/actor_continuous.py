"""Module with basic PPO continuous actor network."""

import torch
import torch.nn as nn


class ContinuousPPOActor(nn.Module):
    """Basic agent actor neural network for PPO."""
    def __init__(self, n_input: int, n_actions: int, hidden: int=128):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass."""
        return self.mu(x.type(torch.float32))

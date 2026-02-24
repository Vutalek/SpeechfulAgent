from typing import Tuple

import torch
import torch.nn as nn


class ContinuousA2C(nn.Module):
    def __init__(self, n_input: int, n_actions: int, hidden: int=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU()
        )
        
        self.mu = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

        self.var = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone = self.backbone(x.type(torch.float32))
        return self.mu(backbone), self.var(backbone), self.critic(backbone)

from typing import Tuple

import torch
import torch.nn as nn


class DicreteA2C(nn.Module):
    def __init__(self, n_input: int, n_actions: int, hidden: int=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone = self.backbone(x.type(torch.float32))
        return self.actor(backbone), self.critic(backbone)

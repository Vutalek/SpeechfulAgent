from typing import Tuple

import torch
import torch.nn as nn


class PPO(nn.Module):
    def __init__(self, input_shape: int, n_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone = self.backbone(x.type(torch.float32))
        return self.actor(backbone), self.critic(backbone)
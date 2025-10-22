import typing as tt

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.type(torch.float32))

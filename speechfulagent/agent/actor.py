import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_shape: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 400),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.type(torch.float32))
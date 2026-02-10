import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.type(torch.float32))
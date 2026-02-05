import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_shape: int, n_actions: int):
        super().__init__()
        self.obs = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(256+n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        obs = self.obs(x.type(torch.float32))
        return self.net(torch.cat([obs, a], dim=1))

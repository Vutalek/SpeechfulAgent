import torch
import torch.nn as nn


class DiscretePPOActor(nn.Module):
    def __init__(self, n_input: int, n_actions: int, hidden: int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.type(torch.float32))
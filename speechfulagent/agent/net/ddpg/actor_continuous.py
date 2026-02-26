import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    def __init__(self, n_input: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.type(torch.float32))
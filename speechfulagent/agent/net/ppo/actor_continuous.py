import torch
import torch.nn as nn


class ContinuousPPOActor(nn.Module):
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

    def forward(self, x: torch.Tensor):
        return self.mu(x.type(torch.float32))
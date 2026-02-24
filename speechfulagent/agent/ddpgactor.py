import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_shape: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 400),
            nn.ReLU(),
<<<<<<<< HEAD:speechfulagent/agent/ddpgactor.py
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh()
========
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
>>>>>>>> research/ppo:speechfulagent/agent/actor.py
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.type(torch.float32))
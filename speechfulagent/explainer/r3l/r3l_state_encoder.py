from typing import List

import torch
import torch.nn as nn


class R3LStateEncoder(nn.Module):
    def __init__(
        self,
        modules: List[int],
        module_size: int=32,
        hidden_size: int=512,
        projected_size: int=2048,
        device: str="cpu"
    ):
        super().__init__()
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.projected_size = projected_size
        self.device = device

        self.modules_sizes = modules
        self.linear_modules = []
        for size in modules:
            self.linear_modules.append(nn.Linear(size, module_size).to(self.device))

        self.lstm = nn.LSTM(input_size=len(modules)*module_size, hidden_size=hidden_size, batch_first=True).to(self.device)
        self.norm = nn.LayerNorm(hidden_size).to(self.device)
        self.project = nn.Linear(hidden_size, projected_size).to(self.device)

    def forward(self, inputs: List[torch.Tensor]):
        processed_inputs = []
        for module, input in zip(self.linear_modules, inputs):
            processed_inputs.append(
                module.forward(input)
            )
        exp_tensor = torch.concat(processed_inputs, dim=1)
        
        lstm_out, (h_n, c_n) = self.lstm(exp_tensor)
        normed = self.norm(lstm_out)
        projected = self.project(normed)
        return projected
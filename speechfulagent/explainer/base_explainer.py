from abc import ABC, abstractmethod

import torch

from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin


class BaseExplainer(VersioningMixin, ABC):
    def __init__(self):
        pass

    def _apply_top_k(self, logits: torch.Tensor, k: int=0) -> torch.Tensor:
        if k == 0:
            return logits
        top_logits, top_ids = torch.topk(logits, k, dim=-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[top_ids] = False
        return torch.masked_fill(logits, mask=mask, value=float("-inf"))
    
    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / temperature
    
    def _greedy_sampling(self, logits: torch.Tensor) -> int:
        val, ind = torch.max(logits, dim=0)
        return int(ind.item())
    
    def _random_sampling(self, logits: torch.Tensor) -> int:
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    
    @abstractmethod
    def generate(
        self,
        prompt: torch.Tensor,
        context: torch.Tensor,
        max_length: int=32,
        temperature: float=0.0,
        top_k: int=0
    ) -> str:
        """prompt is a tensor of a sequence, that is need to be explained

        context is a vector of context (experiences) before the prompt
        """
        pass
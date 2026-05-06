from abc import ABC, abstractmethod
from typing import List, Any

import torch

from speechfulagent.dataclasses import Experience
from speechfulagent.versioning import VersioningMixin


class BaseExplainer(VersioningMixin, ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: List[Experience],
        context: List[Experience] | torch.Tensor | Any,
        max_tokens: int=32,
        temperature: float=0.0,
        top_k: int=0,
        *args,
        **kwargs
    ) -> str:
        """prompt is a tensor of a sequence, that is need to be explained

        context is all additional imformation for prompt
        """
        pass
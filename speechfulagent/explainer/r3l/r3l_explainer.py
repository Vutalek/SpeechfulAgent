from typing import Optional, List, Callable, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from speechfulagent.dataclasses import Experience
from speechfulagent.explainer.base_explainer import BaseExplainer
from speechfulagent.explainer.r3l.r3l_state_encoder import R3LStateEncoder
from speechfulagent.versioning import VersioningMixin


class R3LExplainer(BaseExplainer, VersioningMixin):
    def __init__(self):
        super().__init__()

        self.encoder: Optional[R3LStateEncoder] = None
        self.transformer: Optional[PreTrainedModel] = None

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
    
    def _default_collation_fn(
        self,
        prompt: List[Experience],
        context: List[Experience] | torch.Tensor | Any
    ) -> List[List[torch.Tensor]]:
        result = []
        for exp in prompt:
            tensors = []

            state = torch.as_tensor([exp.state], dtype=torch.long)
            if isinstance(exp.state, int):
                state = F.one_hot(state, self.encoder.modules_sizes[0])
            tensors.append(state)

            action = torch.as_tensor([exp.action], dtype=torch.long)
            if isinstance(exp.action, int):
                action = F.one_hot(action, self.encoder.modules_sizes[1])
            tensors.append(action)

            reward = torch.as_tensor([exp.reward], dtype=torch.float32)
            tensors.append(reward)

            result.append(tensors)
        return result
    
    def generate_with_loss(
        self,
        prompt: List[Experience],
        context: List[Experience] | torch.Tensor | Any,
        startage: torch.Tensor,
        endage: torch.Tensor,
        startas: torch.Tensor,
        ground_truth: torch.LongTensor,
        collation_fn: Optional[Callable]
    ) -> torch.Tensor:
        if self.transformer is None or self.encoder is None:
            raise RuntimeError("Model not loaded!")
        
        if collation_fn is None:
            collation_fn = self._default_collation_fn
        
        input_embeds = self.encoder(
            collation_fn(prompt, context)
        )
        
        vector_ground_truth = self.transformer.model.embed_tokens(
            ground_truth.to(self.transformer.device)
        )
        
        llm_input = torch.cat([
            startage.squeeze(0), 
            input_embeds.squeeze(0),
            endage.squeeze(0),
            startas.squeeze(0),
            vector_ground_truth.squeeze(0) # Teacher Forcing
        ], dim=0).unsqueeze(0)

        ignore_index = -100
        labels = torch.full((llm_input.shape[0], llm_input.shape[1]), ignore_index, dtype=torch.long, device=ground_truth.device)
        prefix_len = startage.shape[1] + input_embeds.shape[1] + endage.shape[1] + startas.shape[1]
        labels[0, prefix_len:] = ground_truth

        outputs = self.transformer.forward(
            inputs_embeds=llm_input.to(torch.bfloat16), 
            labels=labels,
            use_cache=False
        )

        return outputs.loss
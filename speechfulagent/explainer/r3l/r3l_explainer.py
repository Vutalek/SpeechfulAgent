from typing import Dict, Optional, List, Callable, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer

from speechfulagent.dataclasses import Experience, R3LTrainInfo
from speechfulagent.explainer.base_explainer import BaseExplainer
from speechfulagent.explainer.r3l.r3l_state_encoder import R3LStateEncoder
from speechfulagent.versioning import VersioningMixin


class R3LExplainer(BaseExplainer, VersioningMixin):
    def __init__(self):
        super().__init__()

        self.encoder: Optional[R3LStateEncoder] = None
        self.transformer: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

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
        ground_truth: torch.LongTensor,
        think_start: Optional[List[int]]=None,
        think_end: Optional[List[int]]=None,
        text_start: List[int]=[0],
        collation_fn: Optional[Callable]=None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if self.transformer is None or self.encoder is None:
            raise RuntimeError("Model not loaded!")
        
        if collation_fn is None:
            collation_fn = self._default_collation_fn
        
        input_embeds = self.encoder(
            collation_fn(prompt, context)
        )
        
        vector_think_start = None
        vector_think_end = None
        with torch.no_grad():
            vector_ground_truth = self.transformer.model.embed_tokens(
                ground_truth.to(self.transformer.device)
            )
            if think_start is not None and think_end is not None:
                vector_think_start = self.transformer.model.embed_tokens(
                    torch.as_tensor(think_start, dtype=torch.long).to(self.transformer.device)
                )
                vector_think_end = self.transformer.model.embed_tokens(
                    torch.as_tensor(think_end, dtype=torch.long).to(self.transformer.device)
                )
            vector_text_start = self.transformer.model.embed_tokens(
                torch.as_tensor(text_start, dtype=torch.long).to(self.transformer.device)
            )
        
        if vector_think_start is not None and vector_think_end is not None:
            llm_input = torch.cat([
                vector_think_start.squeeze(0), 
                input_embeds.squeeze(0),
                vector_think_end.squeeze(0),
                vector_text_start.squeeze(0),
                vector_ground_truth.squeeze(0) # Teacher Forcing
            ], dim=0).unsqueeze(0)
        else:
            llm_input = torch.cat([
                vector_text_start.squeeze(0),
                input_embeds.squeeze(0),
                vector_ground_truth.squeeze(0) # Teacher Forcing
            ], dim=0).unsqueeze(0)

        ignore_index = -100
        labels = torch.full((llm_input.shape[0], llm_input.shape[1]), ignore_index, dtype=torch.long, device=ground_truth.device)
        if think_start is not None and think_end is not None:
            prefix_len = len(think_start) + input_embeds.shape[1] + len(think_end) + len(text_start)
        else:
            prefix_len = len(text_start) + input_embeds.shape[1]
        labels[0, prefix_len:] = ground_truth

        outputs = self.transformer.forward(
            inputs_embeds=llm_input.to(torch.bfloat16), 
            labels=labels,
            use_cache=False
        )

        return outputs.loss
    
    def generate(
        self,
        prompt: List[Experience],
        context: List[Experience] | torch.Tensor | Any,
        max_tokens: int=32,
        temperature: float=0.0,
        top_k: int=0,
        think_start: Optional[List[int]]=None,
        think_end: Optional[List[int]]=None,
        text_start: List[int]=[0],
        text_end: List[int]=[1],
        collation_fn: Optional[Callable]=None,
        *args,
        **kwargs
    ) -> str:
        if self.transformer is None or self.encoder is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded!")
        
        if collation_fn is None:
            collation_fn = self._default_collation_fn
        
        input_embeds = self.encoder(
            collation_fn(prompt, context)
        )

        vector_think_start = None
        vector_think_end = None
        with torch.no_grad():
            if think_start is not None and think_end is not None:
                vector_think_start = self.transformer.model.embed_tokens(
                    torch.as_tensor(think_start, dtype=torch.long).to(self.transformer.device)
                )
                vector_think_end = self.transformer.model.embed_tokens(
                    torch.as_tensor(think_end, dtype=torch.long).to(self.transformer.device)
                )
            vector_text_start = self.transformer.model.embed_tokens(
                torch.as_tensor(text_start, dtype=torch.long).to(self.transformer.device)
            )
        
        if vector_think_start is not None and vector_think_end is not None:
            llm_input = torch.cat([
                vector_think_start.squeeze(0), 
                input_embeds.squeeze(0),
                vector_think_end.squeeze(0),
                vector_text_start.squeeze(0),
            ], dim=0).unsqueeze(0)
        else:
            llm_input = torch.cat([
                vector_text_start.squeeze(0),
                input_embeds.squeeze(0),
            ], dim=0).unsqueeze(0)

        tokens = []
        cache = None
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.transformer.forward(
                    inputs_embeds=llm_input.to(torch.bfloat16),
                    cache=cache,
                    use_cache=True,
                )
                next_token_logits = outputs[0, -1, -1, :]
                cache = outputs.past_key_values

                if temperature == 0.0:
                    next_token = self._greedy_sampling(next_token_logits)
                else:
                    next_token_logits = self._apply_temperature(next_token_logits, temperature)
                    next_token_logits = self._apply_top_k(next_token_logits, top_k)
                    next_token = self._random_sampling(next_token_logits)
                tokens.append(next_token)
                input_embeds = torch.cat([
                    input_embeds.squeeze(0),
                    self.transformer.model.embed_tokens(
                        torch.as_tensor([[next_token]]).squeeze(0)
                    )
                ], dim=0).unsqueeze(0)
        result = self.tokenizer.decode(tokens)
        if isinstance(result, str):
            return result
        else:
            return '\n\n'.join(result)
        
    def _save_model(self, path: str, version: str, *args, **kwargs) -> Dict[str, Any]:
        torch.save(self.encoder.state_dict(), path + '/' + "encoder.pth")
        train: R3LTrainInfo = kwargs.get("train")
        info = {
            "version": version,
            "n_params": sum(p.numel() for p in self.encoder.parameters()),
            "training": train.dict()
        }
        return info
    
    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        self.encoder = R3LStateEncoder(
            modules=data["training"]["se_modules"],
            module_size=data["training"]["se_module_size"],
            hidden_size=data["training"]["se_hidden_size"],
            projected_size=data["training"]["se_projected_size"]
        )
        self.encoder.load_state_dict(torch.load(path + '/' + "encoder.pth"))
        self.tokenizer = AutoTokenizer.from_pretrained(data["training"]["tokenizer_name"])
        self.transformer = AutoModelForCausalLM.from_pretrained(data["training"]["model_name"]).to(data["training"]["llm_device"])
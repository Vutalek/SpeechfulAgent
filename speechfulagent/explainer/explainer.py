from typing import Any, Dict
import torch

from .preprocessing import Tokenizer
from .transformer import ExplainerTransformer
from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin


class Explainer(VersioningMixin):
    def __init__(self):
        self.tokenizer: Tokenizer = None
        self.transformer: ExplainerTransformer = None

    def _apply_top_k(self, logits: torch.Tensor, k: int = 0) -> torch.Tensor:
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
    
    def generate(
            self,
            tail: torch.Tensor,
            sequence: torch.Tensor,
            max_length: int = 32,
            temperature: float = 0.0,
            top_k: int = 0
    ) -> str:
        if self.tokenizer is None or self.transformer is None:
            raise RuntimeError("Model not loaded!")
        
        tokens = torch.IntTensor([[self.tokenizer.special_tokens["<BOS>"]]])
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.transformer(tail, sequence, tokens)
                next_token_logits = outputs[0, -1, :]

                if temperature == 0.0:
                    next_token = self._greedy_sampling(next_token_logits)
                else:
                    next_token_logits = self._apply_temperature(next_token_logits, temperature)
                    next_token_logits = self._apply_top_k(next_token_logits, top_k)
                    next_token = self._random_sampling(next_token_logits)
                tokens = torch.cat([tokens, torch.IntTensor([[next_token]])], dim=-1)

                if next_token == self.tokenizer.special_tokens["<EOS>"]:
                    break
        return self.tokenizer.decode(tokens[0, :].tolist())
    
    def _save_model(self, path: str, version: str, *args, **kwargs):
        torch.save(
            {
                "model_state_dict": self.transformer.state_dict(),
                "tokenizer_vocab": self.tokenizer.vocab
            },
            path + "/weights.dat"
        )
        train: ExplainerTrainInfo = kwargs.get("train")
        info = {
            "version": version,
            "n_params": sum(p.numel() for p in self.transformer.parameters()),
            "hyperparameters": self.transformer.hyperparameters,
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        params = torch.load(path + '/' + "weights.dat")
        self.tokenizer = Tokenizer()
        self.tokenizer.set_vocab(params["tokenizer_vocab"])
        self.transformer = ExplainerTransformer(**(data["hyperparameters"]))
        self.transformer.load_state_dict(params["model_state_dict"])
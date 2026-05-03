import json
from typing import List, Any

import torch
import openai

from speechfulagent.dataclasses import Experience
from speechfulagent.explainer import BaseExplainer


class OpenaiExplainer(BaseExplainer):
    def __init__(
        self,
        prompt_path: str,
        api_key: str,
        project: str,
        base_url: str,
        model: str
    ):
        super().__init__()

        self.client = openai.OpenAI(
            api_key=api_key,
            project=project,
            base_url=base_url
        )
        self.model = model
        with open(prompt_path, "rt", encoding="utf-8") as f:
            self.prompt = f.read()

    def _episode_to_json(self, episode: List[Experience]) -> str:
        dicts = [exp.dict() for exp in episode]
        last_exp = {
            "state": dicts[-1]["next_state"],
            "action": None,
            "reward": None,
            "done": True
        }
        for d in dicts:
            del d["next_state"]
        dicts.append(last_exp)
        return json.dumps(dicts)
    
    def generate(
        self,
        prompt: List[Experience],
        context: List[Experience] | torch.Tensor | Any,
        max_tokens: int=32,
        temperature: float=0.0,
        top_k: int=0
    ) -> str:
        """prompt is a tensor of a sequence, that is need to be explained

        context is all additional imformation for prompt
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self._episode_to_json(prompt)}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        if completion.choices[0].message.content is not None:
            return completion.choices[0].message.content
        else:
            return ""
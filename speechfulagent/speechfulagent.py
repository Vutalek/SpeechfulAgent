import random
from typing import List, Tuple

import gymnasium as gym

from .agent import Agent
from .explainer import Explainer, embed_sequence
from speechfulagent.dataclasses import *


class SpeechfulAgent:
    def __init__(
        self,
        agent_dir: str='.',
        explainer_dir: str='.',
        agent_version: str="latest",
        explainer_version: str="latest",
        max_sequence_length:int = 10,
        frequency: int=10,
        max_tokens: int=32,
        temperature: float=0.5,
        top_k: int=10
    ):
        if agent_version is None:
            self.agent = None
        else:
            self.agent = Agent()
            self.agent.load_model(agent_dir, agent_version)
        self.episode: List[Experience] = []
        
        if explainer_version is None:
            self.explainer = None
        else:
            self.explainer = Explainer()
            self.explainer.load_model(explainer_dir, explainer_version)
        self.explanations: List[str] = []

        self.max_sequence_length = max_sequence_length
        self.frequency = frequency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k

        self.env: gym.Env = None

    def set_environment(self, env: gym.Env):
        """Initializes env and agent."""
        self.agent.reset()
        state, _ = env.reset()
        self.agent.init_state(state)
        self.env = env
        self.episode = []
        self.explanations = []

    def run(self, need_print: bool=True) -> Tuple[List[Experience], List[str], float]:
        i = 1
        while True:
            exp = self.agent.step(self.env)
            self.episode.append(exp)
            if i % self.frequency == 0:
                tail = random.randint(1, self.max_sequence_length)
                seq_embed, tail_embed = embed_sequence(self.episode[-self.max_sequence_length:], tail, self.max_sequence_length)
                explanation = self.explainer.generate(
                    tail_embed.unsqueeze(0),
                    seq_embed.unsqueeze(0),
                    self.max_tokens,
                    self.temperature,
                    self.top_k
                )
                explanation = f"{tail+1} actions: " + explanation
                self.explanations.append(explanation)
                if need_print:
                    print(explanation)
            if exp.done:
                break
            i += 1
        return self.episode, self.explanations, self.agent.total_reward
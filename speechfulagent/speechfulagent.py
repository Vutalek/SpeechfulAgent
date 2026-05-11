from typing import List, Tuple, Optional

from speechfulagent.dataclasses import Experience
from speechfulagent.agent import BaseAgent
from speechfulagent.explainer import BaseExplainer


class SpeechfulAgent:
    def __init__(
        self,
        frequency: int=10,
        max_tokens: int=32,
        temperature: float=0.5,
        top_k: int=10
    ):
        self.agent: Optional[BaseAgent] = None
        self.explainer: Optional[BaseExplainer] = None
        
        self.episode: List[Experience]
        self.explanations: List[str] = []

        self.frequency = frequency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k

    def set_agent(self, agent: BaseAgent):
        self.agent = agent

    def set_explainer(self, explainer: BaseExplainer):
        self.explainer = explainer

    def reset(self):
        if self.agent is None:
            raise RuntimeError("Agent is None: Nothing to reset")
        
        self.agent.reset()
        self.episode = []
        self.explanations = []

    def run(self, need_print: bool=True, *args, **kwargs) -> Tuple[List[Experience], List[str], float]:
        if self.agent is None or self.explainer is None:
            raise RuntimeError("Model not loaded!")
        
        i = 1
        while True:
            exp = self.agent.step()
            self.episode.append(exp)
            if self.frequency != 0 and i % self.frequency == 0:
                explanation = self.explainer.generate(
                    self.episode,
                    None,
                    self.max_tokens,
                    self.temperature,
                    self.top_k
                )
                explanation = f"{len(self.episode)} actions: " + explanation
                self.explanations.append(explanation)
                if need_print:
                    print(explanation)
            if exp.done:
                break
            i += 1
        return self.episode, self.explanations, self.agent.total_reward
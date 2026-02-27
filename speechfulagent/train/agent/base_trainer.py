from abc import ABC, abstractmethod
from typing import Tuple

from speechfulagent.agent import BaseAgent
from speechfulagent.dataclasses import EnvInfo, BaseTrainInfo


class BaseTrainer(ABC):
    def __init__(self, seed: int):
        self.seed = seed
    
    @abstractmethod
    def train(self) -> Tuple[BaseAgent, EnvInfo, BaseTrainInfo]:
        pass
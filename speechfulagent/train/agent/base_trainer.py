"""Module with base abstract class for agent trainer."""

from abc import ABC, abstractmethod
from typing import Tuple

from speechfulagent.agent import BaseAgent
from speechfulagent.dataclasses import EnvInfo, BaseTrainInfo


class BaseTrainer(ABC):
    """Abstract trainer."""
    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def train(self) -> Tuple[BaseAgent, EnvInfo, BaseTrainInfo]:
        """Main train procedure. Starts the training cycle."""

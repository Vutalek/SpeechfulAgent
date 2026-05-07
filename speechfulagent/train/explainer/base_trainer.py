"""Module with base abstract class for explainer trainer."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from speechfulagent.explainer import BaseExplainer
from speechfulagent.dataclasses import BaseTrainInfo


class BaseExplainerTrainer(ABC):
    """Abstract trainer."""
    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def train(self) -> Tuple[BaseExplainer, BaseTrainInfo, Dict[str, Any]]:
        """Main train procedure. Starts the training cycle."""

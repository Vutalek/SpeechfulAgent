from abc import ABC, abstractmethod

import numpy as np
import torch
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin


class BaseAgent(VersioningMixin, ABC):
    def __init__(
        self, 
        env: gym.Env,
        seed: int=70
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = env
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.is_obs_cont = False
            self.obs_n = int(env.observation_space.n)
        else:
            self.is_obs_cont = True
            self.obs_shape = env.observation_space.shape
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_act_cont = False
            self.act_n = int(env.action_space.n)
        else:
            self.is_act_cont = True
            self.act_shape = env.action_space.shape
            
        self.env_state = None
        self.total_reward = 0.0

        self.training = False

    def train(self):
        """Sets agent to training mode

        may include exploration mechanisms
        """
        self.training = True

    def eval(self):
        """Sets agent to evaluation mode"""
        self.training = False

    def reset(self):
        """resets environment, state and total reward of agent"""
        self.env_state, _ = self.env.reset(seed=self.seed)
        self.total_reward = 0.0

    @torch.no_grad()
    def step(self) -> Experience:
        """Agent's step in environment.

        Firstly, agent must be initialized with reset() method.

        method _step() must be implemented.
        """
        return self._step()
    
    @abstractmethod
    def _step(self) -> Experience:
        pass
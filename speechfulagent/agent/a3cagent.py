from typing import Dict, Any

import numpy as np
import torch
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin
from .a3cnet import A2C


class Agent(VersioningMixin):
    """A2C agent. Only works in environments with both discrete action
    and observation spaces."""
    def __init__(self):
        self.env_state = None
        self.total_reward = 0.0
        self.net = None

    def reset(self):
        # resets the state and total reward of agent
        self.env_state = None
        self.total_reward = 0.0

    def init_state(self, state: State):
        # initializes first state
        self.env_state = state

    def _ohe(self, x, size):
        enc = np.zeros(size)
        enc[x] = 1
        return enc
    
    @torch.no_grad()
    def step(self, env: gym.Env) -> Experience:
        """Agent's step in environment.

        Firstly, agent must be initialized with init_state() method.
        """
        # checking if model is loaded
        if self.net is None:
            raise RuntimeError("Model not loaded!")
        # checking if state is not None
        if self.env_state is None:
            raise RuntimeError("Uninitialized environment!")

        state_t = torch.as_tensor(self._ohe(self.env_state, env.observation_space.n))
        # state_t = torch.as_tensor(self.env_state)
        state_t.unsqueeze_(0)
        policy, _ = self.net(state_t)
        probs = torch.softmax(policy, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        
        next_state, reward, is_done, is_trunc, _ = env.step(action)
        self.total_reward += float(reward)

        old_state = self.env_state
        self.env_state = next_state

        done = is_done or is_trunc
        exp = Experience(
            old_state,
            action,
            float(reward),
            next_state,
            done
        )
        return exp
    
    def _save_model(self, path: str, version: str, *args, **kwargs) -> Dict[str, Any]:
        torch.save(self.net.state_dict(), path + '/' + 'weights.dat')
        env: EnvInfo = kwargs.get("env")
        train: AgentTrainInfo = kwargs.get("train")
        info = {
            "version": version,
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        state_dict = torch.load(path + '/' + "weights.dat")
        self.net = A2C(data["environment"]["n_observations"], data["environment"]["n_actions"])
        self.net.load_state_dict(state_dict)

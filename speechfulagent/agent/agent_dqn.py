from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """DQN agent. Only works in environments with discrete action spaces."""
    def __init__(self, env: gym.Env, seed: int=70):
        super().__init__(env, seed)
        self.net: Optional[torch.nn.Module] = None
        self.epsilon = 0.0
    
    def _step(self) -> Experience:
        # checking if model is loaded
        if self.net is None:
            raise RuntimeError("Model not loaded!")
        # checking if state is not None
        if self.env_state is None:
            raise RuntimeError("Uninitialized environment!")
        # check validity of env
        if self.is_act_cont:
            raise RuntimeError("DQN is not suitable for environments with continuous actions space.")
        
        # exploration
        if self.training and np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            if self.is_obs_cont:
                state = torch.as_tensor(self.env_state)
            else:
                state = F.one_hot(torch.as_tensor(self.env_state), self.obs_n)
            state.unsqueeze_(0)
            q_values = self.net(state)
            _, act_idx = torch.max(q_values, dim=1)
            action = int(act_idx.item())
        
        next_state, reward, is_done, is_trunc, _ = self.env.step(action)
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
        # checking if model is not None
        if self.net is None:
            raise RuntimeError("Nothing to save!")
        torch.save(self.net, path + '/' + 'weights.dat')
        env: EnvInfo = kwargs.get("env", EnvInfo)
        train: DQNTrainInfo = kwargs.get("train", DQNTrainInfo)
        info = {
            "version": version,
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        self.net = torch.load(path + '/' + "weights.dat")

    def load_model(self, net: torch.nn.Module):
        self.net = net
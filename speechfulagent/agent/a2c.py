from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from .base_agent import BaseAgent


class A2CAgent(BaseAgent):
    """A2C agent"""
    def __init__(self, env: gym.Env, seed: int=70):
        super().__init__(env, seed)
        self.net: Optional[torch.nn.Module] = None
    
    def _step(self) -> Experience:
        # checking if model is loaded
        if self.net is None:
            raise RuntimeError("Model not loaded!")
        # checking if state is not None
        if self.env_state is None:
            raise RuntimeError("Uninitialized environment!")

        if self.is_obs_cont:
            state = torch.as_tensor(self.env_state)
        else:
            state = F.one_hot(torch.as_tensor(self.env_state), self.obs_n)
        state.unsqueeze_(0)

        if self.is_act_cont:
            mu, var, _ = self.net(state)
            mu = mu.data.numpy()
            sigma = torch.sqrt(var).data.numpy()
            action = np.random.normal(mu, sigma)
            action = np.clip(action, -1, 1).squeeze()
        else:
            policy, _ = self.net(state)
            probs = torch.softmax(policy, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        
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
        torch.save(self.net, path + '/' + 'model.pth')
        env: EnvInfo = kwargs.get("env", EnvInfo)
        train: A3CTrainInfo = kwargs.get("train", A3CTrainInfo)
        info = {
            "version": version,
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        self.net = torch.load(path + '/' + "model.pth", weights_only=False)

    def set_model(self, net: torch.nn.Module):
        """Sets local model for agent"""
        self.net = net
